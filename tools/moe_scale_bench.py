"""Where does TE FP8 start to pay off for our MoE family? Whole MCore GPT model, forward+backward, BF16 vs FP8.

The family is docs/design.md scaled by hidden size only: 64 routed experts top-8 + 1 shared, expert width d/4
(G=8), dense FFN 2.75d, first layer dense, GQA with head dim 128 and 4 query heads per KV head, QKNorm, seq 2048,
GPT-2 vocabulary with an untied head. Routing is MCore's force-load-balancing (benchmark-only), which is what a
trained loss-free router looks like. No optimizer: one step = K micro-batches, FP8 weight cache refreshed on the
first, as MCore's training loop does. Two depths are built per shape, so (T_deep - T_shallow) / (extra MoE layers)
is the cost of one MoE layer and the rest is the dense layer + embedding + LM head.

  moe_scale_bench.py HIDDEN [HIDDEN...]   env: SCALE_MB=2:4:8  SCALE_RECIPES=bf16:blockwise:...  (mlsub rejects commas)  SCALE_K=4
Prints one MOE_SCALE json line per (hidden, micro-batch, recipe, depth).
"""
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

SEQ = 2048
VOCAB = 50257
RECIPES = {
    "bf16": {},
    "blockwise": {"fp8": "e4m3", "fp8_recipe": "blockwise"},
    "delayed1024": {"fp8": "hybrid", "fp8_recipe": "delayed", "fp8_amax_history_len": 1024, "fp8_amax_compute_algo": "max"},
    "tensorwise": {"fp8": "hybrid", "fp8_recipe": "tensorwise"},
}


def shape(hidden):
    heads = hidden // 128
    return {"hidden": hidden, "expert": hidden // 4, "dense_ffn": round(2.75 * hidden / 128) * 128,
            "heads": heads, "kv_heads": max(heads // 4, 1)}


def build(hidden, layers, recipe):
    s = shape(hidden)
    config = TransformerConfig(
        num_layers=layers, hidden_size=hidden, ffn_hidden_size=s["dense_ffn"], num_attention_heads=s["heads"],
        kv_channels=128, num_query_groups=s["kv_heads"], qk_layernorm=True, normalization="RMSNorm",
        layernorm_epsilon=1e-5, add_bias_linear=False, gated_linear_unit=True, activation_func=F.silu,
        hidden_dropout=0.0, attention_dropout=0.0, init_method_std=0.006, bf16=True, params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16, gradient_accumulation_fusion=False,
        num_moe_experts=64, moe_layer_freq=[0] + [1] * (layers - 1), moe_ffn_hidden_size=s["expert"],
        moe_shared_expert_intermediate_size=s["expert"], moe_router_topk=8, moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0, moe_router_score_function="sigmoid", moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=2.5, moe_router_enable_expert_bias=True, moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall", moe_grouped_gemm=True, moe_router_force_load_balancing=True,
        **RECIPES[recipe])
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=VOCAB, max_sequence_length=SEQ,
                     pre_process=True, post_process=True, share_embeddings_and_output_weights=False,
                     position_embedding_type="rope", rotary_percent=1.0).cuda()
    return model


def run(model, micro_batch, k, steps=5, warmup=2):
    tokens = torch.randint(0, VOCAB, (micro_batch, SEQ + 1), device="cuda")
    ids, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    position = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(micro_batch, -1)
    params = [p for p in model.parameters() if p.requires_grad]

    def step():
        for p in params:
            p.grad = None
        model.set_is_first_microbatch()
        for _ in range(k):
            model(ids, position, None, labels=labels).float().mean().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2] / k, torch.cuda.max_memory_allocated() / 2**30


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    micro_batches = [int(x) for x in os.environ.get("SCALE_MB", "2:4:8").replace(",", ":").split(":")]
    recipes = os.environ.get("SCALE_RECIPES", ":".join(RECIPES)).replace(",", ":").split(":")
    k = int(os.environ.get("SCALE_K", "4"))
    print("MOE_SCALE " + json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                     "te": __import__("transformer_engine").__version__}), flush=True)
    for hidden in [int(h) for h in sys.argv[1:]]:
        depths = (2, 4) if hidden <= 5120 else (2, 3)
        for micro_batch in micro_batches:
            for recipe in recipes:
                for layers in depths:
                    record = {"hidden": hidden, "layers": layers, "moe_layers": layers - 1, "micro_batch": micro_batch,
                              "tokens": micro_batch * SEQ, "rows_per_expert": micro_batch * SEQ * 8 // 64, "recipe": recipe, "k": k}
                    model = None
                    try:
                        model = build(hidden, layers, recipe)
                        record["params"] = sum(p.numel() for p in model.parameters())
                        seconds, peak = run(model, micro_batch, k)
                        record.update(ms_per_microbatch=round(seconds * 1e3, 3), peak_gb=round(peak, 2))
                    except Exception as error:
                        record["error"] = repr(error)[:400]
                    print("MOE_SCALE " + json.dumps(record), flush=True)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    if "error" in record and "out of memory" in record["error"].lower():
                        break


if __name__ == "__main__":
    main()
