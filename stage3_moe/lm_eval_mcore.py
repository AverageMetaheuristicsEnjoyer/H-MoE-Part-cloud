"""Score downstream suites with the live MCore model.

Serving lm-eval from the model MCore just loaded keeps the forward identical to the one the
run was trained and validated with -- a conversion to another runtime would put an
unmeasured difference between the arms being compared.

The training data is pre-tokenized GPT-2 (`NullTokenizer`, vocab 50257), so scoring text
needs the real GPT-2 tokenizer; its ids are the ones the model was trained on.
"""

import torch
from lm_eval.api.model import LM

from stage3_moe.downstream_artifacts import collect_downstream

# A multiple of 16 satisfies both halves of TE's rule for any batch size.
FP8_ALIGNMENT = 16


class MCoreLM(LM):
    """The lm-eval `LM` interface, backed by an MCore GPTModel."""

    def __init__(self, model, tokenizer, *, max_length=2048, batch_size=8, rank=0, world_size=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self._rank = rank
        self._world_size = world_size
        self.eot_token_id = tokenizer.eos_token_id

    # lm-eval reads these off the object.
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _forward_logprobs(self, batch):
        """batch: list of token-id lists. Returns log-softmax logits per sequence."""
        # TE refuses an FP8 GEMM unless the flattened token count divides by 8:
        # "FP8 execution requires the product of all dimensions except the last to be
        # divisible by 8". Training never trips this because it only ever feeds full
        # 2048-token sequences; scoring batches are as long as their longest member.
        # Padding sits at the end of every sequence, attention is causal, and the
        # logprobs are read only over the real continuation, so the scores are unchanged.
        width = max(len(ids) for ids in batch)
        width = -(-width // FP8_ALIGNMENT) * FP8_ALIGNMENT
        padded = torch.full((len(batch), width), self.eot_token_id, dtype=torch.long)
        for row, ids in enumerate(batch):
            padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded = padded.cuda()
        position_ids = torch.arange(width, dtype=torch.long, device=padded.device)
        position_ids = position_ids.unsqueeze(0).expand(len(batch), width)
        with torch.no_grad():
            # attention_mask None: the run trains with
            # --no-create-attention-mask-in-dataloader, so TE builds the causal mask itself.
            logits = self.model(padded, position_ids, None)
        return torch.log_softmax(logits.float(), dim=-1)

    def _encode_pair(self, context, continuation):
        if context:
            context_ids = self.tok_encode(context)
            whole_ids = self.tok_encode(context + continuation)
            # Re-tokenizing the pair together is what the model actually sees; the split
            # point is wherever the two encodings stop agreeing.
            split = len(context_ids)
            while split > 0 and whole_ids[:split] != context_ids[:split]:
                split -= 1
        else:
            whole_ids = [self.eot_token_id] + self.tok_encode(continuation)
            split = 1
        return whole_ids, split

    def loglikelihood(self, requests, **kwargs):
        encoded = []
        for index, request in enumerate(requests):
            context, continuation = request.args
            whole_ids, split = self._encode_pair(context, continuation)
            # Keep the tail: the continuation must survive truncation, the context need not.
            whole_ids = whole_ids[-(self.max_length + 1) :]
            split = max(1, split - max(0, len(whole_ids) - (self.max_length + 1)))
            encoded.append((index, whole_ids, split))

        # Length-sorted batches keep padding waste low; the original order is restored after.
        encoded.sort(key=lambda item: len(item[1]))
        results = [None] * len(requests)
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start : start + self.batch_size]
            inputs = [ids[:-1] for _, ids, _ in chunk]
            logprobs = self._forward_logprobs(inputs)
            for row, (index, whole_ids, split) in enumerate(chunk):
                targets = torch.tensor(whole_ids[split:], device=logprobs.device)
                span = logprobs[row, split - 1 : len(whole_ids) - 1, :]
                gathered = span.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                greedy = bool((span.argmax(dim=-1) == targets).all())
                results[index] = (float(gathered.sum()), greedy)
        return results

    def loglikelihood_rolling(self, requests, **kwargs):
        from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

        encoded = []
        totals = [0.0] * len(requests)
        for request_index, request in enumerate(requests):
            (text,) = request.args
            windows = map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=self.tok_encode(text),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            )
            for context, continuation in windows:
                encoded.append(
                    (request_index, context + continuation, len(context))
                )

        encoded.sort(key=lambda item: len(item[1]))
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start : start + self.batch_size]
            inputs = [ids[:-1] for _, ids, _ in chunk]
            logprobs = self._forward_logprobs(inputs)
            for row, (request_index, ids, split) in enumerate(chunk):
                targets = torch.tensor(ids[split:], device=logprobs.device)
                span = logprobs[row, split - 1 : len(ids) - 1, :]
                totals[request_index] += float(
                    span.gather(1, targets.unsqueeze(-1)).sum()
                )
        return totals

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError("the basic_v2 suite is multiple-choice only")


def run_suite(model, *, tasks, include_path, artifact_dir, batch_size=8, limit=None,
              max_length=2048, num_fewshot=None):
    """Evaluate `model`, write per-example artifacts, and return the `downstream` array."""
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lm = MCoreLM(model, tokenizer, max_length=max_length, batch_size=batch_size)
    manager = TaskManager(include_path=str(include_path))
    output = evaluator.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        task_manager=manager,
        log_samples=True,
        limit=limit,
        num_fewshot=num_fewshot,
        bootstrap_iters=0,
        verbosity="WARNING",
    )

    downstream = collect_downstream(output, artifact_dir)
    print(f"DOWNSTREAM_ARTIFACTS={artifact_dir} n_metrics={len(downstream)}", flush=True)
    for item in downstream:
        print(f"DOWNSTREAM {item['task']} {item['metric']}={item['value']:.6f}", flush=True)
    return downstream
