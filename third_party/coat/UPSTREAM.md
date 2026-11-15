# NVlabs/COAT, vendored subset

- Source: https://github.com/NVlabs/COAT
- Commit: 80ec99f47aaa09231b07ace1fd04c30a1e30ec18 (2025-08-08, "Update pyproject.toml")
- Files are byte-identical to that commit; `stage3_moe` adapts them without editing them.
- License: Apache-2.0 (LICENSE).

A local copy at `fineweb-streaming/Efficient-Training/third_party/coat` was NOT used: its
`optimizer/fp8_adamw.py` differs from upstream (its `fp8_adamw_expand_cuda_kernel.cu` matches).
