# Building the memory/time tables

The two sweeps run in different repositories and print their points with an `export`
mode, because `mlsub logs` keeps only a tail and the workspace disk is not otherwise
reachable. Paste each into `results/`, then build:

```bash
ssh brain_lab mlsub run --repo .../H-MoE-Part-cloud --branch bench/fp8-membench \
  --entry scripts/cloud_membench.sh --gpus cpu --no-pip --args "export"     # -> results/moe-points.tsv
ssh brain_lab mlsub run --repo .../efficient-training-membench-cloud --branch main \
  --entry scripts/cloud_run.sh --gpus cpu --no-pip --args "export"          # -> results/dense-points.tsv

python reports/make_membench_pdf.py reports/membench.tex
xelatex -output-directory=reports reports/membench.tex
```

Only the `PT`-prefixed lines matter; the generator ignores everything else, so the raw
job output can be pasted whole. A model or variant with no points is left blank rather
than dropped, so a partial sweep still builds.
