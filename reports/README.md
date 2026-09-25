# The memory/time report

`reports/membench.pdf` is the deliverable: tables of step time, peak memory, optimizer
state and the ratios against each baseline, for both halves of the benchmark in one
document. It is built to be read while the sweeps are still running.

```bash
reports/refresh.sh          # ask both sweeps for their points, rebuild the PDF
```

That submits an `export` job to each sweep, waits, keeps the `PT` lines in
`results/{moe,dense}-points.tsv`, and runs the generator and xelatex. The workspace
disk is not reachable from a workstation, so asking each sweep to print its own table
is the only way to read one out.

Reading a partial report:

- **section 1** says how many points of each model exist out of how many are planned;
- `---` is a point that has not been measured yet;
- grey `OOM` is a point that ran out of memory, which is a result and not a gap;
- **влезает** in a ratio table means the baseline did not fit and the variant did.

To rebuild from the TSVs without touching the cloud:

```bash
python3 reports/make_membench_pdf.py reports/membench.tex
xelatex -output-directory=reports reports/membench.tex
```
