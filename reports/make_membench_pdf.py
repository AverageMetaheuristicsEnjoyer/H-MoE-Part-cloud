#!/usr/bin/env python3
"""Tables for the memory/time benchmark, both halves in one document. Compile with xelatex.

    python reports/make_membench_pdf.py out.tex && xelatex -output-directory=... out.tex

Reads the two `export` outputs -- the MoE sweep in this repository and the dense
sweep in efficient-training-membench-cloud -- which carry different columns because
they come from different harnesses, and normalizes them onto one record.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage3_moe import MODEL_SHAPES  # noqa: E402

MOE_POINTS = ROOT / "results/moe-points.tsv"
DENSE_POINTS = ROOT / "results/dense-points.tsv"
OUT = Path(sys.argv[1])

BATCHES = [1, 2, 4, 8, 16]

# Dense parameter counts are geometry, not measurement; they are restated here so the
# first table can be built from a sweep that did not reach every model.
DENSE_MODELS = [("257m", "257M", 257_188_864), ("500m", "0.5B", 494_516_480),
                ("1p4b", "1.44B", 1_439_270_912)]
MOE_MODELS = [("1p029b", "MoE 1.03B"), ("2p094b", "MoE 2.09B")]

DENSE_VARIANTS = [
    ("adamw_bf16_state_fp32", "AdamW, bf16"),
    ("adamw_fp8gemm_state_fp32", "AdamW, FP8 GEMM"),
    ("adamw_bf16_state_fp8", "AdamW, FP8-состояния"),
    ("muon_bf16_state_fp32", "Muon, bf16"),
    ("muon_fp8gemm_state_fp32", "Muon, FP8 GEMM"),
    ("muon_bf16_state_fp8", "Muon, FP8-состояния"),
    ("soap_bf16_state_fp32", "SOAP, bf16"),
    ("soap_fp8gemm_state_fp32", "SOAP, FP8 GEMM"),
    ("soap_bf16_state_fp8", "SOAP, FP8-состояния"),
]
MOE_VARIANTS = [item for item in DENSE_VARIANTS if not item[0].startswith("soap_")]

# (baseline, treatment, label) -- the three ratios the project is actually about.
CONTRASTS = [
    ("adamw_bf16_state_fp32", "adamw_bf16_state_fp8", "AdamW: FP8-состояния"),
    ("muon_bf16_state_fp32", "muon_bf16_state_fp8", "Muon: FP8-состояния"),
    ("soap_bf16_state_fp32", "soap_bf16_state_fp8", "SOAP: FP8-состояния"),
    ("adamw_bf16_state_fp32", "adamw_fp8gemm_state_fp32", "AdamW: FP8 GEMM"),
    ("muon_bf16_state_fp32", "muon_fp8gemm_state_fp32", "Muon: FP8 GEMM"),
    ("soap_bf16_state_fp32", "soap_fp8gemm_state_fp32", "SOAP: FP8 GEMM"),
]

OOM = r"\textcolor{oomgray}{OOM}"
MISSING = "---"


def read_points(path, columns):
    """`PT`-prefixed rows, as the cloud entry scripts print them."""
    points = {}
    if not path.is_file():
        print(f"warning: {path} is absent; its tables will be empty", file=sys.stderr)
        return points
    for line in path.read_text().splitlines():
        if not line.startswith("PT\t"):
            continue
        fields = line.split("\t")[1:]
        if fields[0] == columns[0]:  # the header row repeats the column names
            continue
        record = dict(zip(columns, fields))
        points[(record["model"], record["variant"], int(record["batch"]))] = record
    return points


MOE_COLUMNS = ("model", "variant", "batch", "status", "peak_gb", "reserved_gb",
               "state_gb", "step_ms", "opt_ms", "tokens_per_second")
DENSE_COLUMNS = ("model", "variant", "batch", "status", "step_ms", "forward_ms",
                 "backward_ms", "opt_ms", "tokens_per_second", "peak_gb",
                 "reserved_gb", "state_gb", "params_gb")

points = read_points(MOE_POINTS, MOE_COLUMNS)
points.update(read_points(DENSE_POINTS, DENSE_COLUMNS))


def cell(model, variant, batch, key, digits):
    record = points.get((model, variant, batch))
    if record is None:
        return MISSING
    if record["status"] == "oom":
        return OOM
    if record["status"] != "complete" or not record.get(key):
        return MISSING
    return f"{float(record[key]):.{digits}f}"


def render(rows, columns, corner="Метод"):
    head = f"\\textbf{{{corner}}}" + "".join(f" & \\textbf{{{column}}}" for column in columns)
    lines = [r"\small", r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    for label, cells in rows:
        lines.append(label + "".join(f" & {value}" for value in cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def centered(table):
    return [r"\begin{center}", table, r"\end{center}"]


def per_batch(model, variants, key, digits):
    return [
        (label, [cell(model, variant, batch, key, digits) for batch in BATCHES])
        for variant, label in variants
    ]


def ratio(model, baseline, treatment, batch, key):
    """Baseline over treatment: above 1 means the treatment costs less."""
    base = points.get((model, baseline, batch))
    treat = points.get((model, treatment, batch))
    if base is None or treat is None:
        return MISSING
    if base["status"] == "oom" or treat["status"] == "oom":
        # A treatment that fits where the baseline does not is the interesting case,
        # and it has no ratio; say which side ran out rather than printing a dash.
        return OOM if treat["status"] == "oom" else r"\textbf{---}"
    if base["status"] != "complete" or treat["status"] != "complete":
        return MISSING
    if not float(treat[key]):
        return MISSING
    return f"{float(base[key]) / float(treat[key]):.3f}"


body = []

body.append(r"\section{Модели}")
model_rows = [
    ("Плотная, всего параметров",
     [f"{count / 1e9:.3f}" for _, _, count in DENSE_MODELS]
     + [f"{MODEL_SHAPES[name].total / 1e9:.3f}" for name, _ in MOE_MODELS]),
    ("Активных на токен",
     [f"{count / 1e9:.3f}" for _, _, count in DENSE_MODELS]
     + [f"{MODEL_SHAPES[name].active / 1e9:.3f}" for name, _ in MOE_MODELS]),
    ("Всего / активных",
     ["1.00"] * len(DENSE_MODELS)
     + [f"{MODEL_SHAPES[name].total / MODEL_SHAPES[name].active:.2f}" for name, _ in MOE_MODELS]),
]
body += centered(render(model_rows,
                        [label for _, label, _ in DENSE_MODELS] + [label for _, label in MOE_MODELS],
                        corner="Параметры, млрд"))

body.append(r"\section{Полный шаг обучения (fwd + bwd + optimizer), мс}")
for name, label, _ in DENSE_MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, DENSE_VARIANTS, "step_ms", 1),
                            [f"mb {batch}" for batch in BATCHES]))
for name, label in MOE_MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, MOE_VARIANTS, "step_ms", 1),
                            [f"mb {batch}" for batch in BATCHES]))

body.append(r"\section{Пиковая память, ГБ}")
for name, label, _ in DENSE_MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, DENSE_VARIANTS, "peak_gb", 2),
                            [f"mb {batch}" for batch in BATCHES]))
for name, label in MOE_MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, MOE_VARIANTS, "peak_gb", 2),
                            [f"mb {batch}" for batch in BATCHES]))

# Optimizer state does not depend on the batch, so it collapses to one table.
body.append(r"\section{Состояние оптимизатора, ГБ}")
state_rows = []
for variant, label in DENSE_VARIANTS:
    cells = [cell(name, variant, 1, "state_gb", 3) for name, _, _ in DENSE_MODELS]
    cells += [cell(name, variant, 1, "state_gb", 3) for name, _ in MOE_MODELS]
    state_rows.append((label, cells))
body += centered(render(state_rows,
                        [label for _, label, _ in DENSE_MODELS] + [label for _, label in MOE_MODELS]))

body.append(r"\section{Отношение к базовой линии: память (база / вариант)}")
body.append(r"Больше 1 --- вариант занимает меньше. Порог гейта --- 1.10.")
for name, label, _ in DENSE_MODELS + [(a, b, None) for a, b in MOE_MODELS]:
    variants = MOE_VARIANTS if name in dict(MOE_MODELS) else DENSE_VARIANTS
    known = {variant for variant, _ in variants}
    rows = [
        (contrast_label, [ratio(name, base, treat, batch, "peak_gb") for batch in BATCHES])
        for base, treat, contrast_label in CONTRASTS
        if base in known and treat in known
    ]
    body += [r"\subsection{" + label + "}"]
    body += centered(render(rows, [f"mb {batch}" for batch in BATCHES]))

body.append(r"\section{Отношение к базовой линии: время шага (база / вариант)}")
body.append(r"Больше 1 --- вариант быстрее.")
for name, label, _ in DENSE_MODELS + [(a, b, None) for a, b in MOE_MODELS]:
    variants = MOE_VARIANTS if name in dict(MOE_MODELS) else DENSE_VARIANTS
    known = {variant for variant, _ in variants}
    rows = [
        (contrast_label, [ratio(name, base, treat, batch, "step_ms") for batch in BATCHES])
        for base, treat, contrast_label in CONTRASTS
        if base in known and treat in known
    ]
    body += [r"\subsection{" + label + "}"]
    body += centered(render(rows, [f"mb {batch}" for batch in BATCHES]))

document = r"""\documentclass[10pt]{article}
\usepackage[a4paper,top=16mm,bottom=16mm,left=18mm,right=18mm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{titlesec}
\setmainfont{DejaVu Sans}
\definecolor{oomgray}{gray}{0.55}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.1}
\titleformat{\section}{\Large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.6em}{}
\titlespacing{\section}{0pt}{14pt}{6pt}
\titlespacing{\subsection}{0pt}{8pt}{3pt}
\begin{document}
\begin{center}
{\LARGE\bfseries Память и время: плотные модели и MoE}\\[4pt]
{\large H100 80GB HBM3 --- плотные: seq 1024, 16384 токена на шаг; MoE: seq 2048, 32768 токенов на шаг}\\[2pt]
{\normalsize Число токенов на шаг оптимизатора фиксировано, accumulation меняется вместе с mb}
\end{center}
\vspace{6pt}
BODY
\end{document}
"""
OUT.write_text(document.replace("BODY", "\n".join(body)))
print(f"wrote {OUT}")
