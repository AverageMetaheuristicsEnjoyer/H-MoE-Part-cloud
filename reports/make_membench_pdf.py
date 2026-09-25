#!/usr/bin/env python3
"""Tables for the memory/time benchmark, both halves in one document. Compile with xelatex.

    python reports/make_membench_pdf.py reports/membench.tex
    xelatex -output-directory=reports reports/membench.tex

Reads the two `export` outputs -- the MoE sweep in this repository and the dense sweep
in efficient-training-membench-cloud -- which carry different columns because they come
from different harnesses, and normalizes them onto one record.

The document is built to be read while the sweeps are still running: a point that has
not been measured is a dash, a point that ran out of memory is grey, and section 1
says how much of each table exists.
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

# Dense parameter counts are geometry, not measurement, so the first table can be built
# from a sweep that has not reached every model yet.
DENSE_MODELS = [
    ("257m", "257M", 257_188_864),
    ("500m", "0.5B", 494_516_480),
    ("1p4b", "1.44B", 1_439_270_912),
    ("3p5b", "3.48B", 3_480_136_704),
    ("6p9b", "6.89B", 6_888_361_984),
]
MOE_MODELS = [("1p029b", "MoE 1.03B"), ("2p094b", "MoE 2.09B"), ("3p599b", "MoE 3.60B")]
MODELS = [(name, label, "dense") for name, label, _ in DENSE_MODELS] + \
         [(name, label, "moe") for name, label in MOE_MODELS]

PRECISION_LABELS = [
    ("bf16_state_fp32", "bf16"),
    ("fp8gemm_state_fp32", "FP8 GEMM"),
    ("bf16_state_fp8", "FP8-состояния"),
    ("fp8gemm_state_fp8", "FP8 GEMM + состояния"),
]
DENSE_VARIANTS = [
    (f"{optimizer}_{suffix}", f"{name}, {label}")
    for optimizer, name in (("adamw", "AdamW"), ("muon", "Muon"), ("soap", "SOAP"))
    for suffix, label in PRECISION_LABELS
]
# The MoE arms are AdamW and Muon only, and they never combine the two FP8 axes.
MOE_VARIANTS = [
    (name, label) for name, label in DENSE_VARIANTS
    if not name.startswith("soap_") and not name.endswith("fp8gemm_state_fp8")
]

# (baseline, treatment, label) -- the contrasts the project is about.
CONTRASTS = [
    ("bf16_state_fp32", "bf16_state_fp8", "FP8-состояния"),
    ("bf16_state_fp32", "fp8gemm_state_fp32", "FP8 GEMM"),
    ("bf16_state_fp32", "fp8gemm_state_fp8", "обе оси"),
]
OPTIMIZERS = [("adamw", "AdamW"), ("muon", "Muon"), ("soap", "SOAP")]

OOM = r"\textcolor{oomgray}{OOM}"
MISSING = "---"

MOE_COLUMNS = ("model", "variant", "batch", "status", "peak_gb", "reserved_gb",
               "state_gb", "step_ms", "opt_ms", "tokens_per_second")
DENSE_COLUMNS = ("model", "variant", "batch", "status", "step_ms", "forward_ms",
                 "backward_ms", "opt_ms", "tokens_per_second", "peak_gb",
                 "reserved_gb", "state_gb", "params_gb")


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


points = read_points(MOE_POINTS, MOE_COLUMNS)
points.update(read_points(DENSE_POINTS, DENSE_COLUMNS))


def variants_for(family):
    return MOE_VARIANTS if family == "moe" else DENSE_VARIANTS


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


def per_batch(model, family, key, digits):
    return [
        (label, [cell(model, variant, batch, key, digits) for batch in BATCHES])
        for variant, label in variants_for(family)
    ]


def ratio(model, family, optimizer, baseline_suffix, treatment_suffix, batch, key):
    """Baseline over treatment: above 1 means the treatment costs less, or is faster."""
    known = {variant for variant, _ in variants_for(family)}
    base_name = f"{optimizer}_{baseline_suffix}"
    treat_name = f"{optimizer}_{treatment_suffix}"
    if base_name not in known or treat_name not in known:
        return None
    base = points.get((model, base_name, batch))
    treat = points.get((model, treat_name, batch))
    if base is None or treat is None:
        return MISSING
    if base["status"] == "oom" or treat["status"] == "oom":
        # A treatment that fits where the baseline does not has no ratio, and is the
        # most interesting cell on the page; say which side hit the wall.
        return OOM if treat["status"] == "oom" else r"\textbf{влезает}"
    if base["status"] != "complete" or treat["status"] != "complete":
        return MISSING
    if not float(treat[key]):
        return MISSING
    return f"{float(base[key]) / float(treat[key]):.3f}"


def ratio_rows(model, family, key):
    rows = []
    for optimizer, optimizer_label in OPTIMIZERS:
        for baseline, treatment, contrast_label in CONTRASTS:
            cells = [ratio(model, family, optimizer, baseline, treatment, batch, key)
                     for batch in BATCHES]
            if any(value is None for value in cells):
                continue
            if all(value == MISSING for value in cells):
                continue
            rows.append((f"{optimizer_label}: {contrast_label}", cells))
    return rows


body = []

# --- 1. what exists ---------------------------------------------------------------
body.append(r"\section{Покрытие: сколько точек измерено}")
body.append(r"Точка --- одна модель, один вариант, один микробатч. OOM засчитывается "
            r"как измерение: это результат, а не пропуск.")
coverage_rows = []
for name, label, family in MODELS:
    total = len(variants_for(family)) * len(BATCHES)
    done = sum(
        1 for variant, _ in variants_for(family) for batch in BATCHES
        if (record := points.get((name, variant, batch))) is not None
        and record["status"] in ("complete", "oom")
    )
    oom = sum(
        1 for variant, _ in variants_for(family) for batch in BATCHES
        if (record := points.get((name, variant, batch))) is not None
        and record["status"] == "oom"
    )
    coverage_rows.append((label, [str(done), str(total), f"{100 * done / total:.0f}", str(oom)]))
body += centered(render(coverage_rows, ["измерено", "всего", r"\%", "из них OOM"],
                        corner="Модель"))

# --- 2. the models ----------------------------------------------------------------
body.append(r"\section{Модели}")
model_rows = [
    ("Всего параметров, млрд",
     [f"{count / 1e9:.3f}" for _, _, count in DENSE_MODELS]
     + [f"{MODEL_SHAPES[name].total / 1e9:.3f}" for name, _ in MOE_MODELS]),
    ("Активных на токен, млрд",
     [f"{count / 1e9:.3f}" for _, _, count in DENSE_MODELS]
     + [f"{MODEL_SHAPES[name].active / 1e9:.3f}" for name, _ in MOE_MODELS]),
    ("Всего / активных",
     ["1.00"] * len(DENSE_MODELS)
     + [f"{MODEL_SHAPES[name].total / MODEL_SHAPES[name].active:.2f}" for name, _ in MOE_MODELS]),
    ("Роутед-экспертов",
     [MISSING] * len(DENSE_MODELS)
     + [str(MODEL_SHAPES[name].routed_experts) for name, _ in MOE_MODELS]),
    ("Активация банка, \\%",
     [MISSING] * len(DENSE_MODELS)
     + [f"{100 * 9 / (MODEL_SHAPES[name].routed_experts + 1):.2f}" for name, _ in MOE_MODELS]),
]
body += centered(render(model_rows, [label for _, label, _ in MODELS], corner="Параметры"))

# --- 3. time ----------------------------------------------------------------------
body.append(r"\section{Полный шаг обучения (fwd + bwd + optimizer), мс}")
for name, label, family in MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, family, "step_ms", 1),
                            [f"mb {batch}" for batch in BATCHES]))

# --- 4. memory --------------------------------------------------------------------
body.append(r"\section{Пиковая память, ГБ}")
for name, label, family in MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, family, "peak_gb", 2),
                            [f"mb {batch}" for batch in BATCHES]))

# --- 5. optimizer state, which does not depend on the batch ------------------------
body.append(r"\section{Состояние оптимизатора, ГБ}")
body.append(r"От микробатча не зависит; взято с наименьшего измеренного. "
            r"Плотные и MoE в отдельных таблицах: девять колонок с этими подписями "
            r"не помещаются в полосу.")
for group, group_label in ((MODELS[:len(DENSE_MODELS)], "Плотные"),
                           (MODELS[len(DENSE_MODELS):], "MoE")):
    state_rows = []
    for variant, label in DENSE_VARIANTS:
        cells = []
        for name, _, family in group:
            if variant not in {item for item, _ in variants_for(family)}:
                cells.append(MISSING)
                continue
            value = MISSING
            for batch in BATCHES:
                candidate = cell(name, variant, batch, "state_gb", 3)
                if candidate not in (MISSING, OOM):
                    value = candidate
                    break
            cells.append(value)
        state_rows.append((label, cells))
    if all(value == MISSING for _, cells in state_rows for value in cells):
        continue
    body += [r"\subsection{" + group_label + "}"]
    body += centered(render(state_rows, [label for _, label, _ in group]))

# --- 6. optimizer step time -------------------------------------------------------
body.append(r"\section{Только шаг оптимизатора, мс}")
for name, label, family in MODELS:
    body += [r"\subsection{" + label + "}"]
    body += centered(render(per_batch(name, family, "opt_ms", 2),
                            [f"mb {batch}" for batch in BATCHES]))

# --- 7 and 8. ratios --------------------------------------------------------------
body.append(r"\section{Отношение к базовой линии: память (база / вариант)}")
body.append(r"Больше 1 --- вариант занимает меньше. Порог гейта --- 1.10. "
            r"\textbf{влезает} означает, что база не поместилась, а вариант поместился.")
for name, label, family in MODELS:
    rows = ratio_rows(name, family, "peak_gb")
    if not rows:
        continue
    body += [r"\subsection{" + label + "}"]
    body += centered(render(rows, [f"mb {batch}" for batch in BATCHES]))

body.append(r"\section{Отношение к базовой линии: время шага (база / вариант)}")
body.append(r"Больше 1 --- вариант быстрее.")
for name, label, family in MODELS:
    rows = ratio_rows(name, family, "step_ms")
    if not rows:
        continue
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
{\large H100 80GB HBM3 --- плотные: seq 1024, 16\,384 токена на шаг оптимизатора;
MoE: seq 2048, 32\,768 токенов}\\[2pt]
{\normalsize Число токенов на шаг фиксировано, accumulation меняется вместе с mb.
Плотные: FP32-веса, bf16 autocast. Собрано REVISION}
\end{center}
\vspace{6pt}
BODY
\end{document}
"""
import datetime  # noqa: E402

OUT.write_text(
    document.replace("BODY", "\n".join(body))
    .replace("REVISION", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
)
measured = sum(1 for record in points.values() if record["status"] in ("complete", "oom"))
print(f"wrote {OUT} from {measured} measured points")
