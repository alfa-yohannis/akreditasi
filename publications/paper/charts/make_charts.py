"""Draw the paper's result charts from the files in evaluation_results/.

    ..\\..\\..\\.venv\\Scripts\\python.exe make_charts.py [--png DIR]

Writes vector PDFs to ../figures/ (results-endtoend, results-ablation,
results-models); --png also writes previews. Needs matplotlib, which the app
itself does not (``pip install matplotlib``).

Style: a colour-blind-safe categorical palette, validated with the dataviz
validator (worst adjacent CVD dE 9.2, normal-vision dE 27.6 on white); one
colour per entity throughout; hairline recessive grids; a white gap between
touching bars instead of outlines; text in ink colours, never series colours;
values labelled selectively, the full numbers being in the paper's tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "evaluation_results"
FIGURES = HERE.parent / "figures"

BLUE, ORANGE, AQUA = "#2A78D6", "#EB6834", "#1BAF7A"
MUTED, GRID, AXIS = "#C3C2B7", "#E1E0D9", "#898781"
INK, INK2 = "#0B0B0B", "#52514E"
COLUMN, PAGE = 3.5, 7.16  # IEEE column and text widths, inches

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 2,
    "ytick.major.pad": 3,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,  # embed TrueType, as IEEE requires
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BAR = dict(edgecolor="white", linewidth=0.8)  # the white gap between fills


def latest(prefix: str) -> dict:
    """The newest result file whose name starts with prefix."""
    paths = sorted(RESULTS.glob(f"{prefix}_*.json"))
    if not paths:
        raise SystemExit(f"no {prefix}_*.json in {RESULTS}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def recessive(axis, grid_axis: str = "x") -> None:
    """Hairline grid behind the marks; only the baseline spine stays."""
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    if grid_axis == "x":
        axis.spines["left"].set_color(MUTED)
        axis.spines["bottom"].set_visible(False)
    else:
        axis.spines["bottom"].set_color(MUTED)
        axis.spines["left"].set_visible(False)


def label_inside(axis, bar, text: str, colour: str = "white") -> None:
    """Write text inside a horizontal bar segment only if it fits."""
    x, y = bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2
    note = axis.text(x, y, text, ha="center", va="center", color=colour, fontsize=6.5)
    renderer = axis.figure.canvas.get_renderer()
    text_width = note.get_window_extent(renderer).width
    bar_width = bar.get_window_extent(renderer).width
    if text_width + 4 > bar_width:
        note.remove()


def save(figure, name: str, png_directory: Path | None) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    if png_directory:
        png_directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_directory / f"{name}.png", dpi=300)
    plt.close(figure)
    print(f"wrote figures/{name}.pdf")


# --- Fig.: end to end -------------------------------------------------------

def end_to_end_chart(png: Path | None) -> None:
    rows = [("As deployed", latest("endtoend_current")), ("Improved", latest("endtoend_best"))]
    segments = []
    for _, run in rows:
        timed = [s for s in run["llm_samples"] if not s["warmup"] and s["done_ms"]]
        n = len(timed)
        cards = sum(s["cards_ms"] for s in timed) / n / 1000
        reading = sum(s["first_token_ms"] - s["cards_ms"] for s in timed) / n / 1000
        writing = sum(s["done_ms"] - s["first_token_ms"] for s in timed) / n / 1000
        segments.append((cards, reading, writing))

    figure, axis = plt.subplots(figsize=(COLUMN, 1.25))
    names = ["search and re-scoring", "reading the prompt", "writing the answer"]
    colours = [BLUE, ORANGE, AQUA]
    positions = list(range(len(rows)))
    # Fix the scale first, so the fit test for in-bar labels measures real widths.
    axis.set_xlim(0, max(sum(segment) for segment in segments) * 1.13)
    axis.set_ylim(len(rows) - 0.5, -0.5)
    left = [0.0] * len(rows)
    for index, (name, colour) in enumerate(zip(names, colours)):
        widths = [segment[index] for segment in segments]
        bars = axis.barh(positions, widths, left=left, height=0.52, color=colour, label=name, **BAR)
        for bar, width in zip(bars, widths):
            label_inside(axis, bar, f"{width:.1f}")
        left = [l + w for l, w in zip(left, widths)]
    for position, total in zip(positions, left):
        axis.text(total + max(left) * 0.012, position, f"{total:.1f} s", va="center",
                  ha="left", color=INK, fontsize=7, fontweight="bold")
    axis.set_yticks(positions, [name for name, _ in rows])
    axis.set_xlabel("seconds from the question (mean of 50 answers)", color=INK2)
    recessive(axis)
    axis.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, handlelength=1.0,
                handleheight=0.8, columnspacing=1.0, borderaxespad=0.2)
    save(figure, "results-endtoend", png)


# --- Fig.: one change at a time ------------------------------------------------

def ablation_chart(png: Path | None) -> None:
    configurations = [
        ("As deployed", "answering_answer-qwen3-8b-default"),
        ("8-bit key-value cache", "answering_answer-qwen3-8b-kv-q8"),
        ("Encoder in half precision", "answering_answer-qwen3-8b-embed-fp16"),
        ("Encoder on the processor", "answering_answer-qwen3-8b-embed-cpu"),
        ("... and 8-bit cache", "answering_answer-qwen3-8b-tuned"),
    ]
    chosen = "Encoder on the processor"
    runs = [(name, latest(prefix)) for name, prefix in configurations]
    names = [name for name, _ in runs]
    own = [run["gpu_memory"]["with_model"]["dedicated_mib"] / 1024 for _, run in runs]
    lent = [run["gpu_memory"]["with_model"]["shared_mib"] / 1024 for _, run in runs]
    reading = [run["llm_summary"]["prompt_tokens_per_s"]["median"] for _, run in runs]
    answer = [run["llm_summary"]["total_ms"]["median"] / 1000 for _, run in runs]
    positions = list(range(len(runs)))
    emphasis = [BLUE if name == chosen else MUTED for name in names]

    figure, axes = plt.subplots(1, 3, figsize=(PAGE, 1.75), sharey=True,
                                gridspec_kw={"wspace": 0.12})
    memory, speed, time = axes

    memory.barh(positions, own, height=0.55, color=MUTED, label="card's own", **BAR)
    memory.barh(positions, lent, left=own, height=0.55, color=ORANGE, label="lent by Windows", **BAR)
    memory.axvline(8, color=INK, linewidth=0.7)
    # Room above the first bar for the line's label, clear of the title.
    memory.set_ylim(len(positions) - 0.45, -1.05)
    memory.text(8.12, -0.78, "8 GB card", color=INK, fontsize=6.5, va="center", ha="left")
    memory.set_xlim(0, 10.4)
    memory.set_xticks([0, 2, 4, 6, 8, 10])
    memory.set_title("GPU memory (GB)", loc="left")
    memory.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=2,
                  handlelength=1.0, handleheight=0.8, columnspacing=1.0)

    speed.barh(positions, reading, height=0.55, color=emphasis, **BAR)
    speed.set_xlim(0, 3000)
    speed.set_xticks([0, 1000, 2000, 3000], ["0", "1,000", "2,000", "3,000"])
    speed.set_title("Prompt reading (tokens/s)", loc="left")

    time.barh(positions, answer, height=0.55, color=emphasis, **BAR)
    time.set_xlim(0, 40)
    time.set_title("Complete answer (s)", loc="left")

    for axis, values, fmt in ((speed, reading, "{:,.0f}"), (time, answer, "{:.1f}")):
        span = axis.get_xlim()[1]
        for index in (0, names.index(chosen)):  # as deployed, and the fix
            axis.text(values[index] + span * 0.015, index, fmt.format(values[index]),
                      va="center", ha="left", color=INK, fontsize=6.5)
    for axis in axes:
        recessive(axis)
    memory.set_yticks(positions, names)
    save(figure, "results-ablation", png)


# --- Fig.: the two answering models --------------------------------------------

def models_chart(png: Path | None) -> None:
    large, small = latest("answering_answer-qwen3-8b-embed-cpu"), latest("answering_answer-qwen3.5-4b-tuned")
    review = json.loads((RESULTS / "review_q12.json").read_text(encoding="utf-8"))["runs"]

    def wrong(prefix: str) -> tuple[int, int]:
        picked = [v for k, v in review.items() if k.startswith(prefix)]
        return sum(v["wrong_codes"] for v in picked), sum(v["answers"] for v in picked)

    models = [("Qwen3 8B", large, wrong("answer-qwen3-8b"), BLUE),
              ("Qwen3.5 4B", small, wrong("answer-qwen3.5-4b"), ORANGE)]
    figure, axes = plt.subplots(1, 3, figsize=(COLUMN, 1.55),
                                gridspec_kw={"width_ratios": [2, 1, 1], "wspace": 0.55})
    seconds, length, codes = axes
    width = 0.36

    for offset, (name, run, _, colour) in zip((-width / 2, width / 2), models):
        values = [run["llm_summary"]["first_token_ms"]["median"] / 1000,
                  run["llm_summary"]["total_ms"]["median"] / 1000]
        bars = seconds.bar([0 + offset, 1 + offset], values, width=width, color=colour, label=name, **BAR)
        for bar, value in zip(bars, values):
            seconds.text(bar.get_x() + bar.get_width() / 2, value + 0.1, f"{value:.1f}",
                         ha="center", va="bottom", color=INK, fontsize=6.5)
    seconds.set_xticks([0, 1], ["first word", "complete"])
    seconds.set_ylim(0, 5.6)
    seconds.set_title("Seconds (median)", loc="left")

    for offset, (name, run, _, colour) in zip((-width / 2, width / 2), models):
        value = run["llm_summary"]["output_tokens"]["median"]
        length.bar(offset, value, width=width, color=colour, **BAR)
        length.text(offset, value + 6, f"{value:.0f}", ha="center", va="bottom", color=INK, fontsize=6.5)
    length.set_ylim(0, 320)
    length.set_title("Tokens", loc="left")

    for offset, (name, _, (bad, total), colour) in zip((-width / 2, width / 2), models):
        share = 100 * bad / total
        codes.bar(offset, share, width=width, color=colour, **BAR)
        codes.text(offset, share + 2.5, f"{bad}/{total}", ha="center", va="bottom", color=INK, fontsize=6.5)
    codes.set_ylim(0, 105)
    codes.set_title("Wrong codes", loc="left")
    codes.set_ylabel("% of Q12 answers", color=INK2, labelpad=1)

    for axis in (length, codes):
        axis.set_xticks([])
        axis.set_xlim(-0.6, 0.6)
    for axis in axes:
        recessive(axis, grid_axis="y")
    figure.legend(*seconds.get_legend_handles_labels(), loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, 1.0), handlelength=1.0, handleheight=0.8)
    save(figure, "results-models", png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--png", type=Path, default=None, help="also write PNG previews here")
    parser.add_argument("charts", nargs="*", default=["endtoend", "ablation", "models"])
    arguments = parser.parse_args()
    for chart in arguments.charts:
        {"endtoend": end_to_end_chart, "ablation": ablation_chart, "models": models_chart}[chart](arguments.png)


if __name__ == "__main__":
    main()
