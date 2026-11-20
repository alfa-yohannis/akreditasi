"""Render the paper's diagrams to vector PDF, ready for \\includegraphics.

    ..\\..\\..\\.venv\\Scripts\\python.exe render_diagrams.py              # all
    ..\\..\\..\\.venv\\Scripts\\python.exe render_diagrams.py query        # one
    ..\\..\\..\\.venv\\Scripts\\python.exe render_diagrams.py --png DIR    # + PNG previews

Sources live next to this script: ``*.dot`` is Graphviz, ``*.mmd`` is Mermaid.
Each becomes ``../figures/<name>.pdf``, cropped to the drawing.

Nothing needs installing beyond the project's own Playwright: Graphviz runs as
WebAssembly (@viz-js/viz) and Mermaid as plain JavaScript inside the Chromium
that Playwright already has, and Chromium prints the SVG straight to PDF, so
the text stays vector. Both libraries come from jsDelivr at pinned versions.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"

VIZ_JS = "https://cdn.jsdelivr.net/npm/@viz-js/viz@3.31.0/dist/viz-global.js"
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"

PAGE = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  #out {{ display: inline-block; }}
  #out svg {{ display: block; }}
</style>
<script src="{VIZ_JS}"></script>
<script src="{MERMAID_JS}"></script>
</head><body><div id="out"></div></body></html>"""

#: Renders one source into #out and returns the drawing's size in CSS pixels.
RENDER = """
async ([kind, source]) => {
  const out = document.getElementById('out');
  if (kind === 'dot') {
    const viz = await Viz.instance();
    out.replaceChildren(viz.renderSVGElement(source));
    await document.fonts.ready;
    // Graphviz places each font run of an HTML label from its own width
    // estimate, which is short for bold text and for characters such as "→";
    // the next run on the line then overlaps. Re-flow every line of runs with
    // real widths. Runs further apart than a few points are separate table
    // cells and are left alone. Node lines stay centred on where they were.
    for (const group of out.querySelectorAll('g.edge, g.node')) {
      const isNode = group.classList.contains('node');
      const lines = new Map();
      for (const run of group.querySelectorAll('text')) {
        const y = run.getAttribute('y');
        if (!lines.has(y)) lines.set(y, []);
        lines.get(y).push(run);
      }
      for (const runs of lines.values()) {
        if (runs.length < 2 || runs.some(r => r.getAttribute('text-anchor') !== 'start')) continue;
        runs.sort((a, b) => parseFloat(a.getAttribute('x')) - parseFloat(b.getAttribute('x')));
        for (const run of runs) run.setAttributeNS('http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve');
        let chain = [runs[0]];
        const settle = () => {
          if (chain.length < 2) return;
          const first = parseFloat(chain[0].getAttribute('x'));
          const last = chain[chain.length - 1];
          const before = parseFloat(last.getAttribute('x')) + last.getComputedTextLength() - first;
          for (let i = 1; i < chain.length; i++) {
            const previous = chain[i - 1];
            chain[i].setAttribute('x', (parseFloat(previous.getAttribute('x')) + previous.getComputedTextLength()).toFixed(2));
          }
          if (!isNode) return;
          const after = parseFloat(last.getAttribute('x')) + last.getComputedTextLength() - first;
          const shift = (after - before) / 2;
          for (const run of chain) run.setAttribute('x', (parseFloat(run.getAttribute('x')) - shift).toFixed(2));
        };
        for (let i = 1; i < runs.length; i++) {
          const previous = runs[i - 1];
          const end = parseFloat(previous.getAttribute('x')) + previous.getComputedTextLength();
          if (parseFloat(runs[i].getAttribute('x')) - end < 3) chain.push(runs[i]);
          else { settle(); chain = [runs[i]]; }
        }
        settle();
      }
    }
    // An edge routed through an invisible waypoint arrives as two halves that
    // meet at an angle. Halves sharing a class "join-NAME" (in drawing order)
    // are replaced by one smooth curve: it leaves the tail the way the first
    // half did, and reaches the head the way the second half did, so the
    // arrowhead Graphviz drew still fits. The curve lives in the second half,
    // which also carries the label.
    const halves = new Map();
    for (const edge of out.querySelectorAll('g.edge')) {
      const name = [...edge.classList].find(c => c.startsWith('join-'));
      if (!name) continue;
      if (!halves.has(name)) halves.set(name, []);
      halves.get(name).push(edge.querySelector('path'));
    }
    const direction = (from, to) => {
      const length = Math.hypot(to.x - from.x, to.y - from.y) || 1;
      return { x: (to.x - from.x) / length, y: (to.y - from.y) / length };
    };
    for (const [first, second] of halves.values()) {
      if (!first || !second) continue;
      const start = first.getPointAtLength(0);
      const leave = direction(start, first.getPointAtLength(Math.min(2, first.getTotalLength())));
      const end = second.getPointAtLength(second.getTotalLength());
      const arrive = direction(second.getPointAtLength(Math.max(0, second.getTotalLength() - 2)), end);
      const span = Math.hypot(end.x - start.x, end.y - start.y);
      const out1 = 0.12 * span, in2 = 0.5 * span;
      second.setAttribute('d',
        `M${start.x},${start.y}` +
        `C${start.x + leave.x * out1},${start.y + leave.y * out1} ` +
        `${end.x - arrive.x * in2},${end.y - arrive.y * in2} ${end.x},${end.y}`);
      first.remove();
    }
    // Edges with class "mid" carry their label beside the middle of the line,
    // just clear of it, rather than wherever Graphviz parked it. "atNN" moves
    // the anchor to NN % of the way along the line; the side is set below.
    for (const edge of out.querySelectorAll('g.edge.mid')) {
      const path = edge.querySelector('path');
      const runs = [...edge.querySelectorAll('text')];
      if (!path || runs.length === 0) continue;
      const at = [...edge.classList].map(c => /^at([0-9]+)$/.exec(c)).find(Boolean);
      const length = path.getTotalLength() * (at ? Number(at[1]) / 100 : 0.5);
      const anchor = path.getPointAtLength(length);
      const behind = path.getPointAtLength(Math.max(0, length - 2));
      const ahead = path.getPointAtLength(Math.min(path.getTotalLength(), length + 2));
      let tx = ahead.x - behind.x, ty = ahead.y - behind.y;
      const norm = Math.hypot(tx, ty) || 1;
      tx /= norm; ty /= norm;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const run of runs) {
        const b = run.getBBox();
        x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y);
        x1 = Math.max(x1, b.x + b.width); y1 = Math.max(y1, b.y + b.height);
      }
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      // normal to the line, on the side Graphviz chose (or the other, if
      // "flip"); "north", "south", "east" or "west" pins the side on screen
      let nx = -ty, ny = tx;
      const side = edge.classList.contains('flip') ? -1 : 1;
      if (((cx - anchor.x) * nx + (cy - anchor.y) * ny) * side < 0) { nx = -nx; ny = -ny; }
      const pin = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };
      for (const [name, [px, py]] of Object.entries(pin)) {
        if (edge.classList.contains(name) && nx * px + ny * py < 0) { nx = -nx; ny = -ny; }
      }
      // distance at which the label's box just clears the line, plus a gap
      const clear = (Math.abs((x1 - x0) * nx) + Math.abs((y1 - y0) * ny)) / 2 + 2.5;
      let targetX = anchor.x + nx * clear, targetY = anchor.y + ny * clear;
      // "left", "right", "above" or "below" instead moves the label straight
      // sideways or straight up/down from the anchor, which reads better
      // beside a steep diagonal line than the perpendicular does.
      const w = x1 - x0, h = y1 - y0, gap = 2.5;
      const axis = ['left', 'right', 'above', 'below'].find(c => edge.classList.contains(c));
      if (axis === 'left' || axis === 'right') {
        const reach = w / 2 + (h / 2) * Math.abs(tx) / Math.max(Math.abs(ty), 0.05) + gap;
        targetX = anchor.x + (axis === 'left' ? -reach : reach); targetY = anchor.y;
      } else if (axis) {
        const reach = h / 2 + (w / 2) * Math.abs(ty) / Math.max(Math.abs(tx), 0.05) + gap;
        targetX = anchor.x; targetY = anchor.y + (axis === 'above' ? -reach : reach);
      }
      const dx = targetX - cx, dy = targetY - cy;
      for (const run of runs) {
        run.setAttribute('x', (parseFloat(run.getAttribute('x')) + dx).toFixed(2));
        run.setAttribute('y', (parseFloat(run.getAttribute('y')) + dy).toFixed(2));
      }
    }
  } else {
    mermaid.initialize({ startOnLoad: false });
    const { svg } = await mermaid.render('diagram', source);
    out.innerHTML = svg;
    // Mermaid sizes its SVG to the container; pin it to its own viewBox.
    const element = out.querySelector('svg');
    const box = element.viewBox.baseVal;
    element.setAttribute('width', box.width);
    element.setAttribute('height', box.height);
    element.style.maxWidth = 'none';
  }
  await document.fonts.ready;
  const rect = out.getBoundingClientRect();
  return [rect.width, rect.height];
}
"""


def sources(names: list[str]) -> list[Path]:
    found = sorted([*HERE.glob("*.dot"), *HERE.glob("*.mmd")])
    if names:
        found = [path for path in found if path.stem in names]
    return found


def render(paths: list[Path], preview_directory: Path | None) -> None:
    FIGURES.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 2400, "height": 2400}, device_scale_factor=3)
        page.set_content(PAGE, wait_until="networkidle")
        for path in paths:
            kind = "dot" if path.suffix == ".dot" else "mermaid"
            width, height = page.evaluate(RENDER, [kind, path.read_text(encoding="utf-8")])
            output = FIGURES / f"{path.stem}.pdf"
            page.pdf(
                path=str(output),
                width=f"{math.ceil(width) + 1}px",
                height=f"{math.ceil(height) + 1}px",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                page_ranges="1",
            )
            print(f"{path.name:>20} -> {output.name}  ({width * 0.75:.0f} x {height * 0.75:.0f} pt)")
            if preview_directory is not None:
                preview_directory.mkdir(parents=True, exist_ok=True)
                page.locator("#out").screenshot(path=str(preview_directory / f"{path.stem}.png"))
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="diagram names (file stems); default all")
    parser.add_argument("--png", type=Path, help="also write PNG previews to this directory")
    arguments = parser.parse_args()
    render(sources(arguments.names), arguments.png)
