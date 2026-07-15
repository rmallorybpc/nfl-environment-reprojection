"""
build_site.py — Step 2
=======================
Inlines tmg.css and model.json into the HTML templates to produce
self-contained, deployment-ready index.html and audit.html.

Usage
-----
    python build_site.py

Must be run after build_model.py (requires model.json to exist).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent

TEMPLATE_MAP = {
    ROOT / "tool_template.html":  ROOT / "index.html",
    ROOT / "audit_template.html": ROOT / "audit.html",
}

CSS_PLACEHOLDER   = "/* {{TMG_CSS}} */"
MODEL_PLACEHOLDER = "{{MODEL_JSON}}"


def main() -> None:
    # ── Load shared assets ─────────────────────────────────────────────────────
    css_path   = ROOT / "tmg.css"
    model_path = ROOT / "model.json"

    if not css_path.exists():
        _die(f"tmg.css not found at {css_path}. Run build_site.py from the repo root.")

    if not model_path.exists():
        _die(f"model.json not found at {model_path}. Run build_model.py first.")

    css_text   = css_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")

    # Validate model.json is valid JSON
    try:
        json.loads(model_text)
    except json.JSONDecodeError as exc:
        _die(f"model.json is not valid JSON: {exc}")

    # Minify model JSON for inlining (remove extraneous whitespace)
    model_inline = json.dumps(json.loads(model_text), separators=(",", ":"))

    # ── Process each template ──────────────────────────────────────────────────
    for template_path, output_path in TEMPLATE_MAP.items():
        if not template_path.exists():
            print(f"  WARNING: template {template_path.name} not found — skipping.")
            continue

        print(f"Building {output_path.name} from {template_path.name}…")
        html = template_path.read_text(encoding="utf-8")

        # Inline CSS (replace the placeholder comment inside <style>)
        if CSS_PLACEHOLDER in html:
            html = html.replace(CSS_PLACEHOLDER, css_text)
        else:
            print(f"  WARNING: CSS placeholder not found in {template_path.name}.")

        # Inline model JSON
        if MODEL_PLACEHOLDER in html:
            html = html.replace(MODEL_PLACEHOLDER, model_inline)
        else:
            print(f"  WARNING: model placeholder not found in {template_path.name}.")

        output_path.write_text(html, encoding="utf-8")
        size_kb = output_path.stat().st_size / 1024
        print(f"  Written: {output_path.name} ({size_kb:.1f} KB)")

    print("Done.")


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
