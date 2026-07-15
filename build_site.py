"""
Build the deployable pages: embed the TMG stylesheet (and, for the tool, the
model data) into the templates and write index.html and audit.html.

Run order:
  1. python build_model.py    # pulls nflverse pbp, writes model.json
  2. python build_site.py     # writes index.html and audit.html

Both pages are self-contained: the TMG design system is inlined, so there is no
external CSS dependency. Drop them on GitHub Pages as-is, or open them locally.

To use the shared TMG stylesheet instead of the inlined copy, replace the
<style id="tmg-system"> block in each template with the CDN <link> noted there;
this script then leaves the (absent) placeholder alone.
"""
import pathlib

tmg = pathlib.Path("tmg.css").read_text()
model = pathlib.Path("model.json").read_text()

def build(template_path, out_path, with_model):
    html = pathlib.Path(template_path).read_text()
    if "__TMG_CSS__" in html:
        html = html.replace("__TMG_CSS__", tmg)
    if with_model:
        assert "__MODEL_JSON__" in html, f"{template_path} missing data placeholder"
        html = html.replace("__MODEL_JSON__", model)
    pathlib.Path(out_path).write_text(html)
    print(f"wrote {out_path}")

build("tool_template.html", "index.html", with_model=True)
build("audit_template.html", "audit.html", with_model=False)
