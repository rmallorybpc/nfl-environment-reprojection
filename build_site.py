"""
Build the deployable pages: embed the TMG stylesheet, the model data, and the
pre-rendered fact chips into the templates; write index.html and audit.html.

Run order:
  1. python build_model.py    # pulls nflverse pbp, writes model.json
  2. python build_site.py     # writes index.html and audit.html

Both pages are self-contained: the TMG design system is inlined and the tool's
key numbers (fact chips) are rendered into the HTML at build time, so they are
visible to crawlers, scrapers, and readers without JavaScript.

To use the shared TMG stylesheet instead of the inlined copy, replace the
<style id="tmg-system"> block in each template with the CDN <link> noted there;
this script then leaves the (absent) placeholder alone.
"""
import json, pathlib

tmg = pathlib.Path("tmg.css").read_text()
model_text = pathlib.Path("model.json").read_text()
model = json.loads(model_text)


def facts_html():
    L, M = model["league"], model["meta"]
    pct = lambda x: f"{x*100:.1f}"
    facts = [
        ("League dome effect", f"+{L['d_league']*100:.1f} pts"),
        ("Indoor / outdoor", f"<b>{pct(L['c_in'])}</b> / <b>{pct(L['c_out'])}</b>"),
        ("Between-QB spread", f"<b>{L['sd_between']*100:.1f}</b> pts (Q={L['Q']}, df={L['Q_df']})"),
        ("Roster", f"<b>{M['n_players']}</b> QBs"),
        ("Effect basis", "road games only"),
        ("Min road games / env", f"<b>{M['min_games_per_env']}</b> (10+ att)"),
        ("Window", f"<b>{M['season_start']}&ndash;{M['season_end']}</b> reg. season"),
        ("Source", "nflverse play-by-play"),
    ]
    return "".join(f'<span class="fact">{k} &nbsp;{v}</span>' for k, v in facts)


def build(template_path, out_path, with_model):
    html = pathlib.Path(template_path).read_text()
    if "__TMG_CSS__" in html:
        html = html.replace("__TMG_CSS__", tmg)
    if "__FACTS_HTML__" in html:
        html = html.replace("__FACTS_HTML__", facts_html())
    if with_model:
        assert "__MODEL_JSON__" in html, f"{template_path} missing data placeholder"
        html = html.replace("__MODEL_JSON__", model_text)
    pathlib.Path(out_path).write_text(html)
    print(f"wrote {out_path}")


build("tool_template.html", "index.html", with_model=True)
build("audit_template.html", "audit.html", with_model=False)
