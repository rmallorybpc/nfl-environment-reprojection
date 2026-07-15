# NFL environment reprojection

An interactive tool that re-projects an NFL quarterback's career completion
percentage as if his environment mix had been different, dome versus outdoor,
built on the documented league environment effect and shown with error bars.

The point of the tool is honesty about size. A dome lifts completion percentage
by about two and a half points across the league, the swings for any one
quarterback are small, and the uncertainty band is usually as wide as the swing.
The band is drawn as the hero: it narrows at the quarterback's real environment
mix and flares toward the hypothetical extremes.

- **Tool:** `index.html`
- **Audit of the decisions and their weaknesses:** `audit.html`

## Method, in one paragraph

Each quarterback's own dome-minus-outdoor split is measured on road games only,
so home-field advantage does not contaminate it, then blended toward the league
average and weighted by sample size (empirical Bayes, DerSimonian-Laird). A
quarterback with many road games in both settings keeps most of his own effect;
a thin road sample is pulled toward the league. The anchor is his real career
completion percentage across all games, so the projection passes through his true
career number at his real environment mix. See `audit.html` for the honest limits,
including the weather-not-roof caveat.

## Build

Data is public: nflverse play-by-play, 1999-2024 regular season.

```bash
pip install pandas pyarrow requests
python build_model.py     # downloads pbp, writes model.json
python build_site.py      # writes index.html and audit.html
```

Tunable constants live at the top of `build_model.py` (season window, minimum
attempts per game, minimum games per environment). Change them and rerun both
steps.

## Deploy

Pages is deployed from GitHub Actions, not from a branch. In
**Settings > Pages**, set the source to **GitHub Actions**. The workflow in
`.github/workflows/deploy.yml` publishes the repo root on every push to `main`.

## Files

| File | Purpose |
|---|---|
| `index.html` | The interactive tool (self-contained, data and styles inlined) |
| `audit.html` | Audit of the build decisions |
| `build_model.py` | Computes splits, league effect, and shrinkage from nflverse data |
| `build_site.py` | Inlines the stylesheet and data into the templates |
| `tool_template.html`, `audit_template.html` | Source templates |
| `tmg.css` | TMG design system stylesheet |
| `model.json` | Generated model output consumed by the tool |
