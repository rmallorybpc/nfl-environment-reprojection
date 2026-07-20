# NFL environment reprojection

**Live site:** https://rmallorybpc.github.io/nfl-environment-reprojection/

An interactive tool that re-projects an NFL quarterback's career completion
percentage as if his environment mix had been different, dome versus outdoor,
built on the documented league environment effect and shown with error bars.

The point of the tool is how small the effect is. A dome lifts completion percentage
by roughly two to three points across the league, the swings for any one
quarterback are small, and the uncertainty band is usually as wide as the swing.
The band is drawn as the hero: it narrows at the quarterback's real environment
mix and flares toward the hypothetical extremes.

- **Tool:** [`index.html`](https://rmallorybpc.github.io/nfl-environment-reprojection/)
- **Audit of the decisions and their weaknesses:** [`audit.html`](https://rmallorybpc.github.io/nfl-environment-reprojection/audit.html)

## Method, in one paragraph

Each quarterback's own dome-minus-outdoor split is measured on road games only,
so home-field advantage does not contaminate it, then blended toward the league
average and weighted by sample size. The blend is empirical Bayes shrinkage;
the between-quarterback variance that sets the weights is estimated with the
DerSimonian-Laird method, treating each quarterback's split as a small study. A
quarterback with many road games in both settings keeps most of his own effect;
a thin road sample is pulled toward the league. The anchor is his real career
completion percentage across all games, so the projection passes through his true
career number at his real environment mix. See `audit.html` for the limits,
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

## Data and attribution

This tool is possible because the NFL analytics community maintains free, open
data. Credit where it is due:

- **Play-by-play data:** [nflverse](https://nflverse.nflverse.com/), downloaded
  from the [nflverse-data releases](https://github.com/nflverse/nflverse-data)
  (`play_by_play_{season}.parquet`, 1999-2024). The data releases are licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which requires
  attribution; this section and the site footers provide it.
- **Data generation:** the play-by-play files are built by
  [nflfastR](https://www.nflfastr.com/) (Ben Baldwin and Sebastian Carl),
  MIT-licensed, descended from nflscrapR (Maksim Horowitz, Ron Yurko, and Sam
  Ventura).
- **Fields used:** roof, home/away teams, passer, pass attempts, completions,
  sacks, and season type. No FTN charting data or other restricted nflverse
  add-ons are used, so plain nflverse attribution applies.
- **Corroborating sources** cited on the audit page:
  [Sports Info Solutions](https://www.sportsinfosolutions.com/2020/05/06/dome-field-advantage-how-much-does-weather-affect-quarterback-play/)
  on the dome effect and schedule-driven swing, and
  [GiveMeSport](https://www.givemesport.com/how-stadium-types-affect-nfl-scoring/)
  on the ten-season indoor/outdoor gap.
- **Method precedent:** measuring the environment effect on road games only, to
  hold home-field advantage out of the split, follows the Sports Info Solutions
  approach.
- **Statistics:** empirical Bayes shrinkage, with the between-quarterback
  variance estimated by the DerSimonian-Laird random-effects method
  (DerSimonian and Laird, 1986, *Controlled Clinical Trials*).
- **Design system:** The Mallory Group brand guide; Inter typeface via Google
  Fonts.

NFL data belongs to its respective owners and is governed by their terms of
use. This project is unaffiliated with the NFL and with nflverse.

## Limits

The tool states these on its face and the audit page treats them in full: the
projection is stylized, not causal; the roof is a proxy for weather, which is
the real driver; completion percentage is the only metric; and the roster is
limited to quarterbacks with at least 8 qualifying road games in each
environment (79 QBs).

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
