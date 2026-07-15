"""
NFL environment reprojection: model layer.

Reads nflverse play-by-play, computes each QB's dome/outdoor completion effect,
the league environment effect, and an empirical-Bayes shrinkage of each QB's
own effect toward the league mean. Emits a compact JSON the front end consumes.

Method summary
- Environment per game: indoor = dome or retractable-closed. outdoor = open-air
  or retractable-open. Classified by the game's roof, not the QB's home team.
- Qualifying game: the QB threw at least MIN_ATT_PER_GAME attempts. This is the
  games-started proxy.
- ROAD-ONLY EFFECT: the environment split (indoor minus outdoor) is measured on
  AWAY games only. A quarterback's home games are all one environment, so pooling
  home and away lets home-field advantage ride along inside the environment
  number. Restricting the split to away games holds home field out of both sides,
  isolating the environment effect. (This mirrors Sports Info Solutions.)
- TRUE ANCHOR: each quarterback's baseline completion percentage and his real
  environment mix are taken from ALL games, so the tool still shows his real
  career number. Only the slope of the counterfactual uses the clean road-only
  effect.
- Roster filter: keep QBs with at least MIN_GAMES_PER_ENV qualifying AWAY games
  in BOTH environments, so the clean effect is estimable.
- Shrinkage: d_hat = w * d_player + (1 - w) * d_league, reliability weight
  w = var_between / (var_between + var_split), var_between by DerSimonian-Laird.
"""

import glob, json
import numpy as np
import pandas as pd

# Tunable parameters. Change these and rerun.
MIN_ATT_PER_GAME = 10     # attempts needed to count a game as a start
MIN_GAMES_PER_ENV = 8     # qualifying AWAY games needed in EACH environment
ROAD_ONLY = True          # measure the environment effect on away games only
SEASON_START = 1999
SEASON_END = 2024
KEEP_COLS = ["season", "game_id", "roof", "home_team", "away_team", "posteam",
             "passer_player_id", "passer_player_name", "pass_attempt",
             "complete_pass", "sack", "season_type"]

INDOOR = {"dome", "closed"}
OUTDOOR = {"outdoors", "open"}


def load_pergame():
    """One row per (passer, game, env, home/away) with attempts and completions."""
    rows = []
    for fn in sorted(glob.glob("pbp/pbp_*.parquet")):
        df = pd.read_parquet(fn, columns=KEEP_COLS)
        df = df[(df["season_type"] == "REG") &
                (df["pass_attempt"] == 1) &
                (df["sack"] != 1) &
                (df["passer_player_id"].notna())].copy()
        df["env"] = df["roof"].map(
            lambda r: "indoor" if r in INDOOR else ("outdoor" if r in OUTDOOR else None))
        df = df[df["env"].notna()]
        df["ha"] = np.where(df["posteam"] == df["home_team"], "home", "away")
        g = (df.groupby(["passer_player_id", "passer_player_name",
                         "game_id", "env", "ha"], observed=True)
               .agg(att=("pass_attempt", "sum"),
                    comp=("complete_pass", "sum")).reset_index())
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def env_totals(frame):
    """Return wide per-passer totals by environment: att, comp, games."""
    by = (frame.groupby(["passer_player_id", "env"], observed=True)
               .agg(att=("att", "sum"), comp=("comp", "sum"),
                    games=("game_id", "nunique")).reset_index())
    wide = by.pivot(index="passer_player_id", columns="env",
                    values=["att", "comp", "games"]).fillna(0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    for c in ["att_indoor", "att_outdoor", "comp_indoor", "comp_outdoor",
              "games_indoor", "games_outdoor"]:
        if c not in wide:
            wide[c] = 0.0
    return wide


def main():
    pergame = load_pergame()
    pergame = pergame[pergame["att"] >= MIN_ATT_PER_GAME].copy()   # qualifying games

    road = pergame[pergame["ha"] == "away"] if ROAD_ONLY else pergame

    allw = env_totals(pergame)     # anchor: all games
    rdw = env_totals(road)         # effect: road games

    names = (pergame.sort_values("game_id")
                    .groupby("passer_player_id")["passer_player_name"].last())

    # League effect: pooled completion pct by env over road qualifying games.
    ti = road[road.env == "indoor"]["att"].sum()
    ci = road[road.env == "indoor"]["comp"].sum()
    to = road[road.env == "outdoor"]["att"].sum()
    co = road[road.env == "outdoor"]["comp"].sum()
    c_in_league = ci / ti
    c_out_league = co / to
    d_league = c_in_league - c_out_league

    # Roster: two-sided minimum on qualifying road games.
    keep = rdw[(rdw.games_indoor >= MIN_GAMES_PER_ENV) &
               (rdw.games_outdoor >= MIN_GAMES_PER_ENV)].index
    r = rdw.loc[keep].copy()

    # Effect (road games)
    r["c_in"] = r.comp_indoor / r.att_indoor
    r["c_out"] = r.comp_outdoor / r.att_outdoor
    r["d_obs"] = r.c_in - r.c_out
    r["var_split"] = (r.c_in * (1 - r.c_in) / r.att_indoor +
                      r.c_out * (1 - r.c_out) / r.att_outdoor)
    r["rd_in"] = r.games_indoor.astype(int)
    r["rd_out"] = r.games_outdoor.astype(int)

    # Anchor (all games)
    a = allw.loc[keep]
    att_all = a.att_indoor + a.att_outdoor
    comp_all = a.comp_indoor + a.comp_outdoor
    r["c_all"] = comp_all / att_all
    r["p_in"] = a.att_indoor / att_all
    r["att_all"] = att_all
    r["se_c_all"] = np.sqrt(r.c_all * (1 - r.c_all) / att_all)
    r["games_in"] = a.games_indoor.astype(int)
    r["games_out"] = a.games_outdoor.astype(int)

    # Empirical Bayes: between-QB variance by DerSimonian-Laird.
    d = r["d_obs"].values
    vs = r["var_split"].values
    wq = 1.0 / vs
    mu = np.sum(wq * d) / np.sum(wq)
    Q = float(np.sum(wq * (d - mu) ** 2))
    k = len(d)
    c_dl = np.sum(wq) - np.sum(wq ** 2) / np.sum(wq)
    var_between = max(0.0, (Q - (k - 1)) / c_dl)

    r["w"] = var_between / (var_between + r.var_split)
    r["d_hat"] = r.w * r.d_obs + (1 - r.w) * d_league
    r["se_dhat"] = np.sqrt(r.w * r.var_split)

    r = r.sort_values("att_all", ascending=False)
    players = []
    for pid, row in r.iterrows():
        players.append({
            "id": pid, "name": names.get(pid, pid),
            "c_all": round(float(row.c_all), 5),
            "p_in": round(float(row.p_in), 4),
            "c_in_obs": round(float(row.c_in), 5),
            "c_out_obs": round(float(row.c_out), 5),
            "d_obs": round(float(row.d_obs), 5),
            "d_hat": round(float(row.d_hat), 5),
            "se_dhat": round(float(row.se_dhat), 5),
            "se_c_all": round(float(row.se_c_all), 5),
            "w": round(float(row.w), 3),
            "games_in": int(row.games_in), "games_out": int(row.games_out),
            "rd_in": int(row.rd_in), "rd_out": int(row.rd_out),
        })

    out = {
        "meta": {
            "season_start": SEASON_START, "season_end": SEASON_END,
            "min_att_per_game": MIN_ATT_PER_GAME,
            "min_games_per_env": MIN_GAMES_PER_ENV,
            "road_only": ROAD_ONLY, "season_type": "regular season",
            "n_players": len(players),
        },
        "league": {
            "c_in": round(float(c_in_league), 5),
            "c_out": round(float(c_out_league), 5),
            "d_league": round(float(d_league), 5),
            "var_between": float(var_between),
            "sd_between": round(float(np.sqrt(var_between)), 5),
            "Q": round(Q, 1), "Q_df": int(k - 1),
        },
        "players": players,
    }
    with open("model.json", "w") as f:
        json.dump(out, f, indent=1)

    print(f"road_only={ROAD_ONLY}  players={len(players)}")
    print(f"league (road) indoor {c_in_league:.4f} outdoor {c_out_league:.4f} "
          f"delta {d_league*100:+.2f} pts")
    print(f"sd_between {np.sqrt(var_between)*100:.2f} pts  Q={Q:.1f} df={k-1}")
    print("\nexamples:")
    for nm in ["T.Brady", "D.Brees", "P.Manning", "M.Ryan", "A.Rodgers"]:
        p = next((x for x in players if x["name"] == nm), None)
        if p:
            print(f"  {nm:>12}  career {p['c_all']*100:.1f}%  d_hat {p['d_hat']*100:+.2f}  "
                  f"w {p['w']:.2f}  road gms {p['rd_in']}/{p['rd_out']}  "
                  f"(career {p['games_in']}/{p['games_out']})")


if __name__ == "__main__":
    main()
