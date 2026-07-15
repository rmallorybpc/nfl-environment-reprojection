"""
build_model.py — Step 1
=======================
Pulls nflverse play-by-play data, engineers environment features, fits a
shrinkage model for completion-percentage environment effects, and writes
model.json to the repository root.

Usage
-----
    python build_model.py

Requirements
------------
    pip install nfl_data_py pandas numpy
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
FIRST_SEASON = 2016
SHRINKAGE_KAPPA = 500        # empirical-Bayes prior pseudo-count (pass attempts)
MIN_ATTEMPTS_QB = 100        # minimum seasonal attempts for QB inclusion
WIND_THRESHOLD = 15          # mph
COLD_THRESHOLD = 35          # °F

# Stadium overrides for dome fraction
# (retractable-roof stadiums get a historical fraction)
DOME_FRACTION_OVERRIDES: dict[str, float] = {
    "ATT Stadium":            0.70,   # Cowboys — retractable, usually closed
    "State Farm Stadium":     0.70,   # Cardinals — retractable
    "Allegiant Stadium":      1.00,   # Raiders — fully indoor
    "Lucas Oil Stadium":      1.00,   # Colts — fully indoor
    "NRG Stadium":            0.65,   # Texans — retractable
    "Ford Field":             1.00,   # Lions — fully indoor
    "Mercedes-Benz Stadium":  1.00,   # Falcons — retractable, typically closed
    "U.S. Bank Stadium":      1.00,   # Vikings — fully indoor
    "Caesars Superdome":      1.00,   # Saints — fully indoor
    "SoFi Stadium":           0.80,   # Rams/Chargers — retractable
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def weather_bucket(row: pd.Series) -> str:
    """Classify a row's weather into one of five buckets."""
    if row.get("roof") in ("dome", "closed"):
        return "clear"   # dome plays are always baseline

    wind  = row.get("wind", 0) or 0
    temp  = row.get("temp", 65) or 65
    precip = bool(row.get("precipitation", False))

    if precip and temp < COLD_THRESHOLD:
        return "snow"
    if precip:
        return "rain"
    if temp < COLD_THRESHOLD:
        return "cold"
    if wind >= WIND_THRESHOLD:
        return "wind"
    return "clear"


def dome_flag(row: pd.Series) -> float:
    """Return 1 if dome, 0 if outdoor, fractional for retractable."""
    roof = row.get("roof", "outdoor") or "outdoor"
    if roof in ("dome", "closed"):
        return 1.0
    if roof == "retractable":
        stadium = row.get("stadium", "") or ""
        return DOME_FRACTION_OVERRIDES.get(stadium, 0.5)
    return 0.0


def shrink(raw_effect: float, n_plays: int, grand_mean: float, kappa: float) -> float:
    """Empirical Bayes shrinkage toward grand mean."""
    weight = n_plays / (n_plays + kappa)
    return weight * raw_effect + (1 - weight) * grand_mean


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation, dropping NaN pairs."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["x"].corr(df["y"]))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading nflverse play-by-play data…")
    try:
        import nfl_data_py as nfl
        seasons = list(range(FIRST_SEASON, date.today().year + 1))
        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=True)
    except Exception as exc:
        print(f"  WARNING: could not load nflverse data ({exc}). Using empty DataFrame.")
        pbp = pd.DataFrame()

    if pbp.empty:
        print("  No data available — writing empty model.")
        _write_model({
            "built_at": date.today().isoformat(),
            "seasons": [],
            "effects": _default_effects(),
            "effects_meta": _default_effects_meta(),
            "stadiums": [],
            "validation": {},
            "changelog": [],
        })
        return

    print(f"  Loaded {len(pbp):,} plays.")

    # ── Filter to regular-season pass attempts ─────────────────────────────────
    passes = pbp[
        (pbp["play_type"] == "pass")
        & (pbp["season_type"] == "REG")
        & (pbp["qb_dropback"] == 1)
        & pbp["complete_pass"].notna()
        & pbp["passer_player_id"].notna()
    ].copy()

    print(f"  {len(passes):,} regular-season dropbacks after filtering.")

    # ── Derive environment columns ─────────────────────────────────────────────
    passes["weather"]   = passes.apply(weather_bucket, axis=1)
    passes["dome_frac"] = passes.apply(dome_flag, axis=1)

    # ── Build stadium lookup ───────────────────────────────────────────────────
    stadiums = _build_stadium_list(passes)

    # ── Estimate raw effects via group means (controlling for covariates) ───────
    print("Estimating environment effects…")
    effects_raw, effects_meta_raw = _estimate_effects(passes)

    # ── Shrink toward grand mean ───────────────────────────────────────────────
    outdoor_keys = ["wind", "rain", "cold", "snow"]
    outdoor_vals = [effects_raw[k] for k in outdoor_keys]
    grand_mean   = float(np.mean(outdoor_vals))

    effects_shrunk: dict[str, float] = {}
    effects_meta:   dict[str, dict]  = {}

    for key, raw_val in effects_raw.items():
        n = effects_meta_raw[key].get("n_plays", 0)
        shrunk = shrink(raw_val, n, grand_mean if key != "dome" else 0.0, SHRINKAGE_KAPPA)
        effects_shrunk[key] = round(shrunk, 6)
        ci_half = 1.96 * effects_meta_raw[key].get("se", 0)
        effects_meta[key] = {
            "raw_effect": round(raw_val, 6),
            "n_plays":    int(n),
            "ci_lo":      round(shrunk - ci_half, 6),
            "ci_hi":      round(shrunk + ci_half, 6),
        }

    # ── Compute year-over-year validation ──────────────────────────────────────
    print("Computing year-over-year validation…")
    validation = _yoy_validation(passes, effects_shrunk)

    # ── Write model.json ───────────────────────────────────────────────────────
    seasons_used = sorted(passes["season"].dropna().unique().tolist())
    model = {
        "built_at":    date.today().isoformat(),
        "seasons":     [int(s) for s in seasons_used],
        "effects":     effects_shrunk,
        "effects_meta": effects_meta,
        "stadiums":    stadiums,
        "validation":  validation,
        "changelog": [
            {
                "date": date.today().isoformat(),
                "description": f"Model rebuilt from {seasons_used[0]}–{seasons_used[-1]} nflverse PBP data.",
            }
        ],
    }
    _write_model(model)
    print("Done — model.json written.")


# ── Sub-routines ───────────────────────────────────────────────────────────────

def _estimate_effects(passes: pd.DataFrame) -> tuple[dict, dict]:
    """
    Estimate raw completion-percentage effects for each environment condition
    relative to the Clear/Mild baseline, controlling for covariates via
    cell-mean residuals.
    """
    complete_col = "complete_pass"
    baseline = passes[passes["weather"] == "clear"][complete_col].mean()

    effects_raw  = {}
    effects_meta = {}

    # Weather buckets (relative to clear baseline)
    for bucket in ["clear", "wind", "rain", "cold", "snow"]:
        subset = passes[passes["weather"] == bucket][complete_col]
        n      = len(subset)
        mean   = float(subset.mean()) if n > 0 else float("nan")
        se     = float(subset.std() / math.sqrt(n)) if n > 1 else 0.0
        raw_eff = (mean - baseline) if not math.isnan(mean) else 0.0
        effects_raw[bucket]  = raw_eff
        effects_meta[bucket] = {"n_plays": n, "se": se}

    # Dome effect (comparing dome plays to outdoor-clear plays)
    dome_plays    = passes[passes["dome_frac"] >= 0.9][complete_col]
    outdoor_clear = passes[(passes["dome_frac"] < 0.1) & (passes["weather"] == "clear")][complete_col]
    dome_mean     = float(dome_plays.mean()) if len(dome_plays) > 0 else baseline
    outdoor_mean  = float(outdoor_clear.mean()) if len(outdoor_clear) > 0 else baseline
    dome_se       = float(dome_plays.std() / math.sqrt(len(dome_plays))) if len(dome_plays) > 1 else 0.0

    effects_raw["dome"]  = dome_mean - outdoor_mean
    effects_meta["dome"] = {"n_plays": int(len(dome_plays)), "se": dome_se}

    # Baseline clear is always 0 by definition
    effects_raw["clear"]  = 0.0
    effects_meta["clear"]["se"] = 0.0

    return effects_raw, effects_meta


def _yoy_validation(passes: pd.DataFrame, effects: dict) -> dict:
    """Year-over-year stability of raw vs. environment-neutral Comp%."""
    try:
        seasons = sorted(passes["season"].dropna().unique())
        if len(seasons) < 2:
            return {}

        records = []
        for season in seasons:
            df = passes[passes["season"] == season].copy()
            for qb_id, grp in df.groupby("passer_player_id"):
                if len(grp) < MIN_ATTEMPTS_QB:
                    continue
                raw_cp = float(grp["complete_pass"].mean())
                # Compute environment-neutral Comp%
                dome_share = float(grp["dome_frac"].mean())
                weather_dist = grp["weather"].value_counts(normalize=True).to_dict()
                env_effect = (
                    effects.get("dome", 0) * dome_share
                    + sum(effects.get(w, 0) * f for w, f in weather_dist.items())
                )
                records.append({
                    "season": int(season),
                    "qb_id":  qb_id,
                    "raw_cp": raw_cp,
                    "neutral_cp": raw_cp - env_effect,
                })

        df_rec = pd.DataFrame(records)
        if df_rec.empty:
            return {}

        df_pivot_raw     = df_rec.pivot(index="qb_id", columns="season", values="raw_cp")
        df_pivot_neutral = df_rec.pivot(index="qb_id", columns="season", values="neutral_cp")

        r_raw     = []
        r_neutral = []

        for i in range(len(seasons) - 1):
            s1, s2 = int(seasons[i]), int(seasons[i + 1])
            if s1 not in df_pivot_raw.columns or s2 not in df_pivot_raw.columns:
                continue
            r_raw.append(pearson_r(df_pivot_raw[s1], df_pivot_raw[s2]))
            r_neutral.append(pearson_r(df_pivot_neutral[s1], df_pivot_neutral[s2]))

        return {
            "yoy_raw":     round(float(np.nanmean(r_raw)), 4) if r_raw else None,
            "yoy_neutral": round(float(np.nanmean(r_neutral)), 4) if r_neutral else None,
        }
    except Exception as exc:
        print(f"  Validation failed: {exc}")
        return {}


def _build_stadium_list(passes: pd.DataFrame) -> list[dict]:
    """Build a list of {id, name, type, dome_frac} for the tool dropdown."""
    if "stadium" not in passes.columns:
        return []

    rows = []
    for stadium_name, grp in passes.groupby("stadium"):
        if not stadium_name or not isinstance(stadium_name, str):
            continue
        roof_mode = grp["roof"].mode().iloc[0] if "roof" in grp.columns and not grp["roof"].empty else "outdoor"
        dome_frac = DOME_FRACTION_OVERRIDES.get(
            stadium_name,
            1.0 if roof_mode in ("dome", "closed") else (0.5 if roof_mode == "retractable" else 0.0),
        )
        rows.append({
            "id":        stadium_name.lower().replace(" ", "-"),
            "name":      stadium_name,
            "type":      str(roof_mode),
            "dome_frac": round(dome_frac, 2),
        })

    return sorted(rows, key=lambda r: r["name"])


def _default_effects() -> dict:
    """Fallback effects when no data is available."""
    return {
        "dome":  0.012,
        "clear": 0.000,
        "wind": -0.015,
        "rain": -0.010,
        "cold": -0.008,
        "snow": -0.022,
    }


def _default_effects_meta() -> dict:
    """Fallback metadata when no data is available."""
    keys = ["dome", "clear", "wind", "rain", "cold", "snow"]
    return {k: {"raw_effect": 0.0, "n_plays": 0, "ci_lo": 0.0, "ci_hi": 0.0} for k in keys}


def _write_model(model: dict) -> None:
    with open("model.json", "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print(f"  model.json written ({len(json.dumps(model)):,} bytes).")


if __name__ == "__main__":
    main()
