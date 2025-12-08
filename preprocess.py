import pandas as pd
import numpy as np
import re

RAW_FILE = "f1_raw_2021_2025.csv"
OUT_FILE = "f1_processed_2021_2025.csv"


def parse_lap_time_to_sec(s):
    """
    Parse strings like '1:29.841' or '1:42:06.304' into seconds.
    Returns np.nan if it can't parse.
    """
    if pd.isna(s) or s == "":
        return np.nan
    s = s.strip()
    parts = s.split(":")

    try:
        if len(parts) == 2:
            # mm:ss.sss
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        elif len(parts) == 3:
            # hh:mm:ss.sss
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
        else:
            # Something like '+8.481s', dnf or sumn else
            m = re.search(r"([\d\.]+)s$", s)
            if m:
                return float(m.group(1))
            return np.nan
    except ValueError:
        return np.nan


def parse_speed_to_float(s):
    """
    Turn something like '220.345' or '220.345 kph' into float.
    """
    if pd.isna(s) or s == "":
        return np.nan
    s = str(s)
    cleaned = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def main():
    df = pd.read_csv(RAW_FILE)

    # Basic numeric conversions
    numeric_cols = [
        "total_laps",
        "grid_position",
        "final_position",
        "laps_completed",
        "pit_stops",
        "first_pit_lap",
        "last_pit_lap",
        "fastest_lap_rank",
        "fastest_lap_lap",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Number of drivers per race (group by year, round, race_slug)
    grp_keys = ["year", "round", "race_slug"]
    df["num_drivers"] = df.groupby(grp_keys)["driver_number"].transform("count")

    # Parse times to seconds
    if "grid_time" in df.columns:
        df["grid_time_sec"] = df["grid_time"].apply(parse_lap_time_to_sec)
    else:
        df["grid_time_sec"] = np.nan

    if "fastest_lap_time" in df.columns:
        df["fastest_lap_time_sec"] = df["fastest_lap_time"].apply(parse_lap_time_to_sec)
    else:
        df["fastest_lap_time_sec"] = np.nan

    if "total_pit_time" in df.columns:
        df["total_pit_time_sec"] = pd.to_numeric(df["total_pit_time"], errors="coerce")
    else:
        df["total_pit_time_sec"] = np.nan

    if "fastest_lap_avg_speed" in df.columns:
        df["fastest_lap_avg_speed_float"] = df["fastest_lap_avg_speed"].apply(
            parse_speed_to_float
        )
    else:
        df["fastest_lap_avg_speed_float"] = np.nan

    # Per-race fastest lap time (to compute deltas)
    df["race_fastest_lap_time_sec"] = (
        df.groupby(grp_keys)["fastest_lap_time_sec"].transform("min")
    )

    
    # Grid features
    df["grid_pos"] = df["grid_position"]
    df["grid_pos_norm"] = df["grid_pos"] / df["num_drivers"]

    # Pit strategy features
    df["pit_stops_int"] = df["pit_stops"].fillna(0).astype(int)

    df["first_pit_frac"] = df["first_pit_lap"] / df["total_laps"]
    df["last_pit_frac"] = df["last_pit_lap"] / df["total_laps"]

    df["pitted_at_all"] = (df["pit_stops_int"] > 0).astype(int)

    # If never pitted, push frac > 1 so pit_before_half = 0
    df.loc[df["pitted_at_all"] == 0, "first_pit_frac"] = 1.1
    df.loc[df["pitted_at_all"] == 0, "last_pit_frac"] = 1.1

    df["pit_before_half"] = (df["first_pit_frac"] <= 0.5).astype(int)

    # Pace features from fastest lap
    df["fastest_lap_rank_norm"] = df["fastest_lap_rank"] / df["num_drivers"]
    df["fastest_lap_lap_frac"] = df["fastest_lap_lap"] / df["total_laps"]
    df["fastest_lap_delta"] = (
        df["fastest_lap_time_sec"] - df["race_fastest_lap_time_sec"]
    )

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Numeric feature columns we want to use
    feature_cols_numeric = [
        "grid_pos",
        "grid_pos_norm",
        "pit_stops_int",
        "first_pit_frac",
        "last_pit_frac",
        "pitted_at_all",
        "pit_before_half",
        "total_pit_time_sec",
        "fastest_lap_rank_norm",
        "fastest_lap_lap_frac",
        "fastest_lap_delta",
    ]

    # Fill NaNs in numeric features with median (simple and safe)
    for col in feature_cols_numeric:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Label
    df["is_winner"] = df["is_winner"].astype(int)

    # One-hot encode team_name
    team_dummies = pd.get_dummies(df["team_name"], prefix="team")
    df = pd.concat([df, team_dummies], axis=1)

    # Final feature list = numeric + team one-hots
    feature_cols = feature_cols_numeric + list(team_dummies.columns)

    # Meta columns to keep for grouping/interpretation
    meta_cols = [
        "year",
        "round",
        "race_slug",
        "race_name",
        "driver_code",
        "driver_name",
        "team_name",
    ]

    out_cols = meta_cols + feature_cols + ["is_winner"]

    df_out = df[out_cols].copy()
    df_out.to_csv(OUT_FILE, index=False)

    print(f"Wrote processed dataset with {len(df_out)} rows to {OUT_FILE}")
    print("Feature columns used:")
    for c in feature_cols:
        print("  -", c)


if __name__ == "__main__":
    main()
