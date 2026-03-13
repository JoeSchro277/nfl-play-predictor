"""
Pre-compute Team Stats for Web App
====================================
Run this ONCE before launching the web app.
It calculates team tendencies, coach patterns, and
opponent defensive stats, then saves them as JSON.

Usage: python precompute_team_stats.py
Output: team_stats.json (loaded by the Flask app)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("Loading NFL data...")
import nfl_data_py as nfl

# Use most recent full season
pbp = nfl.import_pbp_data([2024])
pbp = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
pbp['is_pass'] = (pbp['play_type'] == 'pass').astype(int)

print(f"Loaded {len(pbp):,} plays from 2024 season")

teams = sorted(pbp['posteam'].dropna().unique().tolist())
print(f"Teams: {len(teams)}")

stats = {}

for team in teams:
    team_plays = pbp[pbp['posteam'] == team]
    team_def = pbp[pbp['defteam'] == team]

    # Overall pass rate
    overall_pass_rate = team_plays['is_pass'].mean()

    # Pass rate by down
    down_rates = {}
    for d in [1, 2, 3, 4]:
        d_plays = team_plays[team_plays['down'] == d]
        if len(d_plays) > 0:
            down_rates[str(d)] = round(float(d_plays['is_pass'].mean()), 3)
        else:
            down_rates[str(d)] = round(float(overall_pass_rate), 3)

    # Run ratio (last 3 games approximation — use season average)
    team_ratio = round(float(1 - overall_pass_rate), 3)

    # Defensive stats
    if len(team_def) > 0:
        def_yards_per_play = round(float(team_def['yards_gained'].mean()), 2)
        pass_plays_def = team_def[team_def['play_type'] == 'pass']
        run_plays_def = team_def[team_def['play_type'] == 'run']
        def_pass_ypp = round(float(pass_plays_def['yards_gained'].mean()), 2) if len(pass_plays_def) > 0 else 5.5
        def_rush_ypp = round(float(run_plays_def['yards_gained'].mean()), 2) if len(run_plays_def) > 0 else 4.0
    else:
        def_yards_per_play = 5.0
        def_pass_ypp = 5.5
        def_rush_ypp = 4.0

    stats[team] = {
        'team_ratio': team_ratio,
        'coach_aggressiveness': round(float(overall_pass_rate), 3),
        'coach_pass_rate_by_down': down_rates,
        'def_yards_per_play': def_yards_per_play,
        'def_pass_yards_per_play': def_pass_ypp,
        'def_rush_yards_per_play': def_rush_ypp,
        'def_pass_rush_diff': round(def_pass_ypp - def_rush_ypp, 2),
    }

# Save
with open('team_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved stats for {len(stats)} teams to team_stats.json")
print("\nSample (KC):")
print(json.dumps(stats.get('KC', {}), indent=2))
