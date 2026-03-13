"""
NFL Run vs Pass Predictor — Game Replay Data Preparation
=========================================================
Pulls real games from nfl_data_py, runs the EXACT same feature
engineering pipeline as NFLDefinitiveModel.py, and saves complete
play-by-play data as JSON files ready for the replay UI.

Usage:
  python prepare_replay_data.py

This will:
  1. Download 2022-2024 PBP + FTN data
  2. Run the full feature engineering pipeline
  3. Save every game as a separate JSON in replay_games/
  4. Also reconstruct model_features_v3.joblib if missing

Requirements:
  pip install nfl_data_py pandas numpy joblib
"""

import pandas as pd
import numpy as np
import re
import json
import os
import time
import warnings
import joblib

warnings.filterwarnings('ignore')

REPLAY_DIR = 'replay_games'
os.makedirs(REPLAY_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA (identical to NFLDefinitiveModel.py)
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

import nfl_data_py as nfl

seasons = [2022, 2023, 2024]

start = time.time()
pbp = nfl.import_pbp_data(seasons)
print(f"PBP loaded: {len(pbp):,} plays ({time.time()-start:.1f}s)")

start = time.time()
ftn = nfl.import_ftn_data(seasons)
print(f"FTN loaded: {len(ftn):,} plays ({time.time()-start:.1f}s)")

# Filter to run and pass
pbp = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
pbp['is_pass'] = (pbp['play_type'] == 'pass').astype(int)
print(f"\nRun + pass plays: {len(pbp):,}")

# ============================================================
# STEP 2: MERGE FTN DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Merging FTN charting data...")
print("=" * 60)

ftn_features = ftn[[
    'nflverse_game_id', 'nflverse_play_id',
    'is_motion', 'n_offense_backfield', 'n_defense_box',
    'qb_location', 'starting_hash'
]].copy()

pbp = pbp.merge(
    ftn_features,
    left_on=['game_id', 'play_id'],
    right_on=['nflverse_game_id', 'nflverse_play_id'],
    how='left'
)
print(f"FTN matched: {pbp['is_motion'].notna().sum():,}/{len(pbp):,}")

# ============================================================
# STEP 3: ENGINEER ALL FEATURES (exact copy from NFLDefinitiveModel.py)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Engineering features...")
print("=" * 60)

# --- 3A: ORIGINAL FEATURES ---
print("  [3A] Original features...")
pbp['home_or_away'] = (pbp['posteam'] == pbp['home_team']).astype(int)
if 'score_differential' not in pbp.columns:
    pbp['score_differential'] = pbp['posteam_score'] - pbp['defteam_score']

# --- 3B: PERSONNEL FEATURES ---
print("  [3B] Personnel features...")

def parse_offense_personnel(s):
    if pd.isna(s):
        return {'n_rb': np.nan, 'n_te': np.nan, 'n_wr': np.nan, 'n_ol_extra': 0}
    result = {'n_rb': 0, 'n_te': 0, 'n_wr': 0, 'n_ol_extra': 0}
    for key, pat in [('n_rb', r'(\d+)\s*RB'), ('n_te', r'(\d+)\s*TE'),
                      ('n_wr', r'(\d+)\s*WR'), ('n_ol_extra', r'(\d+)\s*OL')]:
        m = re.search(pat, s)
        if m:
            result[key] = int(m.group(1))
    return result

def parse_defense_personnel(s):
    if pd.isna(s):
        return {'n_dl': np.nan, 'n_lb': np.nan, 'n_db': np.nan}
    result = {'n_dl': 0, 'n_lb': 0, 'n_db': 0}
    for key, pat in [('n_dl', r'(\d+)\s*DL'), ('n_lb', r'(\d+)\s*LB'),
                      ('n_db', r'(\d+)\s*DB')]:
        m = re.search(pat, s)
        if m:
            result[key] = int(m.group(1))
    return result

off_pers = pbp['offense_personnel'].apply(parse_offense_personnel).apply(pd.Series)
def_pers = pbp['defense_personnel'].apply(parse_defense_personnel).apply(pd.Series)
pbp = pd.concat([pbp, off_pers, def_pers], axis=1)

# --- 3C: FORMATION ENCODING ---
print("  [3C] Formation encoding...")
formation_dummies = pd.get_dummies(pbp['offense_formation'], prefix='formation', dummy_na=False).astype(int)
pbp = pd.concat([pbp, formation_dummies], axis=1)
formation_cols = sorted(formation_dummies.columns.tolist())
print(f"    Formation columns: {formation_cols}")

# --- 3D: FTN FEATURE ENCODING ---
print("  [3D] FTN features...")
pbp['ftn_motion'] = pbp['is_motion'].astype(float)
pbp['ftn_backfield'] = pbp['n_offense_backfield'].astype(float)
pbp['ftn_defense_box'] = pbp['n_defense_box'].astype(float)

qb_loc_dummies = pd.get_dummies(pbp['qb_location'], prefix='qb_loc', dummy_na=False).astype(int)
pbp = pd.concat([pbp, qb_loc_dummies], axis=1)
qb_loc_cols = sorted(qb_loc_dummies.columns.tolist())
print(f"    QB location columns: {qb_loc_cols}")

hash_dummies = pd.get_dummies(pbp['starting_hash'], prefix='hash', dummy_na=False).astype(int)
pbp = pd.concat([pbp, hash_dummies], axis=1)
hash_cols = sorted(hash_dummies.columns.tolist())
print(f"    Hash columns: {hash_cols}")

# --- 3E: TEAM RUN TENDENCY ---
print("  [3E] Team run tendency...")
pbp = pbp.sort_values(['posteam', 'game_id', 'play_id']).reset_index(drop=True)

game_run_ratio = pbp.groupby(['posteam', 'game_id']).agg(
    total_plays=('is_pass', 'count'),
    pass_plays=('is_pass', 'sum')
).reset_index()
game_run_ratio['run_ratio'] = 1 - (game_run_ratio['pass_plays'] / game_run_ratio['total_plays'])
game_run_ratio['team_ratio'] = game_run_ratio.groupby('posteam')['run_ratio'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
pbp = pbp.merge(game_run_ratio[['posteam', 'game_id', 'team_ratio']],
                on=['posteam', 'game_id'], how='left')

# --- 3F: INTERACTION FEATURES ---
print("  [3F] Interaction features...")
pbp['score_time_interaction'] = pbp['score_differential'] * (pbp['game_seconds_remaining'] / 3600)
pbp['down_distance_interaction'] = pbp['down'] * pbp['ydstogo']

# --- 3G: PLAY SEQUENCING ---
print("  [3G] Play sequencing (drive context)...")
pbp = pbp.sort_values(['game_id', 'fixed_drive', 'play_id']).reset_index(drop=True)
drive_groups = pbp.groupby(['game_id', 'fixed_drive'])

pbp['prev_play_pass'] = drive_groups['is_pass'].shift(1)
pbp['prev_play_2_pass'] = drive_groups['is_pass'].shift(2)
pbp['prev_play_3_pass'] = drive_groups['is_pass'].shift(3)

pbp['drive_play_num'] = drive_groups.cumcount()
pbp['drive_passes_so_far'] = drive_groups['is_pass'].cumsum() - pbp['is_pass']
pbp['drive_pass_ratio'] = np.where(
    pbp['drive_play_num'] > 0,
    pbp['drive_passes_so_far'] / pbp['drive_play_num'],
    0.5
)

def count_consecutive_same(series):
    result = pd.Series(0, index=series.index)
    for i in range(1, len(series)):
        idx = series.index[i]
        prev_idx = series.index[i-1]
        if i >= 2:
            prev2_idx = series.index[i-2]
            if series.iloc[i-1] == series.iloc[i-2]:
                result.iloc[i] = result.iloc[i-1] + 1
            else:
                result.iloc[i] = 1
        else:
            result.iloc[i] = 1
    return result

pbp['consecutive_same_play'] = drive_groups['is_pass'].transform(count_consecutive_same)
pbp['drive_play_number'] = pbp['drive_play_num'] + 1
pbp['prev_first_down'] = drive_groups['first_down'].shift(1).fillna(0)
pbp['prev_yards_gained'] = drive_groups['yards_gained'].shift(1).fillna(0)

# --- 3H: COACH TENDENCIES ---
print("  [3H] Coach tendencies...")
game_down_rates = pbp.groupby(['season', 'posteam', 'game_id', 'down']).agg(
    plays=('is_pass', 'count'),
    passes=('is_pass', 'sum')
).reset_index()
game_down_rates['pass_rate'] = game_down_rates['passes'] / game_down_rates['plays']

coach_tendency = game_down_rates.sort_values(['season', 'posteam', 'game_id']).copy()
coach_tendency['coach_pass_rate'] = coach_tendency.groupby(
    ['season', 'posteam', 'down']
)['pass_rate'].transform(lambda x: x.shift(1).expanding().mean())

pbp = pbp.merge(
    coach_tendency[['season', 'posteam', 'game_id', 'down', 'coach_pass_rate']],
    on=['season', 'posteam', 'game_id', 'down'],
    how='left'
)

game_overall = pbp.groupby(['season', 'posteam', 'game_id']).agg(
    plays=('is_pass', 'count'),
    passes=('is_pass', 'sum')
).reset_index()
game_overall['overall_pass_rate'] = game_overall['passes'] / game_overall['plays']
game_overall['coach_aggressiveness'] = game_overall.groupby(
    ['season', 'posteam']
)['overall_pass_rate'].transform(lambda x: x.shift(1).expanding().mean())

pbp = pbp.merge(
    game_overall[['season', 'posteam', 'game_id', 'coach_aggressiveness']],
    on=['season', 'posteam', 'game_id'],
    how='left'
)

# --- 3I: WEATHER ---
print("  [3I] Weather features...")
pbp['temperature'] = pbp['temp'].astype(float)
pbp['wind_speed'] = pbp['wind'].astype(float)
pbp['is_indoors'] = pbp['roof'].isin(['dome', 'closed']).astype(int)
pbp['high_wind'] = (pbp['wind_speed'] >= 15).astype(int)
pbp['cold_weather'] = (pbp['temperature'] < 40).astype(int)

indoor_mask = pbp['is_indoors'] == 1
outdoor_avg_temp = pbp.loc[~indoor_mask, 'temperature'].mean()
pbp['adj_temperature'] = np.where(indoor_mask, outdoor_avg_temp, pbp['temperature'])
pbp['adj_wind'] = np.where(indoor_mask, 0, pbp['wind_speed'])

# --- 3J: OPPONENT DEFENSIVE STRENGTH ---
print("  [3J] Opponent defensive strength...")
def_performance = pbp.groupby(['season', 'defteam', 'game_id']).agg(
    def_plays=('yards_gained', 'count'),
    def_yards_allowed=('yards_gained', 'sum'),
    def_passes_faced=('is_pass', 'sum'),
    def_runs_faced=('is_pass', lambda x: (1 - x).sum()),
    def_pass_yards=('passing_yards', lambda x: x.fillna(0).sum()),
    def_rush_yards=('rushing_yards', lambda x: x.fillna(0).sum()),
).reset_index()

def_performance['def_yards_per_play'] = def_performance['def_yards_allowed'] / def_performance['def_plays']
def_performance['def_pass_yards_per_play'] = def_performance['def_pass_yards'] / def_performance['def_passes_faced'].clip(1)
def_performance['def_rush_yards_per_play'] = def_performance['def_rush_yards'] / def_performance['def_runs_faced'].clip(1)

for col in ['def_yards_per_play', 'def_pass_yards_per_play', 'def_rush_yards_per_play']:
    rolling_col = f'opp_{col}'
    def_performance[rolling_col] = def_performance.groupby(
        ['season', 'defteam']
    )[col].transform(lambda x: x.shift(1).expanding().mean())

pbp = pbp.merge(
    def_performance[['season', 'defteam', 'game_id',
                      'opp_def_yards_per_play', 'opp_def_pass_yards_per_play',
                      'opp_def_rush_yards_per_play']],
    on=['season', 'defteam', 'game_id'],
    how='left'
)
pbp['opp_def_pass_rush_diff'] = pbp['opp_def_pass_yards_per_play'] - pbp['opp_def_rush_yards_per_play']

# ============================================================
# STEP 4: DEFINE FEATURE SETS (exactly matching NFLDefinitiveModel.py)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Building feature list...")
print("=" * 60)

original_features = [
    'down', 'ydstogo', 'yardline_100', 'score_differential',
    'game_seconds_remaining', 'shotgun', 'no_huddle', 'home_or_away'
]

personnel_features = [
    'defenders_in_box', 'n_rb', 'n_te', 'n_wr', 'n_ol_extra',
    'n_dl', 'n_lb', 'n_db'
]

ftn_feats = ['ftn_motion', 'ftn_backfield', 'ftn_defense_box']

interaction_features = ['score_time_interaction', 'down_distance_interaction']

sequencing_features = [
    'prev_play_pass', 'prev_play_2_pass', 'prev_play_3_pass',
    'drive_pass_ratio', 'drive_play_number', 'consecutive_same_play',
    'prev_first_down', 'prev_yards_gained'
]

coach_features = ['coach_pass_rate', 'coach_aggressiveness']

weather_features = ['adj_temperature', 'adj_wind', 'is_indoors', 'high_wind', 'cold_weather']

opponent_features = [
    'opp_def_yards_per_play', 'opp_def_pass_yards_per_play',
    'opp_def_rush_yards_per_play', 'opp_def_pass_rush_diff'
]

features_v3_full = (original_features + personnel_features + formation_cols +
                    ftn_feats + qb_loc_cols + hash_cols +
                    interaction_features + ['team_ratio'] +
                    sequencing_features + coach_features +
                    weather_features + opponent_features)

print(f"Total features: {len(features_v3_full)}")
print(f"Feature list: {features_v3_full}")

# Save/reconstruct model_features_v3.joblib
joblib.dump(features_v3_full, 'model_features_v3.joblib')
print(f"\nSaved: model_features_v3.joblib ({len(features_v3_full)} features)")

# ============================================================
# STEP 5: SAVE GAMES FOR REPLAY
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Saving games for replay...")
print("=" * 60)

# Columns we need for the replay UI (metadata + features + target)
metadata_cols = [
    'game_id', 'play_id', 'season', 'week', 'posteam', 'defteam',
    'home_team', 'away_team', 'posteam_score', 'defteam_score',
    'play_type', 'is_pass', 'desc', 'down', 'ydstogo', 'yardline_100',
    'game_seconds_remaining', 'fixed_drive', 'yards_gained', 'first_down',
    'offense_formation', 'offense_personnel', 'defense_personnel',
    'qb_location', 'starting_hash', 'defenders_in_box',
    'shotgun', 'no_huddle', 'home_or_away',
    'is_motion', 'n_offense_backfield',
]

# Get list of games
game_list = pbp.groupby('game_id').agg(
    season=('season', 'first'),
    week=('week', 'first'),
    home_team=('home_team', 'first'),
    away_team=('away_team', 'first'),
    home_score=('home_score', 'max'),  # final score
    away_score=('away_score', 'max'),
    n_plays=('play_id', 'count'),
).reset_index()

# Filter to games with enough FTN data for good predictions
ftn_coverage = pbp.groupby('game_id')['ftn_motion'].apply(lambda x: x.notna().mean()).reset_index()
ftn_coverage.columns = ['game_id', 'ftn_pct']
game_list = game_list.merge(ftn_coverage, on='game_id')
good_games = game_list[game_list['ftn_pct'] > 0.8]  # >80% FTN coverage
print(f"Games with >80% FTN coverage: {len(good_games)}")

# Save game index
game_index = good_games[['game_id', 'season', 'week', 'home_team', 'away_team',
                          'home_score', 'away_score', 'n_plays']].copy()
game_index = game_index.sort_values(['season', 'week', 'game_id']).reset_index(drop=True)
game_index.to_json(os.path.join(REPLAY_DIR, 'game_index.json'), orient='records', indent=2)
print(f"Saved game index: {len(game_index)} games")

# Save each game's play-by-play with pre-computed features
saved = 0
for _, game_row in game_index.iterrows():
    gid = game_row['game_id']
    game_plays = pbp[pbp['game_id'] == gid].sort_values('play_id').copy()

    # Build feature vectors for each play
    plays_data = []
    skipped = 0
    for _, play in game_plays.iterrows():
        # Skip plays missing critical fields (2pt conversions, etc.)
        if pd.isna(play.get('down')) or pd.isna(play.get('ydstogo')) or pd.isna(play.get('yardline_100')):
            skipped += 1
            continue

        # Extract feature values
        feature_values = {}
        has_all = True
        for feat in features_v3_full:
            val = play.get(feat, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                has_all = False
                feature_values[feat] = 0.0  # default for missing
            else:
                feature_values[feat] = float(val)

        # Build play metadata
        play_data = {
            'play_id': int(play['play_id']),
            'drive': int(play['fixed_drive']) if pd.notna(play.get('fixed_drive')) else 0,
            'down': int(play['down']),
            'distance': int(play['ydstogo']),
            'yard_line': int(play['yardline_100']),
            'possession_team': str(play['posteam']),
            'defensive_team': str(play['defteam']),
            'home_score': int(play.get('posteam_score', 0)) if pd.notna(play.get('posteam_score')) else 0,
            'away_score': int(play.get('defteam_score', 0)) if pd.notna(play.get('defteam_score')) else 0,
            'game_seconds_remaining': int(play['game_seconds_remaining']) if pd.notna(play['game_seconds_remaining']) else 0,
            'quarter': int(play.get('qtr', 1)) if pd.notna(play.get('qtr')) else 1,
            'is_pass': int(play['is_pass']),
            'play_type': str(play['play_type']),
            'description': str(play.get('desc', '')),
            'yards_gained': int(play['yards_gained']) if pd.notna(play['yards_gained']) else 0,
            'first_down': int(play.get('first_down', 0)) if pd.notna(play.get('first_down')) else 0,
            'formation': str(play.get('offense_formation', 'UNKNOWN')),
            'qb_location': str(play.get('qb_location', '')) if pd.notna(play.get('qb_location')) else '',
            'shotgun': int(play.get('shotgun', 0)) if pd.notna(play.get('shotgun')) else 0,
            'defenders_in_box': int(play.get('defenders_in_box', 6)) if pd.notna(play.get('defenders_in_box')) else 6,
            'n_rb': int(play.get('n_rb', 1)) if pd.notna(play.get('n_rb')) else 1,
            'n_te': int(play.get('n_te', 1)) if pd.notna(play.get('n_te')) else 1,
            'n_wr': int(play.get('n_wr', 3)) if pd.notna(play.get('n_wr')) else 3,
            'offense_personnel': str(play.get('offense_personnel', '')) if pd.notna(play.get('offense_personnel')) else '',
            'defense_personnel': str(play.get('defense_personnel', '')) if pd.notna(play.get('defense_personnel')) else '',
            'has_all_features': has_all,
            'features': feature_values,
        }
        plays_data.append(play_data)

    game_data = {
        'game_id': gid,
        'season': int(game_row['season']),
        'week': int(game_row['week']),
        'home_team': game_row['home_team'],
        'away_team': game_row['away_team'],
        'home_score': int(game_row['home_score']),
        'away_score': int(game_row['away_score']),
        'plays': plays_data,
        'feature_names': features_v3_full,
    }

    filename = f"{gid}.json"
    with open(os.path.join(REPLAY_DIR, filename), 'w') as f:
        json.dump(game_data, f)
    saved += 1

    if saved % 50 == 0:
        print(f"  Saved {saved}/{len(game_index)} games...")

print(f"\nDone! Saved {saved} games to {REPLAY_DIR}/")
print(f"Game index: {REPLAY_DIR}/game_index.json")
print(f"Feature list: model_features_v3.joblib")
print(f"\nYou can now run: python app.py")
