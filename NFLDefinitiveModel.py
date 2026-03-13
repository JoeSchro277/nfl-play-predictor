"""
NFL Run vs Pass Predictor - DEFINITIVE MODEL (V3)
===================================================
The full kitchen sink model with every available feature.

NEW feature groups:
  1. Play sequencing (what happened on this drive so far)
  2. Coach tendencies (offensive coordinator patterns)
  3. Weather (temp, wind, indoor/outdoor)
  4. Opponent defensive strength (how good is the D they're facing)

Builds on top of the V2 model's 75.4% accuracy.
"""

import pandas as pd
import numpy as np
import re
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD ALL DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading all data sources...")
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
print(pbp['play_type'].value_counts().to_string())

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
# STEP 3: ENGINEER ALL FEATURES
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
formation_cols = formation_dummies.columns.tolist()

# --- 3D: FTN FEATURE ENCODING ---
print("  [3D] FTN features...")
pbp['ftn_motion'] = pbp['is_motion'].astype(float)
pbp['ftn_backfield'] = pbp['n_offense_backfield'].astype(float)
pbp['ftn_defense_box'] = pbp['n_defense_box'].astype(float)

qb_loc_dummies = pd.get_dummies(pbp['qb_location'], prefix='qb_loc', dummy_na=False).astype(int)
pbp = pd.concat([pbp, qb_loc_dummies], axis=1)
qb_loc_cols = qb_loc_dummies.columns.tolist()

hash_dummies = pd.get_dummies(pbp['starting_hash'], prefix='hash', dummy_na=False).astype(int)
pbp = pd.concat([pbp, hash_dummies], axis=1)
hash_cols = hash_dummies.columns.tolist()

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

# ============================================================
# NEW FEATURE GROUP 1: PLAY SEQUENCING
# ============================================================
print("  [3G] Play sequencing (drive context)...")

# Sort by game, drive, and play order
pbp = pbp.sort_values(['game_id', 'fixed_drive', 'play_id']).reset_index(drop=True)

# Within each drive, calculate sequencing features
# Group by game and drive
drive_groups = pbp.groupby(['game_id', 'fixed_drive'])

# Previous play was pass (1) or run (0)
pbp['prev_play_pass'] = drive_groups['is_pass'].shift(1)

# Previous 2nd play
pbp['prev_play_2_pass'] = drive_groups['is_pass'].shift(2)

# Previous 3rd play
pbp['prev_play_3_pass'] = drive_groups['is_pass'].shift(3)

# Rolling pass ratio on current drive (excluding current play)
# Number of passes so far on this drive / total plays so far
pbp['drive_play_num'] = drive_groups.cumcount()  # 0-indexed play number in drive
pbp['drive_passes_so_far'] = drive_groups['is_pass'].cumsum() - pbp['is_pass']  # exclude current
pbp['drive_pass_ratio'] = np.where(
    pbp['drive_play_num'] > 0,
    pbp['drive_passes_so_far'] / pbp['drive_play_num'],
    0.5  # default for first play of drive
)

# Consecutive same play type streak
# How many runs or passes in a row just happened?
def compute_streak(group):
    streak = pd.Series(0, index=group.index)
    for i in range(1, len(group)):
        idx = group.index[i]
        prev_idx = group.index[i - 1]
        if group.loc[prev_idx, 'is_pass'] == group.loc[group.index[max(0, i-2)], 'is_pass'] if i >= 2 else True:
            streak.loc[idx] = streak.loc[prev_idx] + 1
        else:
            streak.loc[idx] = 1
    return streak

# Simpler approach: count consecutive passes or runs before current play
def count_consecutive_same(series):
    """Count how many of the same play type in a row before this play."""
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

# Play number in drive (1-indexed for readability)
pbp['drive_play_number'] = pbp['drive_play_num'] + 1

# Was there a first down on the previous play?
pbp['prev_first_down'] = drive_groups['first_down'].shift(1).fillna(0)

# Yards gained on previous play
pbp['prev_yards_gained'] = drive_groups['yards_gained'].shift(1).fillna(0)

print(f"    Play sequencing features created")
print(f"    Avg drive pass ratio: {pbp['drive_pass_ratio'].mean():.3f}")
print(f"    Avg drive play number: {pbp['drive_play_number'].mean():.1f}")

# ============================================================
# NEW FEATURE GROUP 2: COACH TENDENCIES
# ============================================================
print("  [3H] Coach tendencies...")

# Calculate each team's overall pass rate by down for the season so far
# Use a rolling approach: for each game, use stats from all PRIOR games that season

# First, get per-game, per-down pass rates for each team
game_down_rates = pbp.groupby(['season', 'posteam', 'game_id', 'down']).agg(
    plays=('is_pass', 'count'),
    passes=('is_pass', 'sum')
).reset_index()
game_down_rates['pass_rate'] = game_down_rates['passes'] / game_down_rates['plays']

# For each team-season, compute rolling average pass rate by down
# Shift by 1 game so we don't leak current game
coach_tendency = game_down_rates.sort_values(['season', 'posteam', 'game_id']).copy()

# Rolling average across prior games for each down
coach_tendency['coach_pass_rate'] = coach_tendency.groupby(
    ['season', 'posteam', 'down']
)['pass_rate'].transform(lambda x: x.shift(1).expanding().mean())

# Merge back
pbp = pbp.merge(
    coach_tendency[['season', 'posteam', 'game_id', 'down', 'coach_pass_rate']],
    on=['season', 'posteam', 'game_id', 'down'],
    how='left'
)

# Overall coach aggressiveness (pass rate regardless of down)
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

print(f"    Coach pass rate by down: {pbp['coach_pass_rate'].notna().sum():,} non-null")
print(f"    Coach aggressiveness: {pbp['coach_aggressiveness'].notna().sum():,} non-null")

# ============================================================
# NEW FEATURE GROUP 3: WEATHER
# ============================================================
print("  [3I] Weather features...")

# Temperature (already numeric)
pbp['temperature'] = pbp['temp'].astype(float)

# Wind speed (already numeric)
pbp['wind_speed'] = pbp['wind'].astype(float)

# Is it indoors? (dome or closed roof = no weather effect)
pbp['is_indoors'] = pbp['roof'].isin(['dome', 'closed']).astype(int)

# Wind categories
pbp['high_wind'] = (pbp['wind_speed'] >= 15).astype(int)

# Cold weather (below 40°F)
pbp['cold_weather'] = (pbp['temperature'] < 40).astype(int)

# For indoor games, neutralize temp/wind
# (set to league average so they don't skew)
indoor_mask = pbp['is_indoors'] == 1
outdoor_avg_temp = pbp.loc[~indoor_mask, 'temperature'].mean()
outdoor_avg_wind = pbp.loc[~indoor_mask, 'wind_speed'].mean()

# Don't overwrite — create adjusted versions
pbp['adj_temperature'] = np.where(indoor_mask, outdoor_avg_temp, pbp['temperature'])
pbp['adj_wind'] = np.where(indoor_mask, 0, pbp['wind_speed'])  # no wind indoors

print(f"    Temperature: {pbp['temperature'].notna().sum():,} non-null, avg={pbp['temperature'].mean():.1f}°F")
print(f"    Wind: {pbp['wind_speed'].notna().sum():,} non-null, avg={pbp['wind_speed'].mean():.1f} mph")
print(f"    Indoor games: {indoor_mask.sum():,}/{len(pbp):,} ({indoor_mask.mean()*100:.1f}%)")

# ============================================================
# NEW FEATURE GROUP 4: OPPONENT DEFENSIVE STRENGTH
# ============================================================
print("  [3J] Opponent defensive strength...")

# For each team-game, calculate the opponent's defensive performance
# from prior games that season

# Yards allowed per play by the defense in prior games
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

# Rolling average of prior games
for col in ['def_yards_per_play', 'def_pass_yards_per_play', 'def_rush_yards_per_play']:
    rolling_col = f'opp_{col}'
    def_performance[rolling_col] = def_performance.groupby(
        ['season', 'defteam']
    )[col].transform(lambda x: x.shift(1).expanding().mean())

# Merge back — the opponent's prior defensive performance
pbp = pbp.merge(
    def_performance[['season', 'defteam', 'game_id',
                      'opp_def_yards_per_play', 'opp_def_pass_yards_per_play',
                      'opp_def_rush_yards_per_play']],
    on=['season', 'defteam', 'game_id'],
    how='left'
)

# Pass/rush yards differential tells us if the D is worse against pass or run
pbp['opp_def_pass_rush_diff'] = pbp['opp_def_pass_yards_per_play'] - pbp['opp_def_rush_yards_per_play']

print(f"    Opponent yards/play: {pbp['opp_def_yards_per_play'].notna().sum():,} non-null")
print(f"    Opponent pass/rush diff: avg={pbp['opp_def_pass_rush_diff'].mean():.2f}")

# ============================================================
# STEP 4: DEFINE FEATURE SETS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Defining feature sets...")
print("=" * 60)

original_features = [
    'down', 'ydstogo', 'yardline_100', 'score_differential',
    'game_seconds_remaining', 'shotgun', 'no_huddle', 'home_or_away'
]

personnel_features = [
    'defenders_in_box', 'n_rb', 'n_te', 'n_wr', 'n_ol_extra',
    'n_dl', 'n_lb', 'n_db'
]

ftn_features = ['ftn_motion', 'ftn_backfield', 'ftn_defense_box']

interaction_features = ['score_time_interaction', 'down_distance_interaction']

# NEW
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

# Build progressively
features_v2 = (original_features + personnel_features + formation_cols +
               ftn_features + qb_loc_cols + hash_cols +
               interaction_features + ['team_ratio'])

features_v3_seq = features_v2 + sequencing_features
features_v3_coach = features_v3_seq + coach_features
features_v3_weather = features_v3_coach + weather_features
features_v3_full = features_v3_weather + opponent_features

print(f"V2 (previous best): {len(features_v2)} features")
print(f"+ Sequencing: {len(features_v3_seq)} features")
print(f"+ Coach: {len(features_v3_coach)} features")
print(f"+ Weather: {len(features_v3_weather)} features")
print(f"+ Opponent: {len(features_v3_full)} features (FULL)")

# ============================================================
# STEP 5: PREPARE DATA AND TRAIN
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training models...")
print("=" * 60)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Use full feature set, drop NaNs
all_features = features_v3_full
df = pbp[all_features + ['is_pass']].dropna()

print(f"\nClean dataset: {len(df):,} plays")
print(f"  Pass: {(df['is_pass'] == 1).sum():,} ({(df['is_pass'] == 1).mean()*100:.1f}%)")
print(f"  Run: {(df['is_pass'] == 0).sum():,} ({(df['is_pass'] == 0).mean()*100:.1f}%)")

X = df[all_features]
y = df['is_pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train):,}  |  Test: {len(X_test):,}")

# --- Model A: V2 baseline (previous best) ---
print("\n--- Model A: V2 features (previous best) ---")
feats_a = [f for f in features_v2 if f in X.columns]
gb_a = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_a.fit(X_train[feats_a], y_train)
acc_a = accuracy_score(y_test, gb_a.predict(X_test[feats_a]))
cv_a = cross_val_score(gb_a, X_train[feats_a], y_train, cv=5).mean()
print(f"  Test: {acc_a*100:.2f}%  |  CV: {cv_a*100:.2f}%")

# --- Model B: + Play Sequencing ---
print("\n--- Model B: + Play Sequencing ---")
feats_b = [f for f in features_v3_seq if f in X.columns]
gb_b = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_b.fit(X_train[feats_b], y_train)
acc_b = accuracy_score(y_test, gb_b.predict(X_test[feats_b]))
cv_b = cross_val_score(gb_b, X_train[feats_b], y_train, cv=5).mean()
print(f"  Test: {acc_b*100:.2f}%  |  CV: {cv_b*100:.2f}%")

# --- Model C: + Coach Tendencies ---
print("\n--- Model C: + Coach Tendencies ---")
feats_c = [f for f in features_v3_coach if f in X.columns]
gb_c = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_c.fit(X_train[feats_c], y_train)
acc_c = accuracy_score(y_test, gb_c.predict(X_test[feats_c]))
cv_c = cross_val_score(gb_c, X_train[feats_c], y_train, cv=5).mean()
print(f"  Test: {acc_c*100:.2f}%  |  CV: {cv_c*100:.2f}%")

# --- Model D: + Weather ---
print("\n--- Model D: + Weather ---")
feats_d = [f for f in features_v3_weather if f in X.columns]
gb_d = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_d.fit(X_train[feats_d], y_train)
acc_d = accuracy_score(y_test, gb_d.predict(X_test[feats_d]))
cv_d = cross_val_score(gb_d, X_train[feats_d], y_train, cv=5).mean()
print(f"  Test: {acc_d*100:.2f}%  |  CV: {cv_d*100:.2f}%")

# --- Model E: + Opponent Strength (FULL) ---
print("\n--- Model E: FULL feature set ---")
feats_e = [f for f in features_v3_full if f in X.columns]
gb_e = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_e.fit(X_train[feats_e], y_train)
acc_e = accuracy_score(y_test, gb_e.predict(X_test[feats_e]))
cv_e = cross_val_score(gb_e, X_train[feats_e], y_train, cv=5).mean()
print(f"  Test: {acc_e*100:.2f}%  |  CV: {cv_e*100:.2f}%")

# --- Model F: Tuned FULL ---
print("\n--- Model F: Tuned GB (FULL features) ---")
gb_f = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    min_samples_leaf=20, subsample=0.8, random_state=42
)
gb_f.fit(X_train[feats_e], y_train)
acc_f = accuracy_score(y_test, gb_f.predict(X_test[feats_e]))
cv_f = cross_val_score(gb_f, X_train[feats_e], y_train, cv=5).mean()
print(f"  Test: {acc_f*100:.2f}%  |  CV: {cv_f*100:.2f}%")

# --- Model G: High-capacity tuned ---
print("\n--- Model G: High-capacity tuned GB ---")
gb_g = GradientBoostingClassifier(
    n_estimators=800, learning_rate=0.03, max_depth=5,
    min_samples_leaf=15, subsample=0.85, max_features='sqrt',
    random_state=42
)
gb_g.fit(X_train[feats_e], y_train)
acc_g = accuracy_score(y_test, gb_g.predict(X_test[feats_e]))
cv_g = cross_val_score(gb_g, X_train[feats_e], y_train, cv=5).mean()
print(f"  Test: {acc_g*100:.2f}%  |  CV: {cv_g*100:.2f}%")

# ============================================================
# STEP 6: FIND BEST MODEL
# ============================================================
models = {
    'A (V2 baseline)': (acc_a, cv_a, gb_a, feats_a),
    'B (+ Sequencing)': (acc_b, cv_b, gb_b, feats_b),
    'C (+ Coach)': (acc_c, cv_c, gb_c, feats_c),
    'D (+ Weather)': (acc_d, cv_d, gb_d, feats_d),
    'E (Full)': (acc_e, cv_e, gb_e, feats_e),
    'F (Tuned Full)': (acc_f, cv_f, gb_f, feats_e),
    'G (High-cap)': (acc_g, cv_g, gb_g, feats_e),
}

best_name = max(models, key=lambda k: models[k][0])
best_acc, best_cv, best_model, best_feats = models[best_name]

# ============================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Feature importance (best model)")
print("=" * 60)

importances = pd.DataFrame({
    'feature': best_feats,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nBest model: {best_name}")
print(f"\nTop 25 features:")
print(importances.head(25).to_string(index=False))

# Show feature group contributions
feature_groups = {
    'Original': original_features,
    'Personnel': personnel_features,
    'Formation': formation_cols,
    'FTN': ftn_features + qb_loc_cols + hash_cols,
    'Interactions': interaction_features,
    'Team tendency': ['team_ratio'],
    'Sequencing': sequencing_features,
    'Coach': coach_features,
    'Weather': weather_features,
    'Opponent': opponent_features,
}

print(f"\nFeature group contributions:")
for group_name, group_feats in feature_groups.items():
    available = [f for f in group_feats if f in importances['feature'].values]
    if available:
        group_imp = importances[importances['feature'].isin(available)]['importance'].sum()
        print(f"  {group_name:<15s}: {group_imp*100:.1f}%")

# ============================================================
# STEP 8: CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Classification report")
print("=" * 60)

best_preds = best_model.predict(X_test[best_feats])
print(classification_report(y_test, best_preds, target_names=['Run', 'Pass']))

# ============================================================
# STEP 9: RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: RESULTS SUMMARY")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        ACCURACY PROGRESSION                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  NFLPredictorSTRONG.py (original):               ~71.0%             ║
║  V1 Enhanced (+ personnel/formation):             74.3%             ║
║  V2 Enhanced (+ FTN charting):                    75.4%             ║
║                                                                      ║
║  V3 Model A: V2 baseline                     {acc_a*100:.1f}%  (CV: {cv_a*100:.1f}%)   ║
║  V3 Model B: + Play Sequencing               {acc_b*100:.1f}%  (CV: {cv_b*100:.1f}%)   ║
║  V3 Model C: + Coach Tendencies              {acc_c*100:.1f}%  (CV: {cv_c*100:.1f}%)   ║
║  V3 Model D: + Weather                       {acc_d*100:.1f}%  (CV: {cv_d*100:.1f}%)   ║
║  V3 Model E: + Opponent Strength (FULL)      {acc_e*100:.1f}%  (CV: {cv_e*100:.1f}%)   ║
║  V3 Model F: Tuned GB (Full)                 {acc_f*100:.1f}%  (CV: {cv_f*100:.1f}%)   ║
║  V3 Model G: High-capacity GB                {acc_g*100:.1f}%  (CV: {cv_g*100:.1f}%)   ║
║                                                                      ║
║  BEST: {best_name:<40s}  {best_acc*100:.1f}%              ║
║  Total journey: 71.0% -> {best_acc*100:.1f}%                                ║
╚══════════════════════════════════════════════════════════════════════╝

Seasons: {seasons}
Training plays: {len(X_train):,}
Test plays: {len(X_test):,}
Total features: {len(best_feats)}
""")

# Save
joblib.dump(best_model, 'best_model_v3.joblib')
joblib.dump(best_feats, 'model_features_v3.joblib')
print(f"Saved: best_model_v3.joblib")
print(f"Saved: model_features_v3.joblib")
print(f"\nSend me this output!")
