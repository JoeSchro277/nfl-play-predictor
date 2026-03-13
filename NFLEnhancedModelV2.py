"""
NFL Run vs Pass Predictor - Enhanced Model v2 (with FTN Charting Data)
======================================================================
Builds on NFLEnhancedModel.py by adding FTN charting features.
Goal: Beat the 74.3% from the previous model.

NEW features from FTN data:
  - is_motion (pre-snap motion)
  - n_offense_backfield (players in backfield)
  - n_defense_box (FTN's box count)
  - qb_location (S/U/P/O)
  - starting_hash (L/R/M/O)
"""

import pandas as pd
import numpy as np
import re
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD PBP + FTN DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading play-by-play and FTN charting data...")
print("=" * 60)

import nfl_data_py as nfl

# FTN data available from 2022 onward
seasons = [2022, 2023, 2024]

start = time.time()
pbp = nfl.import_pbp_data(seasons)
print(f"PBP loaded: {len(pbp):,} plays ({time.time()-start:.1f}s)")

start = time.time()
ftn = nfl.import_ftn_data(seasons)
print(f"FTN loaded: {len(ftn):,} plays ({time.time()-start:.1f}s)")

# Filter PBP to run and pass
pbp = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
print(f"\nRun + pass plays: {len(pbp):,}")
print(pbp['play_type'].value_counts().to_string())

# Target variable
pbp['is_pass'] = (pbp['play_type'] == 'pass').astype(int)

# ============================================================
# STEP 2: MERGE PBP WITH FTN
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Merging PBP with FTN data...")
print("=" * 60)

# FTN uses nflverse_game_id and nflverse_play_id
# PBP uses game_id and play_id
# Check what columns FTN has for joining
print(f"FTN join columns: nflverse_game_id, nflverse_play_id")
print(f"PBP join columns: game_id, play_id")

# Select only the pre-snap FTN features we want (avoid leakage)
ftn_features = ftn[[
    'nflverse_game_id', 'nflverse_play_id',
    'is_motion', 'n_offense_backfield', 'n_defense_box',
    'qb_location', 'starting_hash', 'is_no_huddle'
]].copy()

# Merge
pbp = pbp.merge(
    ftn_features,
    left_on=['game_id', 'play_id'],
    right_on=['nflverse_game_id', 'nflverse_play_id'],
    how='left'
)

ftn_matched = pbp['is_motion'].notna().sum()
print(f"Plays matched with FTN data: {ftn_matched:,}/{len(pbp):,} ({ftn_matched/len(pbp)*100:.1f}%)")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Engineering all features...")
print("=" * 60)

# --- Original features ---
pbp['home_or_away'] = (pbp['posteam'] == pbp['home_team']).astype(int)
if 'score_differential' not in pbp.columns:
    pbp['score_differential'] = pbp['posteam_score'] - pbp['defteam_score']

# --- Parse personnel ---
def parse_offense_personnel(personnel_str):
    if pd.isna(personnel_str):
        return {'n_rb': np.nan, 'n_te': np.nan, 'n_wr': np.nan, 'n_ol_extra': 0}
    result = {'n_rb': 0, 'n_te': 0, 'n_wr': 0, 'n_ol_extra': 0}
    for key, pattern in [('n_rb', r'(\d+)\s*RB'), ('n_te', r'(\d+)\s*TE'),
                          ('n_wr', r'(\d+)\s*WR'), ('n_ol_extra', r'(\d+)\s*OL')]:
        match = re.search(pattern, personnel_str)
        if match:
            result[key] = int(match.group(1))
    return result

def parse_defense_personnel(personnel_str):
    if pd.isna(personnel_str):
        return {'n_dl': np.nan, 'n_lb': np.nan, 'n_db': np.nan}
    result = {'n_dl': 0, 'n_lb': 0, 'n_db': 0}
    for key, pattern in [('n_dl', r'(\d+)\s*DL'), ('n_lb', r'(\d+)\s*LB'),
                          ('n_db', r'(\d+)\s*DB')]:
        match = re.search(pattern, personnel_str)
        if match:
            result[key] = int(match.group(1))
    return result

print("Parsing personnel...")
off_pers = pbp['offense_personnel'].apply(parse_offense_personnel).apply(pd.Series)
def_pers = pbp['defense_personnel'].apply(parse_defense_personnel).apply(pd.Series)
pbp = pd.concat([pbp, off_pers, def_pers], axis=1)

# --- Formation dummies ---
print("Encoding formations...")
formation_dummies = pd.get_dummies(pbp['offense_formation'], prefix='formation', dummy_na=False).astype(int)
pbp = pd.concat([pbp, formation_dummies], axis=1)
formation_cols = formation_dummies.columns.tolist()

# --- FTN feature encoding ---
print("Encoding FTN features...")

# Convert is_motion to numeric
pbp['ftn_motion'] = pbp['is_motion'].astype(float)

# n_offense_backfield is already numeric
pbp['ftn_backfield'] = pbp['n_offense_backfield'].astype(float)

# n_defense_box from FTN
pbp['ftn_defense_box'] = pbp['n_defense_box'].astype(float)

# QB location dummies
qb_loc_dummies = pd.get_dummies(pbp['qb_location'], prefix='qb_loc', dummy_na=False).astype(int)
pbp = pd.concat([pbp, qb_loc_dummies], axis=1)
qb_loc_cols = qb_loc_dummies.columns.tolist()

# Starting hash dummies
hash_dummies = pd.get_dummies(pbp['starting_hash'], prefix='hash', dummy_na=False).astype(int)
pbp = pd.concat([pbp, hash_dummies], axis=1)
hash_cols = hash_dummies.columns.tolist()

# --- Team run tendency ---
print("Calculating team run tendencies...")
pbp = pbp.sort_values(['posteam', 'game_id', 'play_id']).reset_index(drop=True)

game_run_ratio = pbp.groupby(['posteam', 'game_id']).agg(
    total_plays=('is_pass', 'count'),
    pass_plays=('is_pass', 'sum')
).reset_index()
game_run_ratio['run_ratio'] = 1 - (game_run_ratio['pass_plays'] / game_run_ratio['total_plays'])
game_run_ratio['team_ratio'] = game_run_ratio.groupby('posteam')['run_ratio'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
pbp = pbp.merge(game_run_ratio[['posteam', 'game_id', 'team_ratio']], on=['posteam', 'game_id'], how='left')

# --- Interaction features ---
print("Creating interaction features...")
# Score differential * time remaining (garbage time indicator)
pbp['score_time_interaction'] = pbp['score_differential'] * (pbp['game_seconds_remaining'] / 3600)
# Down and distance combo
pbp['down_distance_interaction'] = pbp['down'] * pbp['ydstogo']

# ============================================================
# STEP 4: DEFINE FEATURE SETS AND TRAIN
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Training models...")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Feature sets ---
original_features = [
    'down', 'ydstogo', 'yardline_100', 'score_differential',
    'game_seconds_remaining', 'shotgun', 'no_huddle', 'home_or_away'
]

personnel_features = [
    'defenders_in_box', 'n_rb', 'n_te', 'n_wr', 'n_ol_extra',
    'n_dl', 'n_lb', 'n_db'
]

ftn_numeric_features = [
    'ftn_motion', 'ftn_backfield', 'ftn_defense_box'
]

interaction_features = [
    'score_time_interaction', 'down_distance_interaction'
]

# Previous best (Model D/E from last script)
features_previous = original_features + personnel_features + formation_cols + ['team_ratio']

# New: + FTN features
features_v5 = features_previous + ftn_numeric_features + qb_loc_cols + hash_cols

# New: + interaction features
features_v6 = features_v5 + interaction_features

# --- Prepare data (use FTN-matched plays only) ---
all_features = features_v6
df = pbp[all_features + ['is_pass']].dropna()

print(f"Clean dataset: {len(df):,} plays")
print(f"  Pass: {(df['is_pass'] == 1).sum():,} ({(df['is_pass'] == 1).mean()*100:.1f}%)")
print(f"  Run: {(df['is_pass'] == 0).sum():,} ({(df['is_pass'] == 0).mean()*100:.1f}%)")
print(f"  Total features: {len(all_features)}")

X = df[all_features]
y = df['is_pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train):,}  |  Test: {len(X_test):,}")

# --- Model A: Previous best (RF on old features) ---
print("\n--- Model A: Previous best features (Random Forest) ---")
feats_prev = [f for f in features_previous if f in X_train.columns]
rf_prev = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_prev.fit(X_train[feats_prev], y_train)
acc_prev_rf = accuracy_score(y_test, rf_prev.predict(X_test[feats_prev]))
cv_prev_rf = cross_val_score(rf_prev, X_train[feats_prev], y_train, cv=5).mean()
print(f"  Test: {acc_prev_rf*100:.2f}%  |  CV: {cv_prev_rf*100:.2f}%")

# --- Model B: Previous best (GB on old features) ---
print("\n--- Model B: Previous best features (Gradient Boosting) ---")
gb_prev = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_prev.fit(X_train[feats_prev], y_train)
acc_prev_gb = accuracy_score(y_test, gb_prev.predict(X_test[feats_prev]))
cv_prev_gb = cross_val_score(gb_prev, X_train[feats_prev], y_train, cv=5).mean()
print(f"  Test: {acc_prev_gb*100:.2f}%  |  CV: {cv_prev_gb*100:.2f}%")

# --- Model C: + FTN features (RF) ---
print("\n--- Model C: + FTN features (Random Forest) ---")
feats_v5 = [f for f in features_v5 if f in X_train.columns]
rf_ftn = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_ftn.fit(X_train[feats_v5], y_train)
acc_ftn_rf = accuracy_score(y_test, rf_ftn.predict(X_test[feats_v5]))
cv_ftn_rf = cross_val_score(rf_ftn, X_train[feats_v5], y_train, cv=5).mean()
print(f"  Test: {acc_ftn_rf*100:.2f}%  |  CV: {cv_ftn_rf*100:.2f}%")

# --- Model D: + FTN features (GB) ---
print("\n--- Model D: + FTN features (Gradient Boosting) ---")
gb_ftn = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_ftn.fit(X_train[feats_v5], y_train)
acc_ftn_gb = accuracy_score(y_test, gb_ftn.predict(X_test[feats_v5]))
cv_ftn_gb = cross_val_score(gb_ftn, X_train[feats_v5], y_train, cv=5).mean()
print(f"  Test: {acc_ftn_gb*100:.2f}%  |  CV: {cv_ftn_gb*100:.2f}%")

# --- Model E: Full features + interactions (GB) ---
print("\n--- Model E: Full features + interactions (Gradient Boosting) ---")
feats_v6 = [f for f in features_v6 if f in X_train.columns]
gb_full = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
gb_full.fit(X_train[feats_v6], y_train)
acc_full_gb = accuracy_score(y_test, gb_full.predict(X_test[feats_v6]))
cv_full_gb = cross_val_score(gb_full, X_train[feats_v6], y_train, cv=5).mean()
print(f"  Test: {acc_full_gb*100:.2f}%  |  CV: {cv_full_gb*100:.2f}%")

# --- Model F: Tuned GB (more trees, lower learning rate) ---
print("\n--- Model F: Tuned Gradient Boosting (full features) ---")
gb_tuned = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    min_samples_leaf=20, subsample=0.8, random_state=42
)
gb_tuned.fit(X_train[feats_v6], y_train)
acc_tuned = accuracy_score(y_test, gb_tuned.predict(X_test[feats_v6]))
cv_tuned = cross_val_score(gb_tuned, X_train[feats_v6], y_train, cv=5).mean()
print(f"  Test: {acc_tuned*100:.2f}%  |  CV: {cv_tuned*100:.2f}%")

# ============================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Feature importance (best model)")
print("=" * 60)

# Use the best GB model
best_models = {
    'Model E (GB full)': (acc_full_gb, cv_full_gb, gb_full, feats_v6),
    'Model F (GB tuned)': (acc_tuned, cv_tuned, gb_tuned, feats_v6),
    'Model D (GB + FTN)': (acc_ftn_gb, cv_ftn_gb, gb_ftn, feats_v5),
}

best_name = max(best_models, key=lambda k: best_models[k][0])
best_acc, best_cv, best_model, best_feats = best_models[best_name]

importances = pd.DataFrame({
    'feature': best_feats,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nBest model: {best_name}")
print(f"\nAll features ranked:")
print(importances.to_string(index=False))

# ============================================================
# STEP 6: CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Classification report (best model)")
print("=" * 60)

best_preds = best_model.predict(X_test[best_feats])
print(classification_report(y_test, best_preds, target_names=['Run', 'Pass']))

# ============================================================
# STEP 7: RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: RESULTS SUMMARY")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                      ACCURACY COMPARISON                         ║
╠══════════════════════════════════════════════════════════════════╣
║  NFLPredictorSTRONG.py (original):          ~71.0%               ║
║  Previous best (NFLEnhancedModel.py):        74.3%               ║
║                                                                  ║
║  Model A: Previous features (RF)         {acc_prev_rf*100:.1f}%  (CV: {cv_prev_rf*100:.1f}%)  ║
║  Model B: Previous features (GB)         {acc_prev_gb*100:.1f}%  (CV: {cv_prev_gb*100:.1f}%)  ║
║  Model C: + FTN features (RF)            {acc_ftn_rf*100:.1f}%  (CV: {cv_ftn_rf*100:.1f}%)  ║
║  Model D: + FTN features (GB)            {acc_ftn_gb*100:.1f}%  (CV: {cv_ftn_gb*100:.1f}%)  ║
║  Model E: + Interactions (GB)            {acc_full_gb*100:.1f}%  (CV: {cv_full_gb*100:.1f}%)  ║
║  Model F: Tuned GB (full)                {acc_tuned*100:.1f}%  (CV: {cv_tuned*100:.1f}%)  ║
║                                                                  ║
║  BEST: {best_name:<38s} {best_acc*100:.1f}%               ║
╚══════════════════════════════════════════════════════════════════╝

Seasons: {seasons}
Training plays: {len(X_train):,}
Test plays: {len(X_test):,}
Total features: {len(best_feats)}
""")

# Save the best model
joblib.dump(best_model, 'best_model_v2.joblib')
joblib.dump(best_feats, 'model_features_v2.joblib')
print(f"Saved: best_model_v2.joblib")
print(f"Saved: model_features_v2.joblib")
print(f"\nSend me this output!")
