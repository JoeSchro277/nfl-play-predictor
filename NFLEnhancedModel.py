"""
NFL Run vs Pass Predictor - Enhanced Model with Formation Features
==================================================================
This script takes your original NFLPredictorSTRONG.py approach and
adds formation/personnel features available for ALL plays in nfl_data_py.

Goal: Beat your 71% baseline accuracy.

Features being added:
  - offense_formation (SHOTGUN, SINGLEBACK, I_FORM, etc.)
  - defenders_in_box (numeric)
  - n_rb, n_te, n_wr (parsed from offense_personnel)
  - n_dl, n_lb, n_db (parsed from defense_personnel)
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading nfl_data_py play-by-play data...")
print("=" * 60)

import nfl_data_py as nfl

# Use same seasons as your original model — adjust if needed
seasons = [2021, 2022, 2023, 2024]
pbp = nfl.import_pbp_data(seasons)

print(f"Total plays loaded: {len(pbp):,}")

# Filter to run and pass only
pbp = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
print(f"Run + pass plays: {len(pbp):,}")
print(f"\nPlay type breakdown:")
print(pbp['play_type'].value_counts().to_string())

# Target variable
pbp['is_pass'] = (pbp['play_type'] == 'pass').astype(int)

# ============================================================
# STEP 2: YOUR ORIGINAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Setting up original features...")
print("=" * 60)

# Replicate your NFLPredictorSTRONG.py features
# home_or_away needs to be derived
pbp['home_or_away'] = (pbp['posteam'] == pbp['home_team']).astype(int)

# Score differential from possession team's perspective
# (posteam_score - defteam_score)
if 'score_differential' not in pbp.columns:
    pbp['score_differential'] = pbp['posteam_score'] - pbp['defteam_score']

original_features = [
    'down',
    'ydstogo',           # yards_to_1st
    'yardline_100',      # yards_to_end_zone
    'score_differential',
    'game_seconds_remaining',
    'shotgun',
    'no_huddle',
    'home_or_away',
]

# Check availability
print("Original feature availability:")
for feat in original_features:
    non_null = pbp[feat].notna().sum()
    print(f"  {feat}: {non_null:,}/{len(pbp):,} ({non_null/len(pbp)*100:.1f}%)")

# ============================================================
# STEP 3: PARSE PERSONNEL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Parsing personnel groupings...")
print("=" * 60)

def parse_offense_personnel(personnel_str):
    """Parse '1 RB, 1 TE, 3 WR' into counts."""
    if pd.isna(personnel_str):
        return {'n_rb': np.nan, 'n_te': np.nan, 'n_wr': np.nan, 'n_ol_extra': 0}

    result = {'n_rb': 0, 'n_te': 0, 'n_wr': 0, 'n_ol_extra': 0}

    # Find patterns like "1 RB", "2 TE", etc.
    rb_match = re.search(r'(\d+)\s*RB', personnel_str)
    te_match = re.search(r'(\d+)\s*TE', personnel_str)
    wr_match = re.search(r'(\d+)\s*WR', personnel_str)
    ol_match = re.search(r'(\d+)\s*OL', personnel_str)

    if rb_match:
        result['n_rb'] = int(rb_match.group(1))
    if te_match:
        result['n_te'] = int(te_match.group(1))
    if wr_match:
        result['n_wr'] = int(wr_match.group(1))
    if ol_match:
        result['n_ol_extra'] = int(ol_match.group(1))

    return result

def parse_defense_personnel(personnel_str):
    """Parse '4 DL, 2 LB, 5 DB' into counts."""
    if pd.isna(personnel_str):
        return {'n_dl': np.nan, 'n_lb': np.nan, 'n_db': np.nan}

    result = {'n_dl': 0, 'n_lb': 0, 'n_db': 0}

    dl_match = re.search(r'(\d+)\s*DL', personnel_str)
    lb_match = re.search(r'(\d+)\s*LB', personnel_str)
    db_match = re.search(r'(\d+)\s*DB', personnel_str)

    if dl_match:
        result['n_dl'] = int(dl_match.group(1))
    if lb_match:
        result['n_lb'] = int(lb_match.group(1))
    if db_match:
        result['n_db'] = int(db_match.group(1))

    return result

# Parse offensive personnel
print("Parsing offensive personnel...")
off_personnel = pbp['offense_personnel'].apply(parse_offense_personnel).apply(pd.Series)
pbp = pd.concat([pbp, off_personnel], axis=1)

# Parse defensive personnel
print("Parsing defensive personnel...")
def_personnel = pbp['defense_personnel'].apply(parse_defense_personnel).apply(pd.Series)
pbp = pd.concat([pbp, def_personnel], axis=1)

print(f"\nOffensive personnel breakdown (averages):")
print(f"  RBs: {pbp['n_rb'].mean():.2f}")
print(f"  TEs: {pbp['n_te'].mean():.2f}")
print(f"  WRs: {pbp['n_wr'].mean():.2f}")
print(f"  Extra OL: {pbp['n_ol_extra'].mean():.2f}")

print(f"\nDefensive personnel breakdown (averages):")
print(f"  DL: {pbp['n_dl'].mean():.2f}")
print(f"  LB: {pbp['n_lb'].mean():.2f}")
print(f"  DB: {pbp['n_db'].mean():.2f}")

# ============================================================
# STEP 4: ENCODE FORMATION FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Encoding formation features...")
print("=" * 60)

# Offense formation — one-hot encode the main formations
print(f"\nOffense formation distribution:")
print(pbp['offense_formation'].value_counts().to_string())

# Create dummy variables for the main formations
formation_dummies = pd.get_dummies(
    pbp['offense_formation'],
    prefix='formation',
    dummy_na=False
).astype(int)

pbp = pd.concat([pbp, formation_dummies], axis=1)

formation_cols = [col for col in formation_dummies.columns]
print(f"\nFormation features created: {formation_cols}")

# ============================================================
# STEP 5: BUILD FEATURE SETS AND TRAIN MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training models...")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- Define feature sets ---

# Set 1: Your original features (baseline)
features_v1 = original_features.copy()

# Set 2: Original + personnel counts
features_v2 = original_features + [
    'defenders_in_box',
    'n_rb', 'n_te', 'n_wr', 'n_ol_extra',
    'n_dl', 'n_lb', 'n_db',
]

# Set 3: Original + personnel + formation dummies
features_v3 = features_v2 + formation_cols

# --- Team run tendency (your team_ratio concept) ---
print("\nCalculating team run tendencies (rolling 3-game average)...")

# Sort by team and game for rolling calculation
pbp = pbp.sort_values(['posteam', 'game_id', 'play_id']).reset_index(drop=True)

# Calculate per-game run ratio for each team
game_run_ratio = pbp.groupby(['posteam', 'game_id']).agg(
    total_plays=('is_pass', 'count'),
    pass_plays=('is_pass', 'sum')
).reset_index()
game_run_ratio['run_ratio'] = 1 - (game_run_ratio['pass_plays'] / game_run_ratio['total_plays'])

# Rolling 3-game average (shifted so we don't leak current game)
game_run_ratio['team_ratio'] = game_run_ratio.groupby('posteam')['run_ratio'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# Merge back
pbp = pbp.merge(
    game_run_ratio[['posteam', 'game_id', 'team_ratio']],
    on=['posteam', 'game_id'],
    how='left'
)

print(f"team_ratio available for: {pbp['team_ratio'].notna().sum():,}/{len(pbp):,} plays")

# Set 4: Everything including team_ratio
features_v4 = features_v3 + ['team_ratio']

# --- Prepare data ---
# Use the most complete feature set and drop NaNs
all_features = features_v4
df = pbp[all_features + ['is_pass']].dropna()

print(f"\nClean dataset: {len(df):,} plays")
print(f"  Pass: {(df['is_pass'] == 1).sum():,} ({(df['is_pass'] == 1).mean()*100:.1f}%)")
print(f"  Run: {(df['is_pass'] == 0).sum():,} ({(df['is_pass'] == 0).mean()*100:.1f}%)")

X = df[all_features]
y = df['is_pass']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train):,}  |  Test: {len(X_test):,}")

# --- Train Model A: Baseline (original features) ---
print("\n--- Training Model A: Baseline (original features) ---")
X_train_v1 = X_train[features_v1]
X_test_v1 = X_test[features_v1]

rf_v1 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_v1.fit(X_train_v1, y_train)
acc_v1 = accuracy_score(y_test, rf_v1.predict(X_test_v1))
cv_v1 = cross_val_score(rf_v1, X_train_v1, y_train, cv=5, scoring='accuracy').mean()
print(f"  Test Accuracy: {acc_v1*100:.2f}%  |  CV Accuracy: {cv_v1*100:.2f}%")

# --- Train Model B: + Personnel ---
print("\n--- Training Model B: + Personnel counts ---")
feats_v2_available = [f for f in features_v2 if f in X_train.columns]
X_train_v2 = X_train[feats_v2_available]
X_test_v2 = X_test[feats_v2_available]

rf_v2 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_v2.fit(X_train_v2, y_train)
acc_v2 = accuracy_score(y_test, rf_v2.predict(X_test_v2))
cv_v2 = cross_val_score(rf_v2, X_train_v2, y_train, cv=5, scoring='accuracy').mean()
print(f"  Test Accuracy: {acc_v2*100:.2f}%  |  CV Accuracy: {cv_v2*100:.2f}%")

# --- Train Model C: + Formation ---
print("\n--- Training Model C: + Formation dummies ---")
feats_v3_available = [f for f in features_v3 if f in X_train.columns]
X_train_v3 = X_train[feats_v3_available]
X_test_v3 = X_test[feats_v3_available]

rf_v3 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_v3.fit(X_train_v3, y_train)
acc_v3 = accuracy_score(y_test, rf_v3.predict(X_test_v3))
cv_v3 = cross_val_score(rf_v3, X_train_v3, y_train, cv=5, scoring='accuracy').mean()
print(f"  Test Accuracy: {acc_v3*100:.2f}%  |  CV Accuracy: {cv_v3*100:.2f}%")

# --- Train Model D: + Team ratio (full feature set) ---
print("\n--- Training Model D: Full feature set + team_ratio ---")
X_train_v4 = X_train[all_features]
X_test_v4 = X_test[all_features]

rf_v4 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_v4.fit(X_train_v4, y_train)
acc_v4 = accuracy_score(y_test, rf_v4.predict(X_test_v4))
cv_v4 = cross_val_score(rf_v4, X_train_v4, y_train, cv=5, scoring='accuracy').mean()
print(f"  Test Accuracy: {acc_v4*100:.2f}%  |  CV Accuracy: {cv_v4*100:.2f}%")

# --- Train Model E: Gradient Boosting (full features) ---
print("\n--- Training Model E: Gradient Boosting (full features) ---")
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
)
gb.fit(X_train_v4, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_v4))
cv_gb = cross_val_score(gb, X_train_v4, y_train, cv=5, scoring='accuracy').mean()
print(f"  Test Accuracy: {acc_gb*100:.2f}%  |  CV Accuracy: {cv_gb*100:.2f}%")

# ============================================================
# STEP 6: FEATURE IMPORTANCE (best model)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Feature importance (best Random Forest model)")
print("=" * 60)

importances = pd.DataFrame({
    'feature': all_features,
    'importance': rf_v4.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.to_string(index=False))

# ============================================================
# STEP 7: DETAILED CLASSIFICATION REPORT (best model)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Classification report (best model)")
print("=" * 60)

# Pick best model
best_acc = max(acc_v4, acc_gb)
if acc_gb >= acc_v4:
    best_model = gb
    best_name = "Gradient Boosting"
    best_preds = gb.predict(X_test_v4)
else:
    best_model = rf_v4
    best_name = "Random Forest (full)"
    best_preds = rf_v4.predict(X_test_v4)

print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_preds, target_names=['Run', 'Pass']))

# ============================================================
# STEP 8: RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: RESULTS SUMMARY")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ACCURACY COMPARISON                       ║
╠══════════════════════════════════════════════════════════════╣
║  NFLPredictorSTRONG.py (your original):   ~71.0%            ║
║                                                              ║
║  Model A: Original 7 features             {acc_v1*100:.1f}%  (CV: {cv_v1*100:.1f}%) ║
║  Model B: + Personnel counts              {acc_v2*100:.1f}%  (CV: {cv_v2*100:.1f}%) ║
║  Model C: + Formation encoding            {acc_v3*100:.1f}%  (CV: {cv_v3*100:.1f}%) ║
║  Model D: + Team run tendency             {acc_v4*100:.1f}%  (CV: {cv_v4*100:.1f}%) ║
║  Model E: Gradient Boosting (all)         {acc_gb*100:.1f}%  (CV: {cv_gb*100:.1f}%) ║
║                                                              ║
║  Best model: {best_name:<30s}  {best_acc*100:.1f}%            ║
╚══════════════════════════════════════════════════════════════╝

Seasons used: {seasons}
Total training plays: {len(X_train):,}
Total test plays: {len(X_test):,}
Total features: {len(all_features)}
""")

# Save the best model for web app use
import joblib
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(all_features, 'model_features.joblib')
print(f"Saved best model to: best_model.joblib")
print(f"Saved feature list to: model_features.joblib")
print(f"\nSend me this full output and we'll build the web app next!")
