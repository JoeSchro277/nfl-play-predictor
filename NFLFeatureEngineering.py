"""
NFL Run vs Pass Predictor - Phase 1: Feature Engineering Pipeline
=================================================================
This script:
  1. Loads ALL 8 weeks of BDB tracking data
  2. Extracts pre-snap positioning features for every play
  3. Merges with nfl_data_py to get BOTH run and pass plays
  4. Creates your enhanced feature set
  5. Trains a model and compares to your 71% baseline

FEATURES WE'RE ENGINEERING FROM TRACKING DATA:
  - OL orientation spread (your dad's insight about tackle tells)
  - Receiver distribution width / depth
  - RB alignment relative to QB
  - TE inline vs. split positioning
  - Offensive formation compactness
  - Pre-snap motion indicators
  - Defenders in box positioning

Run this AFTER NFLTrackingExplorer.py has confirmed everything works.
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = 'nfl_tracking_data/nfl-big-data-bowl-2023'

# ============================================================
# STEP 1: LOAD ALL SUPPORT FILES
# ============================================================
print("=" * 60)
print("STEP 1: Loading support files...")
print("=" * 60)

games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'))
plays_bdb = pd.read_csv(os.path.join(DATA_DIR, 'plays.csv'))
players = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'))

# Create a position lookup: nflId -> officialPosition
player_positions = players.set_index('nflId')['officialPosition'].to_dict()

print(f"Games: {len(games)}")
print(f"BDB Plays: {len(plays_bdb)}")
print(f"Players: {len(players)}")
print(f"\nPlayer positions available:")
print(players['officialPosition'].value_counts().to_string())

# ============================================================
# STEP 2: LOAD AND PROCESS ALL TRACKING WEEKS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading tracking data (all 8 weeks)...")
print("=" * 60)

all_snap_data = []

for week in range(1, 9):
    start = time.time()
    filepath = os.path.join(DATA_DIR, f'week{week}.csv')
    print(f"  Loading week {week}...", end=" ", flush=True)

    tracking = pd.read_csv(filepath)

    # Filter to ball_snap events only (not autoevent_ballsnap to avoid duplicates)
    snap_data = tracking[tracking['event'] == 'ball_snap'].copy()

    # Add player positions from the players table
    snap_data['position'] = snap_data['nflId'].map(player_positions)

    all_snap_data.append(snap_data)
    elapsed = time.time() - start
    print(f"{len(snap_data):,} snap records ({elapsed:.1f}s)")

# Combine all weeks
snap_all = pd.concat(all_snap_data, ignore_index=True)
print(f"\nTotal snap records across all weeks: {len(snap_all):,}")
unique_plays = snap_all[['gameId', 'playId']].drop_duplicates()
print(f"Unique plays with tracking at snap: {len(unique_plays):,}")

# ============================================================
# STEP 3: IDENTIFY OFFENSE VS DEFENSE PER PLAY
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Identifying offense vs defense...")
print("=" * 60)

# Merge with plays to get possessionTeam
snap_all = snap_all.merge(
    plays_bdb[['gameId', 'playId', 'possessionTeam', 'offenseFormation',
               'personnelO', 'defendersInBox']],
    on=['gameId', 'playId'],
    how='inner'
)

# Tag each player as offense or defense
snap_all['side'] = np.where(
    snap_all['team'] == snap_all['possessionTeam'],
    'offense',
    np.where(snap_all['team'] == 'football', 'football', 'defense')
)

print(f"Records after merge with plays: {len(snap_all):,}")
print(f"\nSide breakdown:")
print(snap_all['side'].value_counts().to_string())

# ============================================================
# STEP 4: NORMALIZE COORDINATES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Normalizing coordinates (all plays left-to-right)...")
print("=" * 60)

# When playDirection is 'left', we need to flip x and y
# so all plays go in the same direction (offense moving right)
# x-axis: 0-120 (length), y-axis: 0-53.3 (width)

mask_left = snap_all['playDirection'] == 'left'
snap_all.loc[mask_left, 'x'] = 120 - snap_all.loc[mask_left, 'x']
snap_all.loc[mask_left, 'y'] = 53.3 - snap_all.loc[mask_left, 'y']

# Also flip orientation and direction by 180 degrees
snap_all.loc[mask_left, 'o'] = (snap_all.loc[mask_left, 'o'] + 180) % 360
snap_all.loc[mask_left, 'dir'] = (snap_all.loc[mask_left, 'dir'] + 180) % 360

print(f"Flipped {mask_left.sum():,} records from 'left' to 'right'")
print("All plays now normalized: offense moving left-to-right")

# ============================================================
# STEP 5: ENGINEER POSITIONAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Engineering positional features...")
print("=" * 60)

def compute_features(play_group):
    """Extract formation features from a single play's snap data."""
    game_id = play_group['gameId'].iloc[0]
    play_id = play_group['playId'].iloc[0]

    offense = play_group[play_group['side'] == 'offense']
    defense = play_group[play_group['side'] == 'defense']

    # Find specific position groups
    o_line = offense[offense['position'].isin(['T', 'G', 'C'])]
    receivers = offense[offense['position'].isin(['WR'])]
    tight_ends = offense[offense['position'].isin(['TE'])]
    running_backs = offense[offense['position'].isin(['RB', 'FB'])]
    qb = offense[offense['position'] == 'QB']

    features = {
        'gameId': game_id,
        'playId': play_id,
    }

    # --- OFFENSIVE LINE FEATURES (Dad's insight!) ---
    if len(o_line) >= 3:
        # OL orientation spread: how varied are the linemen's body angles?
        # Tight spread = likely run block (all facing same way)
        # Wide spread = likely pass protection (fanning out)
        features['ol_orientation_std'] = o_line['o'].std()
        features['ol_orientation_mean'] = o_line['o'].mean()

        # OL y-spread: how wide is the offensive line?
        features['ol_y_spread'] = o_line['y'].max() - o_line['y'].min()

        # OL depth spread: are linemen at different depths? (pulling guard indicator)
        features['ol_x_spread'] = o_line['x'].max() - o_line['x'].min()

        # Average OL speed at snap (movement = pulling/sliding)
        features['ol_avg_speed'] = o_line['s'].mean()

        # Average OL acceleration at snap
        features['ol_avg_accel'] = o_line['a'].mean()
    else:
        features['ol_orientation_std'] = np.nan
        features['ol_orientation_mean'] = np.nan
        features['ol_y_spread'] = np.nan
        features['ol_x_spread'] = np.nan
        features['ol_avg_speed'] = np.nan
        features['ol_avg_accel'] = np.nan

    # --- RECEIVER FEATURES ---
    if len(receivers) >= 1:
        # How spread out are the receivers? (wide = likely pass)
        features['wr_y_spread'] = receivers['y'].max() - receivers['y'].min()

        # Average WR depth behind line of scrimmage
        if len(qb) > 0:
            los_x = qb['x'].iloc[0]  # Approximate LOS from QB position
        else:
            los_x = offense['x'].max()
        features['wr_avg_depth'] = (receivers['x'] - los_x).mean()

        # Number of receivers
        features['n_receivers'] = len(receivers)

        # WR average speed at snap (motion indicator)
        features['wr_avg_speed'] = receivers['s'].mean()
    else:
        features['wr_y_spread'] = np.nan
        features['wr_avg_depth'] = np.nan
        features['n_receivers'] = 0
        features['wr_avg_speed'] = np.nan

    # --- TIGHT END FEATURES ---
    if len(tight_ends) >= 1:
        features['n_tight_ends'] = len(tight_ends)

        # TE distance from nearest OL (inline vs. split out)
        if len(o_line) > 0:
            ol_y_center = o_line['y'].mean()
            features['te_distance_from_ol'] = (tight_ends['y'] - ol_y_center).abs().mean()
        else:
            features['te_distance_from_ol'] = np.nan

        # TE orientation (facing line = block, facing field = route)
        features['te_orientation_mean'] = tight_ends['o'].mean()
    else:
        features['n_tight_ends'] = 0
        features['te_distance_from_ol'] = np.nan
        features['te_orientation_mean'] = np.nan

    # --- RUNNING BACK FEATURES ---
    if len(running_backs) >= 1 and len(qb) > 0:
        qb_x = qb['x'].iloc[0]
        qb_y = qb['y'].iloc[0]

        # RB position relative to QB
        features['rb_x_offset'] = (running_backs['x'] - qb_x).mean()
        features['rb_y_offset'] = (running_backs['y'] - qb_y).mean()

        # RB distance from QB
        rb_dist = np.sqrt(
            (running_backs['x'] - qb_x) ** 2 +
            (running_backs['y'] - qb_y) ** 2
        )
        features['rb_distance_from_qb'] = rb_dist.mean()

        features['n_running_backs'] = len(running_backs)
    else:
        features['rb_x_offset'] = np.nan
        features['rb_y_offset'] = np.nan
        features['rb_distance_from_qb'] = np.nan
        features['n_running_backs'] = 0

    # --- QB FEATURES ---
    if len(qb) > 0:
        # QB depth behind line (shotgun depth vs. under center)
        if len(o_line) > 0:
            features['qb_depth'] = o_line['x'].mean() - qb['x'].iloc[0]
        else:
            features['qb_depth'] = np.nan
    else:
        features['qb_depth'] = np.nan

    # --- OVERALL FORMATION FEATURES ---
    # Offense spread: how spread out is the entire offense?
    features['offense_y_spread'] = offense['y'].max() - offense['y'].min()
    features['offense_x_spread'] = offense['x'].max() - offense['x'].min()

    # Offense centroid
    features['offense_centroid_y'] = offense['y'].mean()

    # Average offense speed at snap
    features['offense_avg_speed'] = offense['s'].mean()

    # --- DEFENSIVE FEATURES ---
    if len(defense) >= 1:
        # Defensive spread
        features['defense_y_spread'] = defense['y'].max() - defense['y'].min()

        # Defenders close to LOS (within 3 yards)
        if len(o_line) > 0:
            los_x = o_line['x'].mean()
            features['defenders_near_los'] = len(
                defense[defense['x'].between(los_x - 1, los_x + 4)]
            )
        else:
            features['defenders_near_los'] = np.nan

        # Average defender depth
        features['defense_avg_depth'] = defense['x'].mean()
    else:
        features['defense_y_spread'] = np.nan
        features['defenders_near_los'] = np.nan
        features['defense_avg_depth'] = np.nan

    return features

# Process all plays
print("Extracting features from each play (this will take a few minutes)...")
start = time.time()

play_groups = snap_all[snap_all['side'] != 'football'].groupby(['gameId', 'playId'])

feature_list = []
count = 0
total = len(play_groups)

for (game_id, play_id), group in play_groups:
    features = compute_features(group)
    feature_list.append(features)
    count += 1
    if count % 1000 == 0:
        elapsed = time.time() - start
        rate = count / elapsed
        remaining = (total - count) / rate
        print(f"  Processed {count:,}/{total:,} plays ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

tracking_features = pd.DataFrame(feature_list)
elapsed = time.time() - start
print(f"\nDone! Extracted {len(tracking_features)} play feature sets in {elapsed:.1f}s")

print(f"\n--- Feature Summary ---")
print(tracking_features.describe().round(2).to_string())

# ============================================================
# STEP 6: LOAD NFL_DATA_PY AND MERGE
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Loading nfl_data_py and merging...")
print("=" * 60)

import nfl_data_py as nfl

pbp = nfl.import_pbp_data([2021])
pbp = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
pbp = pbp[pbp['week'] <= 8].copy()

print(f"nfl_data_py plays (weeks 1-8, run+pass): {len(pbp):,}")
print(f"  Pass: {(pbp['play_type'] == 'pass').sum():,}")
print(f"  Run: {(pbp['play_type'] == 'run').sum():,}")

# Create target variable
pbp['is_pass'] = (pbp['play_type'] == 'pass').astype(int)

# Match game ID formats
# nfl_data_py 'old_game_id' should match BDB 'gameId'
pbp['gameId'] = pbp['old_game_id_x'].astype(int)
pbp['playId'] = pbp['play_id'].astype(int)

# Select the features from nfl_data_py that we already know work
pbp_features = pbp[[
    'gameId', 'playId', 'is_pass',
    'down', 'ydstogo', 'yardline_100',
    'score_differential', 'game_seconds_remaining',
    'shotgun', 'no_huddle',
    'defenders_in_box', 'offense_formation',
    'week', 'posteam', 'defteam'
]].copy()

# Rename to match your NFLPredictorSTRONG.py naming
pbp_features = pbp_features.rename(columns={
    'ydstogo': 'yards_to_1st',
    'yardline_100': 'yards_to_end_zone'
})

print(f"\nnfl_data_py features ready: {len(pbp_features):,} plays")

# ============================================================
# STEP 7: MERGE TRACKING FEATURES WITH PBP
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Merging tracking features with play-by-play...")
print("=" * 60)

# Merge: pbp_features is the BASE (has run + pass)
# tracking_features joins where available (pass plays with tracking)
merged = pbp_features.merge(
    tracking_features,
    on=['gameId', 'playId'],
    how='left'  # Keep ALL plays, even without tracking
)

has_tracking = merged[tracking_features.columns[2:]].notna().any(axis=1)
print(f"Total plays: {len(merged):,}")
print(f"  With tracking data: {has_tracking.sum():,}")
print(f"  Without tracking data: {(~has_tracking).sum():,}")
print(f"\nPlay type breakdown for plays WITH tracking:")
print(merged[has_tracking]['is_pass'].value_counts().rename({1: 'pass', 0: 'run'}).to_string())
print(f"\nPlay type breakdown for plays WITHOUT tracking:")
print(merged[~has_tracking]['is_pass'].value_counts().rename({1: 'pass', 0: 'run'}).to_string())

# ============================================================
# STEP 8: BUILD AND COMPARE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Training models and comparing accuracy...")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# --- MODEL A: Baseline (your original features only) ---
baseline_features = ['down', 'yards_to_1st', 'yards_to_end_zone',
                     'score_differential', 'game_seconds_remaining',
                     'shotgun', 'no_huddle']

df_baseline = merged[baseline_features + ['is_pass']].dropna()
X_base = df_baseline[baseline_features]
y_base = df_baseline['is_pass']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_b, y_train_b)
baseline_acc = accuracy_score(y_test_b, rf_baseline.predict(X_test_b))

print(f"\n--- MODEL A: Baseline (original features) ---")
print(f"Features: {baseline_features}")
print(f"Training samples: {len(X_train_b):,}")
print(f"Test samples: {len(X_test_b):,}")
print(f"Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")

# --- MODEL B: Enhanced (original + tracking features) ---
tracking_feature_cols = [col for col in tracking_features.columns
                         if col not in ['gameId', 'playId']]

enhanced_features = baseline_features + tracking_feature_cols

# Only use plays that have tracking data
df_enhanced = merged[has_tracking][enhanced_features + ['is_pass']].copy()

# Impute missing values in tracking features
imputer = SimpleImputer(strategy='median')
X_enh = pd.DataFrame(
    imputer.fit_transform(df_enhanced[enhanced_features]),
    columns=enhanced_features
)
y_enh = df_enhanced['is_pass'].values

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_enh, y_enh, test_size=0.2, random_state=42
)

rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_enhanced.fit(X_train_e, y_train_e)
enhanced_acc = accuracy_score(y_test_e, rf_enhanced.predict(X_test_e))

print(f"\n--- MODEL B: Enhanced (original + tracking features) ---")
print(f"Features: {len(enhanced_features)} total")
print(f"  Original: {len(baseline_features)}")
print(f"  Tracking: {len(tracking_feature_cols)}")
print(f"Training samples: {len(X_train_e):,}")
print(f"Test samples: {len(X_test_e):,}")
print(f"Accuracy: {enhanced_acc:.4f} ({enhanced_acc*100:.1f}%)")

# --- MODEL C: Baseline on same subset (fair comparison) ---
# Compare baseline on ONLY the plays that have tracking data
df_fair = merged[has_tracking][baseline_features + ['is_pass']].dropna()
X_fair = df_fair[baseline_features]
y_fair = df_fair['is_pass']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fair, y_fair, test_size=0.2, random_state=42
)

rf_fair = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fair.fit(X_train_f, y_train_f)
fair_acc = accuracy_score(y_test_f, rf_fair.predict(X_test_f))

print(f"\n--- MODEL C: Baseline on tracking-only subset (fair comparison) ---")
print(f"Accuracy: {fair_acc:.4f} ({fair_acc*100:.1f}%)")

# ============================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Feature importance (what matters most)")
print("=" * 60)

importances = pd.DataFrame({
    'feature': enhanced_features,
    'importance': rf_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(importances.head(20).to_string(index=False))

# ============================================================
# STEP 10: RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: RESULTS SUMMARY")
print("=" * 60)

improvement = enhanced_acc - fair_acc
print(f"""
╔══════════════════════════════════════════════════════════╗
║  YOUR ORIGINAL NFLPredictorSTRONG.py:        ~71.0%     ║
║  Model A (baseline, all plays):              {baseline_acc*100:.1f}%     ║
║  Model C (baseline, tracking subset only):   {fair_acc*100:.1f}%     ║
║  Model B (enhanced + tracking features):     {enhanced_acc*100:.1f}%     ║
║                                                          ║
║  IMPROVEMENT FROM TRACKING DATA:  {improvement*100:+.1f}%               ║
╚══════════════════════════════════════════════════════════╝

NOTE: The BDB tracking data only has PASS plays, so Models B 
and C are trained on a pass-only subset. The real power comes
when we can get tracking data for BOTH run and pass plays.

NEXT STEPS:
  1. Review the feature importance list above
  2. Send me the full output so I can analyze results
  3. We'll refine features and build the web app (Phase 2)
""")

# Save the merged dataset for later use
output_path = 'merged_tracking_pbp.csv'
merged.to_csv(output_path, index=False)
print(f"Saved merged dataset to: {output_path}")
print(f"Total columns: {len(merged.columns)}")
print(f"Total rows: {len(merged):,}")
