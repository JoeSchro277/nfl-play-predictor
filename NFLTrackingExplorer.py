"""
NFL Run vs Pass Predictor - Phase 1: Tracking Data Explorer (v2)
================================================================
This script works with:
  1. NFL Big Data Bowl 2023 data (pass play tracking from 2021 season)
  2. nfl_data_py play-by-play data (ALL plays from 2021 season)

It merges them to give us both:
  - Detailed pre-snap positioning (from BDB tracking data)
  - Run AND pass play labels (from nfl_data_py)

SETUP:
1. Your BDB 2023 data should be in a folder. Update DATA_DIR below.
2. Install requirements: pip install pandas numpy nfl_data_py
3. Run this script in PyCharm

Your project structure:
   NFLTrackingPredictor/
   ├── NFLPredictorSTRONG.py
   ├── NFLTrackingExplorer.py      (this file)
   └── nfl-big-data-bowl-2023/     (or whatever you named it)
       ├── games.csv
       ├── plays.csv
       ├── players.csv
       ├── pffScoutingData.csv
       ├── week1.csv
       ├── week2.csv
       └── ... (through week8.csv)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — UPDATE THIS PATH
# ============================================================
DATA_DIR = 'nfl_tracking_data/nfl-big-data-bowl-2023'  # <-- Change this to match your folder name

# ============================================================
# STEP 1: LOAD BDB DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Big Data Bowl 2023 files...")
print("=" * 60)

games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'))
plays = pd.read_csv(os.path.join(DATA_DIR, 'plays.csv'))
players = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'))

print(f"Games: {len(games)} (2021 season, weeks 1-8)")
print(f"Plays: {len(plays)} (pass plays only from BDB)")
print(f"Players: {len(players)}")

# ============================================================
# STEP 2: LOAD ONE WEEK OF TRACKING DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading Week 1 tracking data (this may take a moment)...")
print("=" * 60)

tracking = pd.read_csv(os.path.join(DATA_DIR, 'week1.csv'))
print(f"Tracking rows loaded: {len(tracking):,}")
print(f"\nTracking columns:")
print(tracking.columns.tolist())
print(f"\nTracking data types:")
print(tracking.dtypes)
print(f"\nFirst 3 rows:")
print(tracking.head(3).to_string())

# ============================================================
# STEP 3: EXPLORE PRE-SNAP POSITIONING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Isolating pre-snap frames")
print("=" * 60)

# Find the snap event
print(f"\nAll events in tracking data:")
print(tracking['event'].value_counts().to_string())

# Filter to the snap frame
snap_events = tracking[tracking['event'].str.contains('snap', case=False, na=False)]
print(f"\nSnap-related events: {snap_events['event'].unique()}")

# Get tracking data at the moment of the snap
snap_frame_ids = snap_events[['gameId', 'playId', 'frameId']].drop_duplicates()

# Merge to get all players at the snap frame
tracking_at_snap = tracking.merge(
    snap_frame_ids,
    on=['gameId', 'playId', 'frameId'],
    how='inner'
)

print(f"\nPlayer records at snap: {len(tracking_at_snap):,}")
unique_plays_with_tracking = tracking_at_snap[['gameId', 'playId']].drop_duplicates()
print(f"Unique plays with snap data: {len(unique_plays_with_tracking)}")

# ============================================================
# STEP 4: EXAMINE A SINGLE PLAY
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Examining a single play at the snap")
print("=" * 60)

sample_game = tracking_at_snap['gameId'].iloc[0]
sample_play = tracking_at_snap['playId'].iloc[0]

one_play = tracking_at_snap[
    (tracking_at_snap['gameId'] == sample_game) &
    (tracking_at_snap['playId'] == sample_play)
]

# Show play description
play_info = plays[(plays['gameId'] == sample_game) & (plays['playId'] == sample_play)]
if len(play_info) > 0:
    print(f"Play: {play_info['playDescription'].values[0]}")
    print(f"Formation: {play_info['offenseFormation'].values[0]}")
    print(f"Personnel: {play_info['personnelO'].values[0]}")

print(f"\nAll {len(one_play)} entities at the snap:")
print(one_play.to_string())

# ============================================================
# STEP 5: LOAD NFL_DATA_PY FOR RUN + PASS PLAYS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Loading nfl_data_py for ALL plays (run + pass)")
print("=" * 60)

try:
    import nfl_data_py as nfl

    # Load 2021 season play-by-play (same season as BDB 2023 data)
    pbp = nfl.import_pbp_data([2021])
    print(f"Total plays from nfl_data_py: {len(pbp):,}")

    # Filter to run and pass plays only
    pbp_filtered = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
    print(f"Run + pass plays: {len(pbp_filtered):,}")
    print(f"\nPlay type breakdown:")
    print(pbp_filtered['play_type'].value_counts().to_string())

    # Filter to weeks 1-8 to match BDB data
    pbp_w1_8 = pbp_filtered[pbp_filtered['week'] <= 8].copy()
    print(f"\nWeeks 1-8 only: {len(pbp_w1_8):,}")
    print(pbp_w1_8['play_type'].value_counts().to_string())

    # Check what formation features nfl_data_py has for ALL plays
    formation_cols = ['shotgun', 'no_huddle', 'qb_dropback', 'defenders_in_box',
                      'number_of_pass_rushers', 'offense_formation',
                      'offense_personnel', 'defense_personnel', 'n_offense', 'n_defense']

    available_formation_cols = [c for c in formation_cols if c in pbp_w1_8.columns]
    print(f"\nFormation features available in nfl_data_py:")
    for col in available_formation_cols:
        non_null = pbp_w1_8[col].notna().sum()
        print(f"  {col}: {non_null}/{len(pbp_w1_8)} non-null values")

    # ============================================================
    # STEP 6: ATTEMPT TO MERGE BDB + NFL_DATA_PY
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Merging BDB tracking with nfl_data_py")
    print("=" * 60)

    # nfl_data_py uses 'old_game_id' which matches BDB's 'gameId' format
    if 'old_game_id' in pbp_w1_8.columns:
        pbp_w1_8['old_game_id_int'] = pbp_w1_8['old_game_id'].astype(str).str.replace('.0', '', regex=False)

        bdb_game_ids = set(plays['gameId'].astype(str))
        pbp_game_ids = set(pbp_w1_8['old_game_id_int'].astype(str))

        overlap = bdb_game_ids.intersection(pbp_game_ids)
        print(f"BDB game IDs: {len(bdb_game_ids)}")
        print(f"nfl_data_py game IDs (weeks 1-8): {len(pbp_game_ids)}")
        print(f"Overlapping game IDs: {len(overlap)}")

        if len(overlap) > 0:
            print(f"\nSample BDB game ID: {sorted(list(bdb_game_ids))[:3]}")
            print(f"Sample nfl_data_py game ID: {sorted(list(pbp_game_ids))[:3]}")
            print("\n*** GAME IDs MATCH — MERGE IS POSSIBLE! ***")

            # Try merging on play_id too
            # BDB uses 'playId', nfl_data_py might use 'play_id'
            print(f"\nBDB play ID column: 'playId', sample: {plays['playId'].head(3).tolist()}")
            if 'play_id' in pbp_w1_8.columns:
                print(f"nfl_data_py play ID column: 'play_id', sample: {pbp_w1_8['play_id'].head(3).tolist()}")
        else:
            print("\nGame ID formats don't match directly.")
            print(f"Sample BDB: {sorted(list(bdb_game_ids))[:3]}")
            print(f"Sample pbp: {sorted(list(pbp_game_ids))[:3]}")

    # ============================================================
    # STEP 7: FEATURE COMPARISON
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Feature comparison — what each dataset gives us")
    print("=" * 60)

    print("""
    BDB TRACKING DATA (pass plays only, weeks 1-8):
    + x, y position for all 22 players at the snap
    + Speed, acceleration, direction, orientation per player
    + Player positions (T, G, C, WR, TE, RB, etc.)
    + Offensive formation, personnel groupings
    - NO run plays

    NFL_DATA_PY (ALL plays, full season):
    + Run AND pass play labels (our target variable!)
    + shotgun (0/1), no_huddle (0/1)
    + defenders_in_box, offense_formation
    + down, distance, score, time remaining
    + Your existing 9 features from NFLPredictorSTRONG.py
    - No x/y player tracking

    MERGE STRATEGY:
    -> Use nfl_data_py as the BASE (has both run + pass)
    -> Join BDB tracking data where available (pass plays)
    -> Engineer formation-level features from BOTH sources
    -> The tracking data teaches us WHAT to look for
    -> nfl_data_py lets us apply it to ALL plays
    """)

except ImportError:
    print("nfl_data_py not installed! Run: pip install nfl_data_py")
    print("The BDB data exploration above still works without it.")

# ============================================================
# STEP 8: SUMMARY AND NEXT STEPS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: NEXT STEPS")
print("=" * 60)
print("""
COPY ALL OF THE OUTPUT ABOVE AND SEND IT TO ME.

I need to see:
  1. The tracking data columns (from Step 2)
  2. The snap event names (from Step 3)
  3. The single play details (from Step 4)
  4. Whether the merge works (from Step 6)

With that info, I'll build the feature engineering
pipeline that creates your enhanced model.
""")
