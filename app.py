"""
NFL Run vs Pass Predictor — Web App (v2)
==========================================
Three modes:
  1. Game Replay: Replay real NFL games play-by-play with predictions
  2. Analyze: Upload pre-snap screenshot for CV-based prediction (coming soon)
  3. Quick Predict: Manual input for real-time prediction

Usage:
  1. Run prepare_replay_data.py first (downloads and processes games)
  2. python app.py
  3. Open http://localhost:8080
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import json
import os
import glob
import base64
import io

app = Flask(__name__)

# ============================================================
# LOAD MODEL, STATS, AND ANALYZER
# ============================================================
print("Loading model...")
MODEL_PATH = 'best_model_v3.joblib'
FEATURES_PATH = 'model_features_v3.joblib'
STATS_PATH = 'team_stats.json'
REPLAY_DIR = 'replay_games'
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)

# Load feature names — try v3 first, fall back
if os.path.exists(FEATURES_PATH):
    feature_names = joblib.load(FEATURES_PATH)
elif os.path.exists('model_features.joblib'):
    feature_names = joblib.load('model_features.joblib')
    print("WARNING: Using older model_features.joblib — run prepare_replay_data.py to generate v3")
else:
    raise FileNotFoundError("No model features file found. Run prepare_replay_data.py first.")

print(f"Model loaded: {model.n_features_in_} features expected, {len(feature_names)} in feature list")

with open(STATS_PATH, 'r') as f:
    team_stats = json.load(f)

print(f"Team stats loaded: {len(team_stats)} teams")

# Load game index if available
game_index = []
game_index_path = os.path.join(REPLAY_DIR, 'game_index.json')
if os.path.exists(game_index_path):
    with open(game_index_path, 'r') as f:
        game_index = json.load(f)
    print(f"Game index loaded: {len(game_index)} games available for replay")
else:
    print(f"WARNING: No game index found at {game_index_path}")
    print("         Run prepare_replay_data.py to generate replay data")

# NFL team metadata for the UI
TEAM_META = {
    'ARI': {'name': 'Cardinals', 'city': 'Arizona', 'color': '#97233F', 'color2': '#000000'},
    'ATL': {'name': 'Falcons', 'city': 'Atlanta', 'color': '#A71930', 'color2': '#000000'},
    'BAL': {'name': 'Ravens', 'city': 'Baltimore', 'color': '#241773', 'color2': '#9E7C0C'},
    'BUF': {'name': 'Bills', 'city': 'Buffalo', 'color': '#00338D', 'color2': '#C60C30'},
    'CAR': {'name': 'Panthers', 'city': 'Carolina', 'color': '#0085CA', 'color2': '#101820'},
    'CHI': {'name': 'Bears', 'city': 'Chicago', 'color': '#0B162A', 'color2': '#C83803'},
    'CIN': {'name': 'Bengals', 'city': 'Cincinnati', 'color': '#FB4F14', 'color2': '#000000'},
    'CLE': {'name': 'Browns', 'city': 'Cleveland', 'color': '#311D00', 'color2': '#FF3C00'},
    'DAL': {'name': 'Cowboys', 'city': 'Dallas', 'color': '#003594', 'color2': '#869397'},
    'DEN': {'name': 'Broncos', 'city': 'Denver', 'color': '#FB4F14', 'color2': '#002244'},
    'DET': {'name': 'Lions', 'city': 'Detroit', 'color': '#0076B6', 'color2': '#B0B7BC'},
    'GB':  {'name': 'Packers', 'city': 'Green Bay', 'color': '#203731', 'color2': '#FFB612'},
    'HOU': {'name': 'Texans', 'city': 'Houston', 'color': '#03202F', 'color2': '#A71930'},
    'IND': {'name': 'Colts', 'city': 'Indianapolis', 'color': '#002C5F', 'color2': '#A2AAAD'},
    'JAX': {'name': 'Jaguars', 'city': 'Jacksonville', 'color': '#006778', 'color2': '#9F792C'},
    'KC':  {'name': 'Chiefs', 'city': 'Kansas City', 'color': '#E31837', 'color2': '#FFB81C'},
    'LA':  {'name': 'Rams', 'city': 'Los Angeles', 'color': '#003594', 'color2': '#FFA300'},
    'LAC': {'name': 'Chargers', 'city': 'Los Angeles', 'color': '#0080C6', 'color2': '#FFC20E'},
    'LV':  {'name': 'Raiders', 'city': 'Las Vegas', 'color': '#000000', 'color2': '#A5ACAF'},
    'MIA': {'name': 'Dolphins', 'city': 'Miami', 'color': '#008E97', 'color2': '#FC4C02'},
    'MIN': {'name': 'Vikings', 'city': 'Minnesota', 'color': '#4F2683', 'color2': '#FFC62F'},
    'NE':  {'name': 'Patriots', 'city': 'New England', 'color': '#002244', 'color2': '#C60C30'},
    'NO':  {'name': 'Saints', 'city': 'New Orleans', 'color': '#D3BC8D', 'color2': '#101820'},
    'NYG': {'name': 'Giants', 'city': 'New York', 'color': '#0B2265', 'color2': '#A71930'},
    'NYJ': {'name': 'Jets', 'city': 'New York', 'color': '#125740', 'color2': '#000000'},
    'PHI': {'name': 'Eagles', 'city': 'Philadelphia', 'color': '#004C54', 'color2': '#A5ACAF'},
    'PIT': {'name': 'Steelers', 'city': 'Pittsburgh', 'color': '#FFB612', 'color2': '#101820'},
    'SEA': {'name': 'Seahawks', 'city': 'Seattle', 'color': '#002244', 'color2': '#69BE28'},
    'SF':  {'name': '49ers', 'city': 'San Francisco', 'color': '#AA0000', 'color2': '#B3995D'},
    'TB':  {'name': 'Buccaneers', 'city': 'Tampa Bay', 'color': '#D50A0A', 'color2': '#34302B'},
    'TEN': {'name': 'Titans', 'city': 'Tennessee', 'color': '#0C2340', 'color2': '#4B92DB'},
    'WAS': {'name': 'Commanders', 'city': 'Washington', 'color': '#5A1414', 'color2': '#FFB612'},
}

# ============================================================
# GAME STATE (for Quick Predict mode — same as before)
# ============================================================
game_state = {
    'plays': [],
    'current_drive': [],
    'home_team': None,
    'away_team': None,
}


def reset_game():
    game_state['plays'] = []
    game_state['current_drive'] = []


def compute_sequencing_features():
    drive = game_state['current_drive']
    prev_play_pass = drive[-1]['is_pass'] if len(drive) >= 1 else 0.5
    prev_play_2_pass = drive[-2]['is_pass'] if len(drive) >= 2 else 0.5
    prev_play_3_pass = drive[-3]['is_pass'] if len(drive) >= 3 else 0.5
    drive_play_number = len(drive) + 1
    if len(drive) > 0:
        drive_passes = sum(p['is_pass'] for p in drive)
        drive_pass_ratio = drive_passes / len(drive)
    else:
        drive_pass_ratio = 0.5
    consecutive = 0
    if len(drive) >= 2:
        last_type = drive[-1]['is_pass']
        for p in reversed(drive[:-1]):
            if p['is_pass'] == last_type:
                consecutive += 1
            else:
                break
        consecutive += 1
    elif len(drive) == 1:
        consecutive = 1
    prev_first_down = drive[-1].get('first_down', 0) if len(drive) >= 1 else 0
    prev_yards = drive[-1].get('yards', 0) if len(drive) >= 1 else 0
    return {
        'prev_play_pass': prev_play_pass,
        'prev_play_2_pass': prev_play_2_pass,
        'prev_play_3_pass': prev_play_3_pass,
        'drive_pass_ratio': drive_pass_ratio,
        'drive_play_number': drive_play_number,
        'consecutive_same_play': consecutive,
        'prev_first_down': prev_first_down,
        'prev_yards_gained': prev_yards,
    }


def build_feature_vector(play_input):
    """Convert user input into the model's expected feature vector (Quick Predict)."""
    off_team = play_input['possession_team']
    def_team = play_input['defensive_team']
    off_stats = team_stats.get(off_team, {})
    def_stats = team_stats.get(def_team, {})

    quarter = play_input['quarter']
    minutes = play_input['minutes']
    seconds = play_input['seconds']
    game_seconds_remaining = max(0, (4 - quarter) * 900 + minutes * 60 + seconds)

    home_score = play_input['home_score']
    away_score = play_input['away_score']
    is_home = play_input['possession_team'] == game_state.get('home_team', '')
    score_differential = (home_score - away_score) if is_home else (away_score - home_score)

    temp = play_input.get('temperature', 65)
    wind = play_input.get('wind_speed', 5)
    is_indoors = play_input.get('is_indoors', 0)
    adj_temperature = 65 if is_indoors else temp
    adj_wind = 0 if is_indoors else wind

    seq = compute_sequencing_features()

    features = {
        'down': play_input['down'],
        'ydstogo': play_input['distance'],
        'yardline_100': play_input['yard_line'],
        'score_differential': score_differential,
        'game_seconds_remaining': game_seconds_remaining,
        'shotgun': 1 if play_input.get('qb_location', 'S') == 'S' else 0,
        'no_huddle': play_input.get('no_huddle', 0),
        'home_or_away': 1 if is_home else 0,
        'defenders_in_box': play_input.get('defenders_in_box', 6),
        'n_rb': play_input.get('n_rb', 1),
        'n_te': play_input.get('n_te', 1),
        'n_wr': play_input.get('n_wr', 3),
        'n_ol_extra': play_input.get('n_ol_extra', 0),
        'n_dl': play_input.get('n_dl', 4),
        'n_lb': play_input.get('n_lb', 2),
        'n_db': play_input.get('n_db', 5),
        'ftn_motion': play_input.get('motion', 0),
        'ftn_backfield': play_input.get('n_backfield', 1),
        'ftn_defense_box': play_input.get('defenders_in_box', 6),
        'score_time_interaction': score_differential * (game_seconds_remaining / 3600),
        'down_distance_interaction': play_input['down'] * play_input['distance'],
        'team_ratio': off_stats.get('team_ratio', 0.42),
        **seq,
        'coach_pass_rate': off_stats.get('coach_pass_rate_by_down', {}).get(
            str(play_input['down']), 0.58),
        'coach_aggressiveness': off_stats.get('coach_aggressiveness', 0.58),
        'adj_temperature': adj_temperature,
        'adj_wind': adj_wind,
        'is_indoors': is_indoors,
        'high_wind': 1 if adj_wind >= 15 else 0,
        'cold_weather': 1 if adj_temperature < 40 else 0,
        'opp_def_yards_per_play': def_stats.get('def_yards_per_play', 5.0),
        'opp_def_pass_yards_per_play': def_stats.get('def_pass_yards_per_play', 5.5),
        'opp_def_rush_yards_per_play': def_stats.get('def_rush_yards_per_play', 4.0),
        'opp_def_pass_rush_diff': def_stats.get('def_pass_rush_diff', 1.5),
    }

    # Set formation/qb/hash dummies
    for fname in feature_names:
        if fname.startswith('formation_') and fname not in features:
            features[fname] = 0
        if fname.startswith('qb_loc_') and fname not in features:
            features[fname] = 0
        if fname.startswith('hash_') and fname not in features:
            features[fname] = 0

    formation = play_input.get('formation', 'SHOTGUN')
    formation_key = f'formation_{formation}'
    if formation_key in features or formation_key in feature_names:
        features[formation_key] = 1

    qb_loc = play_input.get('qb_location', 'S')
    qb_key = f'qb_loc_{qb_loc}'
    if qb_key in features or qb_key in feature_names:
        features[qb_key] = 1

    hash_mark = play_input.get('hash_mark', 'M')
    hash_key = f'hash_{hash_mark}'
    if hash_key in features or hash_key in feature_names:
        features[hash_key] = 1

    feature_vector = []
    for fname in feature_names:
        val = features.get(fname, 0)
        feature_vector.append(float(val) if val is not None else 0.0)

    return np.array([feature_vector])


# ============================================================
# ROUTES — MAIN
# ============================================================
@app.route('/')
def index():
    return render_template('index.html',
                           teams=sorted(team_stats.keys()),
                           team_meta=TEAM_META,
                           n_games=len(game_index))


# ============================================================
# ROUTES — GAME REPLAY
# ============================================================
@app.route('/api/replay/games')
def replay_games():
    """Return the list of available games grouped by season/week."""
    return jsonify({
        'games': game_index,
        'team_meta': TEAM_META,
    })


@app.route('/api/replay/load/<game_id>')
def replay_load(game_id):
    """Load a specific game's play-by-play data."""
    filepath = os.path.join(REPLAY_DIR, f'{game_id}.json')
    if not os.path.exists(filepath):
        return jsonify({'error': f'Game {game_id} not found'}), 404

    with open(filepath, 'r') as f:
        game_data = json.load(f)

    return jsonify(game_data)


@app.route('/api/replay/predict', methods=['POST'])
def replay_predict():
    """Get prediction for a single play using pre-computed features."""
    data = request.json
    features_dict = data.get('features', {})

    try:
        # Build feature vector from pre-computed features
        feature_vector = []
        for fname in feature_names:
            val = features_dict.get(fname, 0.0)
            feature_vector.append(float(val) if val is not None else 0.0)

        fv = np.array([feature_vector])
        prediction = model.predict(fv)[0]
        probabilities = model.predict_proba(fv)[0]

        run_prob = float(probabilities[0])
        pass_prob = float(probabilities[1])

        return jsonify({
            'prediction': 'PASS' if prediction == 1 else 'RUN',
            'confidence': float(max(run_prob, pass_prob)),
            'run_probability': run_prob,
            'pass_probability': pass_prob,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/replay/batch_predict', methods=['POST'])
def replay_batch_predict():
    """Get predictions for all plays in a game at once (faster)."""
    data = request.json
    plays = data.get('plays', [])

    results = []
    for play in plays:
        features_dict = play.get('features', {})
        try:
            feature_vector = []
            for fname in feature_names:
                val = features_dict.get(fname, 0.0)
                feature_vector.append(float(val) if val is not None else 0.0)

            fv = np.array([feature_vector])
            prediction = model.predict(fv)[0]
            probabilities = model.predict_proba(fv)[0]

            results.append({
                'play_id': play.get('play_id'),
                'prediction': 'PASS' if prediction == 1 else 'RUN',
                'confidence': float(max(probabilities[0], probabilities[1])),
                'run_probability': float(probabilities[0]),
                'pass_probability': float(probabilities[1]),
            })
        except Exception as e:
            results.append({
                'play_id': play.get('play_id'),
                'error': str(e),
            })

    return jsonify({'predictions': results})


# ============================================================
# ROUTES — QUICK PREDICT (existing)
# ============================================================
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        feature_vector = build_feature_vector(data)
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        run_prob = float(probabilities[0])
        pass_prob = float(probabilities[1])
        return jsonify({
            'prediction': 'PASS' if prediction == 1 else 'RUN',
            'confidence': float(max(run_prob, pass_prob)),
            'run_probability': run_prob,
            'pass_probability': pass_prob,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/record_play', methods=['POST'])
def record_play():
    data = request.json
    play_record = {
        'is_pass': 1 if data.get('was_pass', False) else 0,
        'yards': data.get('yards_gained', 0),
        'first_down': 1 if data.get('first_down', False) else 0,
    }
    if data.get('new_drive', False):
        game_state['current_drive'] = []
    game_state['current_drive'].append(play_record)
    game_state['plays'].append(play_record)
    return jsonify({
        'drive_plays': len(game_state['current_drive']),
        'total_plays': len(game_state['plays']),
    })


@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json
    reset_game()
    game_state['home_team'] = data.get('home_team', '')
    game_state['away_team'] = data.get('away_team', '')
    return jsonify({'status': 'ok'})


@app.route('/api/team_meta')
def get_team_meta():
    return jsonify(TEAM_META)


# ============================================================
# ANALYZE MODE — CV Detection
# ============================================================
analyzer = None

def get_analyzer():
    """Lazy-load the analyzer (YOLO model is big, only load when needed)."""
    global analyzer
    if analyzer is None:
        try:
            from analyze import PreSnapAnalyzer
            analyzer = PreSnapAnalyzer(model_size='n')
        except Exception as e:
            print(f"WARNING: Could not load analyzer: {e}")
            return None
    return analyzer


@app.route('/api/analyze/detect', methods=['POST'])
def analyze_detect():
    """Accept an uploaded image, run YOLO detection + heuristics."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    a = get_analyzer()
    if a is None:
        return jsonify({
            'error': 'Analyzer not available. Install: pip install ultralytics opencv-python-headless'
        }), 500

    try:
        from PIL import Image
        img = Image.open(file.stream)
        result = a.analyze(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/predict', methods=['POST'])
def analyze_predict():
    """Run prediction using features from the analyze detection (same as quick predict)."""
    data = request.json
    try:
        feature_vector = build_feature_vector(data)
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        run_prob = float(probabilities[0])
        pass_prob = float(probabilities[1])
        return jsonify({
            'prediction': 'PASS' if prediction == 1 else 'RUN',
            'confidence': float(max(run_prob, pass_prob)),
            'run_probability': run_prob,
            'pass_probability': pass_prob,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  NFL Run vs Pass Predictor v2")
    print(f"  Model: {len(feature_names)} features")
    print(f"  Games: {len(game_index)} available for replay")
    print("  Open http://localhost:8080")
    print("=" * 50 + "\n")
    app.run(debug=True, port=8080)
