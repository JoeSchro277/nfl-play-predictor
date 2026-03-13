"""
NFL Pre-Snap Analyzer — Computer Vision Module (v5)
=====================================================
Uses YOLOv8 to detect people in pre-snap screenshots.
Uses OCR to read the scorebug (down, distance, quarter, time, scores).
Returns detection coordinates for the interactive labeling UI.

Requirements:
  pip install ultralytics opencv-python-headless pillow pytesseract
  Also: brew install tesseract (Mac) or apt install tesseract-ocr (Linux)
"""

import numpy as np
import re
import io
import base64
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("WARNING: pytesseract not installed. Run: pip install pytesseract")


class PreSnapAnalyzer:

    def __init__(self, model_size='n'):
        self.model = None
        if YOLO_AVAILABLE:
            model_name = f'yolov8{model_size}.pt'
            print(f"Loading YOLO model ({model_name})...")
            self.model = YOLO(model_name)
            print("YOLO model loaded.")

    def analyze(self, image_input, confidence_threshold=0.2):
        """
        Detect players and return box coordinates + original image.
        Frontend handles the interactive labeling UI.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            img = Image.open(image_input)

        img = img.convert('RGB')
        img_w, img_h = img.size

        # Crop scoreboard (bottom ~18%, top ~3%)
        crop_top = int(img_h * 0.03)
        crop_bottom = int(img_h * 0.82)
        img_cropped = img.crop((0, crop_top, img_w, crop_bottom))

        # Detect people
        detections = self._detect_people(img_cropped, confidence_threshold)

        # Adjust coords back to original image
        for d in detections:
            d['y1'] += crop_top
            d['y2'] += crop_top
            d['cy'] += crop_top

        # Filter non-field
        detections = self._filter_field_players(detections, img_w, img_h)

        # Assign IDs
        for i, d in enumerate(detections):
            d['id'] = i

        # Convert all coordinates to percentages (for responsive overlay positioning)
        for d in detections:
            d['x1_pct'] = d['x1'] / img_w * 100
            d['y1_pct'] = d['y1'] / img_h * 100
            d['x2_pct'] = d['x2'] / img_w * 100
            d['y2_pct'] = d['y2'] / img_h * 100
            d['w_pct'] = (d['x2'] - d['x1']) / img_w * 100
            d['h_pct'] = (d['y2'] - d['y1']) / img_h * 100

        # Encode original image as base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Read scorebug via OCR
        scorebug = self._read_scorebug(img, img_w, img_h)

        return {
            'image': f"data:image/jpeg;base64,{img_b64}",
            'detections': [{
                'id': d['id'],
                'x1_pct': round(d['x1_pct'], 2),
                'y1_pct': round(d['y1_pct'], 2),
                'w_pct': round(d['w_pct'], 2),
                'h_pct': round(d['h_pct'], 2),
                'confidence': round(d['confidence'], 2),
            } for d in detections],
            'n_players': len(detections),
            'scorebug': scorebug,
            'image_width': img_w,
            'image_height': img_h,
        }

    def _detect_people(self, img, conf_threshold=0.2):
        if self.model is None:
            return []

        img_np = np.array(img)
        results = self.model(img_np, verbose=False, conf=conf_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                if int(boxes.cls[i]) != 0:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append({
                    'x1': float(x1), 'y1': float(y1),
                    'x2': float(x2), 'y2': float(y2),
                    'cx': float((x1+x2)/2), 'cy': float((y1+y2)/2),
                    'width': float(x2-x1), 'height': float(y2-y1),
                    'confidence': float(boxes.conf[i]),
                })

        detections.sort(key=lambda d: d['cx'])
        return detections

    def _filter_field_players(self, detections, img_w, img_h):
        if len(detections) < 3:
            return detections

        heights = [d['height'] for d in detections]
        median_h = np.median(heights)

        filtered = []
        for d in detections:
            if d['height'] < median_h * 0.3 or d['height'] > median_h * 3.0:
                continue
            if d['cy'] < img_h * 0.02 or d['cy'] > img_h * 0.92:
                continue
            filtered.append(d)

        return filtered if len(filtered) >= 3 else detections

    # ==============================================================
    # SCOREBUG OCR
    # ==============================================================
    # NFL team abbreviations for matching
    NFL_TEAMS = {
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LA', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
        'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS',
        # Common alternates
        'GNB', 'KAN', 'LAS', 'NOR', 'SFO', 'TAM', 'JAC',
        'BILLS', 'BEARS', 'BENGALS', 'BROWNS', 'COWBOYS', 'BRONCOS',
        'LIONS', 'PACKERS', 'TEXANS', 'COLTS', 'JAGUARS', 'CHIEFS',
        'RAMS', 'CHARGERS', 'RAIDERS', 'DOLPHINS', 'VIKINGS',
        'PATRIOTS', 'SAINTS', 'GIANTS', 'JETS', 'EAGLES', 'STEELERS',
        'SEAHAWKS', '49ERS', 'BUCCANEERS', 'TITANS', 'COMMANDERS',
        'CARDINALS', 'FALCONS', 'RAVENS', 'PANTHERS',
    }

    # Map alternate names to standard abbreviations
    TEAM_NORMALIZE = {
        'GNB': 'GB', 'KAN': 'KC', 'LAS': 'LV', 'NOR': 'NO',
        'SFO': 'SF', 'TAM': 'TB', 'JAC': 'JAX', 'LAR': 'LA',
        'BILLS': 'BUF', 'BEARS': 'CHI', 'BENGALS': 'CIN', 'BROWNS': 'CLE',
        'COWBOYS': 'DAL', 'BRONCOS': 'DEN', 'LIONS': 'DET', 'PACKERS': 'GB',
        'TEXANS': 'HOU', 'COLTS': 'IND', 'JAGUARS': 'JAX', 'CHIEFS': 'KC',
        'RAMS': 'LA', 'CHARGERS': 'LAC', 'RAIDERS': 'LV', 'DOLPHINS': 'MIA',
        'VIKINGS': 'MIN', 'PATRIOTS': 'NE', 'SAINTS': 'NO', 'GIANTS': 'NYG',
        'JETS': 'NYJ', 'EAGLES': 'PHI', 'STEELERS': 'PIT', 'SEAHAWKS': 'SEA',
        '49ERS': 'SF', 'BUCCANEERS': 'TB', 'TITANS': 'TEN', 'COMMANDERS': 'WAS',
        'CARDINALS': 'ARI', 'FALCONS': 'ATL', 'RAVENS': 'BAL', 'PANTHERS': 'CAR',
    }

    def _read_scorebug(self, img, img_w, img_h):
        """
        Attempt to read the scorebug via OCR.
        Returns a dict with detected fields and a confidence indicator.
        """
        result = {
            'detected': False,
            'quarter': None,
            'time': None,
            'down': None,
            'distance': None,
            'team1': None,
            'team2': None,
            'score1': None,
            'score2': None,
            'raw_text': '',
            'warnings': [],
        }

        if not OCR_AVAILABLE:
            result['warnings'].append('OCR not available — install pytesseract and tesseract')
            return result

        try:
            # The main scorebug is typically bottom-center of the frame
            # Crop a horizontal strip from the bottom 22%, center 80% (avoid corner tickers)
            bug_top = int(img_h * 0.78)
            bug_left = int(img_w * 0.10)
            bug_right = int(img_w * 0.90)
            bug_img = img.crop((bug_left, bug_top, bug_right, img_h))

            # Preprocess for better OCR:
            # 1. Scale up 2x (small text reads better when larger)
            # 2. Convert to grayscale
            # 3. Boost contrast
            # 4. Apply adaptive threshold to make text crisp
            bug_np = np.array(bug_img)

            if CV2_AVAILABLE:
                # Scale up 2x
                bug_np = cv2.resize(bug_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                # Convert to grayscale
                gray = cv2.cvtColor(bug_np, cv2.COLOR_RGB2GRAY)

                # Run OCR on multiple preprocessed versions and combine results
                ocr_texts = []

                # Version 1: High contrast grayscale
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                contrast = clahe.apply(gray)
                text1 = pytesseract.image_to_string(contrast, config='--psm 6').strip()
                ocr_texts.append(text1)

                # Version 2: Binary threshold (white text on dark bg)
                _, thresh_light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                text2 = pytesseract.image_to_string(thresh_light, config='--psm 6').strip()
                ocr_texts.append(text2)

                # Version 3: Inverted threshold (dark text on light bg)
                _, thresh_dark = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
                text3 = pytesseract.image_to_string(thresh_dark, config='--psm 6').strip()
                ocr_texts.append(text3)

                # Version 4: Adaptive threshold
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
                text4 = pytesseract.image_to_string(adaptive, config='--psm 6').strip()
                ocr_texts.append(text4)

                # Combine all OCR text (more chances to catch different elements)
                all_text = '\n'.join(ocr_texts)
            else:
                # Fallback: just run OCR on the crop
                all_text = pytesseract.image_to_string(bug_img, config='--psm 6').strip()

            result['raw_text'] = all_text

            if len(all_text.strip()) < 3:
                result['warnings'].append('Could not detect scorebug text')
                return result

            result['detected'] = True

            # Parse quarter
            qtr_match = re.search(r'(\d)\s*(?:ST|ND|RD|TH|Q)', all_text, re.IGNORECASE)
            if qtr_match:
                result['quarter'] = int(qtr_match.group(1))

            # Parse time (MM:SS or M:SS)
            time_match = re.search(r'(\d{1,2}):(\d{2})', all_text)
            if time_match:
                result['time'] = f"{time_match.group(1)}:{time_match.group(2)}"

            # Parse down & distance
            # Patterns: "3RD & 5", "3rd and 5", "2ND & 10", "3rd&5", "1ST &10"
            # Also handle OCR noise: "2N0 & 10", "3R0 & 5", ampersand as various chars
            dd_patterns = [
                r'(\d)\s*(?:ST|ND|RD|TH)\s*[&@]\s*(\d+)',          # "2ND & 10"
                r'(\d)\s*(?:ST|ND|RD|TH)\s+(?:AND)\s+(\d+)',       # "2ND AND 10"
                r'(\d)\s*(?:ST|ND|RD|TH)\s+(\d+)',                 # "2ND 10" (missing &)
                r'(\d)\s*(?:N[DO0]|R[DO0]|S[T7]|TH)\s*[&@]\s*(\d+)',  # OCR noise variants
                r'(\d)\s*[&@]\s*(\d+)',                             # bare "2 & 10"
            ]
            dd_match = None
            for pat in dd_patterns:
                dd_match = re.search(pat, all_text, re.IGNORECASE)
                if dd_match:
                    break
            if dd_match:
                down_val = int(dd_match.group(1))
                dist_val = int(dd_match.group(2))
                if 1 <= down_val <= 4 and 1 <= dist_val <= 99:
                    result['down'] = down_val
                    result['distance'] = dist_val

            # Parse scores
            # Look for two large-ish numbers that could be scores
            # Broadcast scorebugs typically show scores as prominent numbers
            # Try pattern: number, some stuff, number (like "28 ... 31")
            score_pattern = re.findall(r'\b(\d{1,2})\b', all_text)

            # Parse team names first (needed to help locate scores)
            words = re.findall(r'[A-Z]{2,}', all_text.upper())
            found_teams = []
            for word in words:
                # Skip common OCR noise words
                if word in ('THE', 'AND', 'THIS', 'HALF', 'RUSH', 'YARDS', 'PASS',
                            'TOTAL', 'RUN', 'YDS', 'AVG', 'ATT'):
                    continue
                if word in self.NFL_TEAMS:
                    normalized = self.TEAM_NORMALIZE.get(word, word)
                    if normalized not in found_teams:
                        found_teams.append(normalized)

            if len(found_teams) >= 2:
                result['team1'] = found_teams[0]
                result['team2'] = found_teams[1]
            elif len(found_teams) == 1:
                result['team1'] = found_teams[0]
                result['warnings'].append('Only detected one team name')

            # Try to extract scores
            # First, look for explicit score-like patterns
            score_match = re.search(r'(\d{1,2})\s*[-–—]\s*(\d{1,2})', all_text)
            if not score_match:
                # Try with more space: "28   31" or "28 to 31"
                score_match = re.search(r'(\d{1,2})\s+(?:to\s+)?(\d{1,2})', all_text, re.IGNORECASE)

            if score_match:
                s1, s2 = int(score_match.group(1)), int(score_match.group(2))
                # Sanity check: scores should be 0-63 range
                if s1 <= 63 and s2 <= 63:
                    result['score1'] = s1
                    result['score2'] = s2

            if result['score1'] is None:
                # Fallback: find all numbers, filter out known non-score numbers
                nums = re.findall(r'\b(\d{1,2})\b', all_text)
                # Build set of numbers we know aren't scores
                exclude = set()
                if result['quarter']: exclude.add(str(result['quarter']))
                if result['down']: exclude.add(str(result['down']))
                if result['distance']: exclude.add(str(result['distance']))
                if result['time']:
                    for part in result['time'].split(':'):
                        exclude.add(part.lstrip('0') or '0')
                        exclude.add(part)
                # Also exclude common non-score numbers (records like 9-4, 11-2)
                # and stat numbers (rush yards, etc.)

                score_candidates = []
                for n in nums:
                    if n in exclude:
                        continue
                    val = int(n)
                    if val > 63:  # no NFL score over 63
                        continue
                    score_candidates.append(val)

                # Take the first two unique candidates
                if len(score_candidates) >= 2:
                    result['score1'] = score_candidates[0]
                    result['score2'] = score_candidates[1]

            # Build warnings for missing fields
            if not result['quarter']:
                result['warnings'].append('Quarter not detected')
            if not result['time']:
                result['warnings'].append('Game clock not detected')
            if not result['down']:
                result['warnings'].append('Down not detected')
            if not result['distance']:
                result['warnings'].append('Distance not detected')
            if not result['team1'] and not result['team2']:
                result['warnings'].append('Team names not detected — may be logos only')
            if result['score1'] is None or result['score2'] is None:
                result['warnings'].append('Scores not detected')

        except Exception as e:
            result['warnings'].append(f'OCR error: {str(e)}')

        return result
