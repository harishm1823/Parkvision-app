"""
REST API for ParkVision
Counts are measured from the uploaded image:
  occupied = vehicles detected by YOLO
  empty    = free slots found in the gaps between parked vehicles in the same row
  total    = occupied + empty
The terminal and the web page always show the same numbers.
"""
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
import base64
from statistics import median

# Try to import OpenCV with error handling
try:
    import cv2
    cv2_available = True
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"⚠️ OpenCV import failed: {e}")
    cv2_available = False

# Import other modules
try:
    from src.detector import ParkingDetector
    from config import get_config
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    ParkingDetector = None

app = Flask(__name__)
app.config.from_object(get_config())
CORS(app)

# ---- Detection settings (tune these) ----
MODEL_NAME = os.environ.get('YOLO_MODEL', 'yolov8m.pt')  # n < s < m < l < x (bigger = more accurate, slower)
IMG_SIZE = 1280          # larger input helps small cars in aerial photos
DETECT_CONF = 0.15       # model-level confidence threshold
COUNT_CONF = float(os.environ.get('COUNT_CONF', 0.20))  # minimum confidence to count a vehicle
TILE_SIZE = int(os.environ.get('TILE_SIZE', 640))      # tile size in pixels for tiled detection
TILE_OVERLAP = 0.25      # overlap between neighbouring tiles
TILE_IMG_SIZE = 1024     # inference size per tile (upscales small cars)

# ---- Empty-slot inference settings ----
ROW_TOL = 0.6            # vehicles whose centres are within this x vehicle height share a row
MIN_ROW_CARS = 3         # a row needs at least this many vehicles before gaps are trusted
MAX_GAP_SLOTS = 4        # ignore gaps that would need more empty slots than this (likely a driving lane)


@app.before_request
def check_api_key():
    # Let CORS preflight, the web page, static files and public routes through
    if request.method == 'OPTIONS':
        return
    if request.path.startswith('/static/') or request.path == '/favicon.ico':
        return
    exempt_routes = ['/', '/upload', '/api/health', '/counts']
    if request.path in exempt_routes:
        return

    if request.headers.get('X-API-Key') != os.environ.get('API_KEY'):
        return jsonify({'error': 'Unauthorized'}), 401


@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection from web interface"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'AI model not loaded',
                'empty': 0,
                'occupied': 0,
                'total': 0
            }), 500

        # Read and process image
        image_bytes = file.read()

        if not cv2_available:
            return jsonify({'error': 'OpenCV not available'}), 500

        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Run detection
        results = detector.detect(image)
        counts = detector.count_spaces(results)

        # Vehicle details (already filtered by COUNT_CONF inside detect)
        detections = []
        if results is not None and getattr(results, 'boxes', None):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = detector.class_names[cls] if cls < len(detector.class_names) else 'vehicle'
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                })

        empty_slots = [[int(v) for v in s] for s in getattr(results, 'empty_slots', [])]

        # One line in the terminal with exactly what the page will show
        print(f"📊 Occupied: {counts['occupied']} | Empty: {counts['empty']} | "
              f"Total: {counts['total']} | Boxes drawn: {len(detections)} vehicles + {len(empty_slots)} empty slots")

        # Draw boxes on a copy of the image: green = vehicle, blue = empty slot
        annotated = None
        try:
            vis = image.copy()
            thickness = max(2, int(round(max(vis.shape[:2]) / 600)))
            font_scale = max(0.4, thickness * 0.25)
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), thickness)
                label = f"{d['confidence']:.2f}"
                cv2.putText(vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
            for x1, y1, x2, y2 in empty_slots:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 140, 0), thickness)
                cv2.putText(vis, "empty", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 140, 0), max(1, thickness - 1), cv2.LINE_AA)
            vh, vw = vis.shape[:2]
            if max(vh, vw) > 1280:
                scale = 1280 / max(vh, vw)
                vis = cv2.resize(vis, (int(vw * scale), int(vh * scale)))
            ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                annotated = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
        except Exception as draw_error:
            print(f"⚠️ Could not draw detections: {draw_error}")

        return jsonify({
            'success': True,
            'message': f'Found {counts["occupied"]} vehicles and {counts["empty"]} empty spaces',
            'empty': counts['empty'],
            'occupied': counts['occupied'],
            'total': counts['total'],
            'detections': detections,
            'empty_slots': empty_slots,
            'annotated_image': annotated,
            'filename': file.filename
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'empty': 0,
            'occupied': 0,
            'total': 0
        }), 500


# Initialize detector with error handling
detector = None
model_loaded = False

if cv2_available:
    try:
        from ultralytics import YOLO
        import torch

        # Newer PyTorch (2.6+) needs safe globals; older versions don't have this API.
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                print("🔄 Setting up PyTorch safe globals...")
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
            except Exception as sg_error:
                print(f"⚠️ Could not set safe globals: {sg_error}")

        import numpy as np

        class _Box:
            """Minimal box object compatible with the rest of the code."""
            def __init__(self, x1, y1, x2, y2, conf, cls):
                self.xyxy = [[x1, y1, x2, y2]]
                self.conf = [conf]
                self.cls = [cls]

        class _Results:
            def __init__(self, boxes, empty_slots=None):
                self.boxes = boxes
                self.empty_slots = empty_slots or []

        class RealDetector:
            def __init__(self):
                print(f"📥 Loading {MODEL_NAME}...")
                self.model = YOLO(MODEL_NAME)
                # COCO class ids: car is 2, bus is 5, truck is 7
                self.vehicle_classes = [2, 5, 7]
                self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
                print("🚗 Vehicle detection ready")

            def _run(self, image, imgsz, offset=(0, 0)):
                """Run YOLO on one image/tile and return boxes in full-image coordinates."""
                out = []
                res = self.model(image, conf=DETECT_CONF, imgsz=imgsz,
                                 classes=self.vehicle_classes, verbose=False)
                if not res or res[0].boxes is None:
                    return out
                ox, oy = offset
                for b in res[0].boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    out.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                                float(b.conf[0]), int(b.cls[0])))
                return out

            @staticmethod
            def _merge(dets, thr=0.6):
                """Greedy NMS using intersection over the smaller box (removes duplicates and cut-off halves)."""
                dets = sorted(dets, key=lambda d: d[4], reverse=True)
                kept = []
                for d in dets:
                    a_area = max(1.0, (d[2] - d[0]) * (d[3] - d[1]))
                    dup = False
                    for k in kept:
                        ix1, iy1 = max(d[0], k[0]), max(d[1], k[1])
                        ix2, iy2 = min(d[2], k[2]), min(d[3], k[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        k_area = max(1.0, (k[2] - k[0]) * (k[3] - k[1]))
                        if inter / min(a_area, k_area) > thr:
                            dup = True
                            break
                    if not dup:
                        kept.append(d)
                return kept

            @staticmethod
            def _infer_empty_slots(dets):
                """
                Find empty parking slots from the picture itself.
                Vehicles are grouped into rows. Inside a row, the typical spacing between
                neighbouring vehicles gives the slot width, and every gap that is wide
                enough for one or more extra slots is counted as empty slots.
                Returns a list of (x1, y1, x2, y2) boxes.
                """
                empty = []
                if len(dets) < MIN_ROW_CARS:
                    return empty

                med_w = median(d[2] - d[0] for d in dets)
                med_h = median(d[3] - d[1] for d in dets)

                # Group vehicles into rows by vertical centre
                rows = []
                for d in sorted(dets, key=lambda d: (d[1] + d[3]) / 2):
                    cy = (d[1] + d[3]) / 2
                    for r in rows:
                        if abs(cy - r['cy']) < ROW_TOL * med_h:
                            r['cars'].append(d)
                            r['cy'] = sum((c[1] + c[3]) / 2 for c in r['cars']) / len(r['cars'])
                            break
                    else:
                        rows.append({'cy': cy, 'cars': [d]})

                for r in rows:
                    cars = sorted(r['cars'], key=lambda d: (d[0] + d[2]) / 2)
                    if len(cars) < MIN_ROW_CARS:
                        continue  # too few vehicles to know where the slots are

                    centers = [(d[0] + d[2]) / 2 for d in cars]
                    gaps = [b - a for a, b in zip(centers, centers[1:])]

                    # Slot pitch = typical distance between neighbouring parked vehicles
                    near = [g for g in gaps if g < 1.6 * med_w]
                    pitch = median(near) if near else med_w * 1.15
                    pitch = max(pitch, med_w * 0.9)

                    row_y1 = median(d[1] for d in cars)
                    row_y2 = median(d[3] for d in cars)

                    for i, g in enumerate(gaps):
                        n = int(round(g / pitch)) - 1
                        if n < 1 or n > MAX_GAP_SLOTS:
                            continue
                        for k in range(1, n + 1):
                            cx = centers[i] + g * k / (n + 1)
                            half = pitch * 0.4
                            empty.append((cx - half, row_y1, cx + half, row_y2))
                return empty

            def detect(self, image):
                try:
                    h, w = image.shape[:2]
                    dets = self._run(image, IMG_SIZE)  # full-image pass

                    # Tiled passes for large images (finds small cars)
                    if max(h, w) > TILE_SIZE:
                        stride = max(1, int(TILE_SIZE * (1 - TILE_OVERLAP)))
                        ys = list(range(0, max(1, h - TILE_SIZE) + 1, stride))
                        xs = list(range(0, max(1, w - TILE_SIZE) + 1, stride))
                        if ys[-1] + TILE_SIZE < h:
                            ys.append(h - TILE_SIZE if h > TILE_SIZE else 0)
                        if xs[-1] + TILE_SIZE < w:
                            xs.append(w - TILE_SIZE if w > TILE_SIZE else 0)
                        for y in ys:
                            for x in xs:
                                tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
                                dets += self._run(tile, TILE_IMG_SIZE, offset=(x, y))

                    # Merge duplicates, then keep only the vehicles that will be counted AND drawn
                    merged = [d for d in self._merge(dets) if d[4] > COUNT_CONF]
                    empty_slots = self._infer_empty_slots(merged)
                    return _Results([_Box(*d) for d in merged], empty_slots)
                except Exception as e:
                    print(f"Detection error: {e}")
                    return None

            def count_spaces(self, results):
                """occupied = vehicles found, empty = free slots found, total = both."""
                if not results or not getattr(results, 'boxes', None):
                    return {'empty': 0, 'occupied': 0, 'total': 0}

                occupied = sum(1 for b in results.boxes if int(b.cls[0]) in self.vehicle_classes)
                empty = len(getattr(results, 'empty_slots', []))
                return {'empty': empty, 'occupied': occupied, 'total': occupied + empty}

        detector = RealDetector()
        model_loaded = True
        print("✅ Real YOLO detector loaded successfully")

    except Exception as e:
        # No fake/random numbers: if the model fails, the app reports an error instead.
        print(f"❌ Failed to load YOLO model: {e}")
        detector = None
        model_loaded = False
else:
    print("❌ OpenCV not available")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    detector_type = f"{MODEL_NAME} Real Detector" if model_loaded else "No Detector"
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_error',
        'version': '1.0.0',
        'model': detector_type,
        'model_loaded': model_loaded,
        'opencv_available': cv2_available,
        'detection_ready': model_loaded and cv2_available
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Detect parking spaces in uploaded image

    Request:
        - image: base64 encoded image or multipart file

    Response:
        - empty: number of empty spaces
        - occupied: number of occupied spaces
        - total: total spaces detected
        - detections: list of detection details
    """
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get image from request
        if 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            image_bytes = base64.b64decode(image_data)
        else:
            return jsonify({'error': 'No image provided'}), 400

        if not cv2_available:
            return jsonify({'error': 'OpenCV not available'}), 500

        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Run detection
        results = detector.detect(image)
        counts = detector.count_spaces(results)

        detections = []
        if results is not None and getattr(results, 'boxes', None):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': detector.class_names[cls] if cls < len(detector.class_names) else 'object',
                    'class_id': cls
                })

        return jsonify({
            'empty': counts['empty'],
            'occupied': counts['occupied'],
            'total': counts['total'],
            'detections': detections,
            'empty_slots': [[int(v) for v in s] for s in getattr(results, 'empty_slots', [])]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current parking statistics"""
    return jsonify({
        'empty': 0,
        'occupied': 0,
        'total': 0,
        'availability': 0.0,
        'timestamp': None
    })


@app.route('/counts', methods=['GET'])
def get_counts():
    """Get current counts for dashboard"""
    return jsonify({
        'empty': 0,
        'occupied': 0,
        'total': 0
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get historical parking data"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', 100, type=int)

    return jsonify({
        'data': [],
        'count': 0
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)