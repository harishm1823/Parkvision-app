"""
Flask web application for ParkVision parking detection
"""
from flask import Flask, render_template, Response, jsonify
import cv2
import json
from detector import ParkingDetector
from pathlib import Path

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Initialize detector
detector = ParkingDetector('models/best.pt')

# Global variables for video stream
camera = None
latest_counts = {'empty': 0, 'occupied': 0, 'total': 0}


def get_camera():
    """Get or initialize camera"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def generate_frames():
    """Generate video frames with detection"""
    global latest_counts
    
    while True:
        cam = get_camera()
        success, frame = cam.read()
        
        if not success:
            break
        
        # Process frame
        annotated, counts = detector.process_frame(frame)
        latest_counts = counts
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/counts')
def get_counts():
    """API endpoint for parking counts"""
    return jsonify(latest_counts)


@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload and process video file"""
    # TODO: Implement video upload functionality
    return jsonify({'status': 'not implemented'})


if __name__ == '__main__':
    print("Starting ParkVision Web Application...")
    print("Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
