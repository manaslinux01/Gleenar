from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import detrend, find_peaks, butter, filtfilt
from werkzeug.utils import secure_filename
import os
import tempfile
from flask_cors import CORS
from utils import generate_frames
import threading
from deepface import DeepFace
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
CORS(app)

# Global variables to store video-related information
current_video = None
total_frames = 0
current_frame_index = 0
frame_delay = 0.1  # Delay between frames (in seconds)
stop_streaming = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video, total_frames, current_frame_index, stop_streaming
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        return 'No video file uploaded', 400
    
    video_file = request.files['video']
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_file.name)
    
    # Open the video with OpenCV
    current_video = cv2.VideoCapture(temp_file.name)
    
    # Get total number of frames
    total_frames = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_index = 0
    stop_streaming = False
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'total_frames': total_frames
    }), 200

def detect_emotion(frame):
    """Detects emotions using DeepFace."""
    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use DeepFace to analyze emotions
    try:
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        
        # Draw the emotion on the frame
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error in emotion detection:", e)
    
    return frame

def generate_frames():
    global current_video, current_frame_index, total_frames, stop_streaming
    
    while not stop_streaming:
        if current_video is None:
            time.sleep(0.1)
            continue
        
        # Set video to specific frame
        current_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        
        # Read the frame
        success, frame = current_video.read()
        
        if success:
            # Detect emotion on the frame
            frame = detect_emotion(frame)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Increment frame index (loop back to start if at end)
            current_frame_index = (current_frame_index + 1) % total_frames
            
            time.sleep(frame_delay)
        else:
            # Reset to beginning if no more frames
            current_frame_index = 0
            time.sleep(0.1)

@app.route('/video_stream')
def video_stream():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stop_streaming
    stop_streaming = True
    return jsonify({'message': 'Streaming stopped'}), 200




app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


rgb_signals = {"R": [], "G": [], "B": []}
frame_count = 0
fps = 30

def reset_rgb_signals():
    global rgb_signals
    rgb_signals["R"].clear()
    rgb_signals["G"].clear()
    rgb_signals["B"].clear()


def katz_fractal_dimension(signal):
    n = len(signal)
    L = np.sum(np.sqrt(1 + np.diff(signal)**2))
    d = np.max(np.abs(signal - signal[0]))
    return np.log10(L) / np.log10(d + 1e-8)

def detrended_fluctuation_analysis(signal):
    signal = detrend(signal)
    return np.sqrt(np.mean(signal**2))

def normalize_rgb_signals(rgb_signals):
    normalized = {}
    for channel, data in rgb_signals.items():
        if len(data) > 1:
            channel_data = np.array(data)
            mean = channel_data.mean()
            std = channel_data.std()
            normalized[channel] = (channel_data - mean) / (std + 1e-8).tolist()
        else:
            normalized[channel] = [0]
    return normalized

def low_pass_filter(signal, cutoff_freq, fps):
    nyquist = 0.5 * fps
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    if len(signal) <= 15:
        return signal
    return filtfilt(b, a, signal)

def smooth_signals(signal):
    if len(signal) < 15:
        return np.pad(signal, (0, 15 - len(signal)), 'constant', constant_values=0)
    return low_pass_filter(signal, 10, fps)

def bandpass_filter(signal, lowcut=0.8, highcut=3.0, fs=30, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_heart_rate(signal, fps):
    smoothed_signal = smooth_signals(signal)
    filtered_signal = bandpass_filter(smoothed_signal, fs=fps)
    peaks, _ = find_peaks(filtered_signal, distance=fps // 2, prominence=0.5)
    heart_rate = (len(peaks) / (len(filtered_signal) / fps)) * 60
    return max(30, min(heart_rate, 200))

def process_frame(frame):
    global frame_count, rgb_signals
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.GaussianBlur(rgb_frame, (5, 5), 0)
    
    results = mp_face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        regions = {
            "forehead": [landmarks.landmark[i] for i in range(10, 20)],
            "left_cheek": [landmarks.landmark[i] for i in range(50, 60)],
            "right_cheek": [landmarks.landmark[i] for i in range(280, 290)],
        }
        
        frame_height, frame_width = frame.shape[:2]
        best_region = None
        best_quality = float('-inf')
        
        for region_name, region_points in regions.items():
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            points = np.array([[int(point.x * frame_width), int(point.y * frame_height)] 
                             for point in region_points])
            cv2.fillConvexPoly(mask, points, 1)
            
            skin_segmented_rgb = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)
            non_black_pixels = skin_segmented_rgb[np.any(skin_segmented_rgb != [0, 0, 0], axis=-1)]
            
            if len(non_black_pixels) > 0:
                r_mean, g_mean, b_mean = np.mean(non_black_pixels, axis=0)
                kfd_value = katz_fractal_dimension(non_black_pixels[:, 0])
                dfa_value = detrended_fluctuation_analysis(non_black_pixels[:, 0])
                quality_score = dfa_value / (kfd_value + 1e-8)
                
                if quality_score > best_quality:
                    best_quality = quality_score
                    best_region = (r_mean, g_mean, b_mean)
        
        if best_region:
            r_mean, g_mean, b_mean = best_region
            rgb_signals["R"].append(float(r_mean))
            rgb_signals["G"].append(float(g_mean))
            rgb_signals["B"].append(float(b_mean))
    
    frame_count += 1
    return cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))[1].tobytes()

def generate_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    global fps
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
              
            break
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + process_frame(frame) + b'\r\n')
    
    cap.release()

def reset_processing():
    global  frame_count
    
    frame_count = 0

socketio = SocketIO(app)


def generate_signals():
    time_sec = 0
    while True:
        
        data = {
            "heart_rate": 72,  # Example heart rate
            "raw": {"R": 100 + time_sec, "G": 150 + time_sec, "B": 200 + time_sec},
            "normalized": {"R": 0.5, "G": 0.3, "B": -0.1},
            "time": time_sec
        }
        socketio.emit('signal_data', data)  
        time.sleep(1)  

@socketio.on('connect')
def handle_connect():
    print("Client connected")

    socketio.start_background_task(target=generate_signals)
    
    
    
@app.route('/signal') 
def signal():
    return render_template('signal.html')

@app.route('/upload', methods=['POST'])
def upload_videos():

    reset_rgb_signals()  
    reset_processing()
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(video.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(filepath)
    
    return jsonify({'success': True, 'filepath': filepath})

@app.route('/video_feed/<path:video_path>')
def video_feeds(video_path):
    return Response(generate_frame(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_signals')
def get_signals():
    
    normalized = normalize_rgb_signals(rgb_signals)
    heart_rate = None

    
    if len(rgb_signals["G"]) > 30:
        heart_rate = int(extract_heart_rate(rgb_signals["G"], fps))
    
    return jsonify({
        'raw': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in rgb_signals.items()},
        'normalized': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in normalized.items()},
        'heart_rate': heart_rate,
        'time': [i / fps for i in range(len(rgb_signals["R"]))]
    })


import os

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode)

