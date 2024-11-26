import cv2
from deepface import DeepFace
import numpy as np

confidence_scores = []

def detect_emotion(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        confidence_score = analysis[0]['emotion'][emotion]

        confidence_scores.append(confidence_score)
        mean_confidence = np.mean(confidence_scores)
        
        n = len(confidence_scores)
        std_error = np.std(confidence_scores, ddof=1) / np.sqrt(n) if n > 1 else 0
        z = 1.96
        confidence_interval = (mean_confidence - z * std_error, mean_confidence + z * std_error)

        print(f"Emotion: {emotion}, Confidence Score: {confidence_score}")
        return emotion, confidence_score, confidence_interval
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None, None, None

model = "res10_300x300_ssd_iter_140000.caffemodel"
file_txt = "deploy.prototxt.txt"

def detect_faces_dnn(frame):
    net = cv2.dnn.readNetFromCaffe(file_txt, model)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    
    return faces


def draw_corner_box(frame, x, y, w, h, color=(0, 0, 225), thickness=2, corner_length=20):
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
    
    # Top-right corner
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
    
    # Bottom-left corner
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
    
    # Bottom-right corner
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)


# def generate_frames(video):
#     cap = cv2.VideoCapture(video)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         faces = detect_faces_dnn(frame)

#         if len(faces) > 0:
#             emotion, confidence_score, confidence_interval = detect_emotion(frame)

#             if emotion and confidence_score > 85.0:
#                 for (x, y, w, h) in faces:
#                     # Replace the rectangle with corner box
#                     draw_corner_box(frame, x, y, w, h, color=(0, 0, 225), thickness=2, corner_length=20)
#                 cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

def generate_frames(video):
    cap = cv2.VideoCapture(video)
    frame_count = 0
    frame_skip = 2 # Process every 5th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 5th frame
        if frame_count % frame_skip == 0:
            faces = detect_faces_dnn(frame)

            if len(faces) > 0:
                emotion, confidence_score, confidence_interval = detect_emotion(frame)

                if emotion and confidence_score > 85.0:
                    for (x, y, w, h) in faces:
                        # Replace the rectangle with a corner box
                        draw_corner_box(frame, x, y, w, h, color=(0, 0, 225), thickness=2, corner_length=20)
                    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1  # Increment the frame count after each iteration

    cap.release()
