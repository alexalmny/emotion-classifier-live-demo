import cv2
import dlib
import numpy as np
import threading
import time
import joblib

predictor_path = "shape_predictor_68_face_landmarks.dat"
emotion_model_path = "emotion_mlp_model.joblib"

print("[INFO] Chargement du détecteur dlib...")
detector = dlib.get_frontal_face_detector()
print("[INFO] Chargement du prédicteur de landmarks...")
predictor = dlib.shape_predictor(predictor_path)
print("[INFO] Chargement du modèle d'émotions...")
emotion_model = joblib.load(emotion_model_path)
print("[INFO] Modèles chargés.")

emotion_labels = {0: "neutre", 1: "happy", 2: "fear", 3: "surprise", 4: "anger", 5: "disgust", 6: "sadness"}

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Impossible d'ouvrir la webcam!")
        exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("[INFO] Webcam ouverte.")

lock = threading.Lock()
cached_dets = []
cached_shapes = []
cached_emotions = []
detection_running = False
stop_thread = False

def extract_landmarks_flat(shape):
    coords = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords.flatten()

def detection_worker():
    global cached_dets, cached_shapes, cached_emotions, detection_running
    while not stop_thread:
        if detection_running:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = np.ascontiguousarray(gray, dtype=np.uint8)
                dets = detector(gray, 0)
                shapes = [predictor(gray, d) for d in dets]
                
                emotions = []
                for shape in shapes:
                    landmarks_flat = extract_landmarks_flat(shape)
                    emotion_pred = emotion_model.predict([landmarks_flat])[0]
                    emotions.append(emotion_pred)
                
                with lock:
                    cached_dets = list(dets)
                    cached_shapes = shapes
                    cached_emotions = emotions
                
                print(f"[INFO] Thread: {len(dets)} visage(s), émotions: {emotions}")
            detection_running = False
        time.sleep(0.01)

thread = threading.Thread(target=detection_worker, daemon=True)
thread.start()

last_trigger = 0
detection_interval = 0.3

print("[INFO] Boucle vidéo. 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    if current_time - last_trigger >= detection_interval and not detection_running:
        detection_running = True
        last_trigger = current_time
    
    with lock:
        dets_copy = cached_dets[:]
        shapes_copy = cached_shapes[:]
        emotions_copy = cached_emotions[:]
    
    for i, d in enumerate(dets_copy):
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        
        if i < len(emotions_copy):
            emotion_id = emotions_copy[i]
            emotion_text = emotion_labels.get(emotion_id, "Inconnu")
            cv2.putText(frame, emotion_text, (d.left(), d.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if i < len(shapes_copy):
            for j in range(68):
                cv2.circle(frame, (shapes_copy[i].part(j).x, shapes_copy[i].part(j).y), 2, (0, 0, 255), -1)
    
    cv2.putText(frame, f"Visages: {len(dets_copy)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam - Detection Faciale', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_thread = True
thread.join()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Terminé.")