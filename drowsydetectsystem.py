import cv2
import dlib
import numpy as np
import torch
import math
import time
import pygame
import json
import threading
import webbrowser
from scipy.spatial import distance as dist

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Define sounds and their delays
sounds = {
    'regarder': ('./regarder.mp3', 10),
    'reposer': ('./reposer.mp3', 15),
    'phone': ('./phone.mp3', 15),
    'welcome_eng': ('./welcomeengl.mp3', 0)
}
last_played = {key: 0 for key in sounds}

def play_sound(sound_key):
    audio_file, delay = sounds[sound_key]
    current_time = time.time()
    if current_time - last_played[sound_key] > delay:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        last_played[sound_key] = current_time  

def sound_thread(sound_key):
    thread = threading.Thread(target=play_sound, args=(sound_key,))
    thread.daemon = True
    thread.start()

# Log detected events
detection_logs = []
start_time = time.time()

print("[INFO] project realized by: RMA assurance Marocaine d'assurance")
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat')
print("[INFO] initializing camera...")
cap = cv2.VideoCapture(0)
desired_fps = 30
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Capture a snapshot at the start of the session
ret, snapshot = cap.read()
if ret:
    snapshot_path = "snapshot.jpg"
    cv2.imwrite(snapshot_path, snapshot)
else:
    snapshot_path = None  

def get_camera_matrix(size):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    return np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

model_points = np.array([
    (0.0, 0.0, 0.0),  
    (-30.0, -125.0, -30.0),  
    (30.0, -125.0, -30.0),  
    (-60.0, -70.0, -60.0),  
    (60.0, -70.0, -60.0),  
    (0.0, -330.0, -65.0)   
])

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def getHeadTiltAndCoords(size, image_points, frame_height):
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [
        0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  camera_matrix, dist_coeffs,
                                                                  flags = cv2.SOLVEPNP_ITERATIVE)
    (nose_end_point2D, _) = cv2.projectPoints(np.array(
        [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    head_tilt_degree = abs(
        [-180] - np.rad2deg([rotationMatrixToEulerAngles(rotation_matrix)[0]]))
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    ending_point_alternate = (ending_point[0], frame_height // 2)
    return head_tilt_degree, starting_point, ending_point, ending_point_alternate

def is_valid_eye(eye_points):
    # Check if eye landmarks form a reasonable shape
    width = dist.euclidean(eye_points[0], eye_points[3])
    height1 = dist.euclidean(eye_points[1], eye_points[5])
    height2 = dist.euclidean(eye_points[2], eye_points[4])
    return width > 10 and (height1/width > 0.1) and (height2/width > 0.1)

def eye_aspect_ratio(eye):
    try:
        # Vertical distances
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Horizontal distance
        C = dist.euclidean(eye[0], eye[3])
        
        # Avoid division by zero
        if C == 0:
            return 0.0
            
        ear = (A + B) / (2.0 * C)
        
        # Additional check for invalid eye shapes
        if ear > 0.5:  # Unrealistically high EAR
            return 0.0
            
        return ear
    except:
        return 0.0

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def nose_aspect_ratio(nose):
    vertical_distance = dist.euclidean(nose[0], nose[2])
    depth_distance = dist.euclidean(nose[0], nose[1])
    return depth_distance / vertical_distance

weights_path = 'C:\\Project\\SafeDriveVision\\yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Eye detection parameters
EYE_AR_THRESHOLD = 0.3  # Initial threshold, will be calibrated
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for blink detection
CALIBRATION_FRAMES = 30  # Frames to calibrate EAR threshold
DROWSY_EYES_CLOSED_DURATION = 2.0  # Seconds for drowsiness detection
eye_ar_history = []  # Stores recent EAR values
calibrated = False
blink_counter = 0
ear_values = []
eyes_closed_start_time = None

COUNTER1 = 0
COUNTER2 = 0
COUNTER3 = 0
repeat_counter = 0
sound_thread('welcome_eng')

# Initialize variables to track event durations
drowsiness_start_time = None
distraction_start_time = None
phone_usage_start_time = None
attentive_start_time = time.time()  # Start with the assumption user is attentive

while True:
    ret, img = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    current_state = "attentive"  # Assume attentive unless proven otherwise

    if len(faces) == 0:
        current_state = "distracted"
        detection_logs.append({"event": "not_looking_ahead", "timestamp": time.time()})
        sound_thread("regarder")
        cv2.putText(img, "The driver is not looking ahead!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    results = model(img)
    detections = results.xyxy[0]

    for detection in detections:
        if int(detection[5]) == 67:  # Cell phone detected
            current_state = "phone_usage"
            x1, y1, x2, y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
            detection_logs.append({"event": "using_phone", "confidence": float(conf), "timestamp": time.time()})
            COUNTER2 += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Cell Phone {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if COUNTER2 >= 3:
                cv2.putText(img, "Driver is using cell phone", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                sound_thread("phone")
                COUNTER2 = 0

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        image_points = np.array([
            landmarks_points[30],  # Nose tip
            landmarks_points[8],   # Chin
            landmarks_points[36],  # Left eye corner
            landmarks_points[45],  # Right eye corner
            landmarks_points[48],  # Left mouth corner
            landmarks_points[54]   # Right mouth corner
        ], dtype="double")

        left_eye = landmarks_points[36:42]
        right_eye = landmarks_points[42:48]
        
        # 1. Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        ear_values.append(ear)

        # 2. Calibration phase (first few frames)
        if not calibrated and len(ear_values) < CALIBRATION_FRAMES:
            cv2.putText(img, "CALIBRATING... Keep eyes open", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            calibration_progress = int(len(ear_values)/CALIBRATION_FRAMES*100)
            cv2.putText(img, f"Calibration: {calibration_progress}%", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            continue

        # 3. Set personalized threshold after calibration
        if not calibrated and len(ear_values) >= CALIBRATION_FRAMES:
            EYE_AR_THRESHOLD = np.mean(ear_values) * 0.85  # 85% of open eye EAR
            calibrated = True
            print(f"Calibration complete. Personalized EAR threshold: {EYE_AR_THRESHOLD:.2f}")

        # 4. Only check for closed eyes if we've calibrated and have valid eye contours
        if calibrated and is_valid_eye(left_eye) and is_valid_eye(right_eye):
            if ear < EYE_AR_THRESHOLD:
                blink_counter += 1
                if eyes_closed_start_time is None:  # Eyes just closed
                    eyes_closed_start_time = time.time()
                else:  # Eyes remain closed
                    closed_duration = time.time() - eyes_closed_start_time
                    if closed_duration >= DROWSY_EYES_CLOSED_DURATION:
                        current_state = "drowsy"
                        detection_logs.append({
                            "event": "eyes_closed_prolonged", 
                            "timestamp": time.time(),
                            "duration": closed_duration
                        })
                        COUNTER1 += 1
                        cv2.putText(img, f"DROWSY! Eyes closed {closed_duration:.1f}s", 
                                   (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if COUNTER1 >= 2:  # Reduced from 4 to make it more sensitive
                            sound_thread("reposer")
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    # Only register as eyes closed if we had consecutive low EAR frames
                    current_state = "drowsy"
                    detection_logs.append({
                        "event": "eyes_closed", 
                        "timestamp": time.time(),
                        "duration": (time.time() - eyes_closed_start_time) if eyes_closed_start_time else 0
                    })
                    COUNTER1 += 1
                    cv2.putText(img, "Eyes Closed!", (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if COUNTER1 >= 2:
                        sound_thread("reposer")
                        repeat_counter += 1
                        if repeat_counter >= 3:
                            detection_logs.append({
                                "event": "eyes_closed_3_times", 
                                "timestamp": time.time()
                            })
                            repeat_counter = 0
                            cv2.putText(img, "Eyes Closed 3 times!", (x, y - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                blink_counter = 0
                eyes_closed_start_time = None  # Reset when eyes open

        # Dynamic threshold adjustment
        if calibrated and len(ear_values) % 100 == 0:
            recent_open_ears = [e for e in ear_values[-100:] if e > EYE_AR_THRESHOLD*1.2]
            if len(recent_open_ears) > 20:
                new_threshold = np.mean(recent_open_ears) * 0.85
                EYE_AR_THRESHOLD = 0.9*EYE_AR_THRESHOLD + 0.1*new_threshold  # Smooth update

        mouth = landmarks_points[48:68]
        mar = mouth_aspect_ratio(mouth)

        cv2.putText(img, f'EAR: {ear:.2f} (Thresh: {EYE_AR_THRESHOLD:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img, f'MAR: {mar:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if mar > 0.7:
            current_state = "drowsy"
            detection_logs.append({"event": "yawning", "timestamp": time.time()})
            sound_thread("reposer")
            cv2.putText(img, "Yawning!", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Head tilt visualization
        size = img.shape
        frame_height = img.shape[0]
        head_tilt_degree, start_point, end_point, end_point_alt = getHeadTiltAndCoords(size, image_points, frame_height)
        cv2.line(img, start_point, end_point, (255, 0, 0), 2)
        cv2.line(img, start_point, end_point_alt, (0, 0, 255), 2)
        cv2.putText(img, f'Head Tilt: {head_tilt_degree[0]:.2f} degrees', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw landmarks
        for point in landmarks_points:
            cv2.circle(img, (point[0], point[1]), 2, (255, 255, 255), -1)

        # Draw eye contours
        left_eyeHull = cv2.convexHull(left_eye)
        right_eyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(img, [left_eyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(img, [right_eyeHull], -1, (255, 255, 255), 1)

        # Draw mouth contour
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)

    # Update attentive time tracking
    if current_state == "attentive":
        if attentive_start_time is None:  # Just became attentive
            attentive_start_time = time.time()
    else:
        if attentive_start_time is not None:  # Just stopped being attentive
            detection_logs.append({"event": "attentive_period", "duration": time.time() - attentive_start_time, "timestamp": time.time()})
            attentive_start_time = None

    cv2.imshow("Video Stream", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
cap.release()
cv2.destroyAllWindows()

# Calculate summary statistics using timestamps
drowsiness_duration = 0
distraction_duration = 0
phone_usage_duration = 0
attentive_duration = 0

drowsiness_start_time = None
distraction_start_time = None
phone_usage_start_time = None
attentive_start_time = None

for i, log in enumerate(detection_logs):
    event = log["event"]
    timestamp = log["timestamp"]

    # Handle Attentive Periods
    if event == "attentive_period":
        attentive_duration += log["duration"]
        continue

    # Handle Drowsiness Events
    if event in ["eyes_closed", "eyes_closed_3_times", "yawning", "eyes_closed_prolonged"]:
        if drowsiness_start_time is None:
            drowsiness_start_time = timestamp
    else:
        if drowsiness_start_time is not None:
            drowsiness_duration += timestamp - drowsiness_start_time
            drowsiness_start_time = None

    # Handle Distraction Events
    if event == "not_looking_ahead":
        if distraction_start_time is None:
            distraction_start_time = timestamp
    else:
        if distraction_start_time is not None:
            distraction_duration += timestamp - distraction_start_time
            distraction_start_time = None

    # Handle Phone Usage Events
    if event == "using_phone":
        if phone_usage_start_time is None:
            phone_usage_start_time = timestamp
    else:
        if phone_usage_start_time is not None:
            phone_usage_duration += timestamp - phone_usage_start_time
            phone_usage_start_time = None

# Add remaining durations for events that were ongoing at the end of the session
if drowsiness_start_time is not None:
    drowsiness_duration += end_time - drowsiness_start_time
if distraction_start_time is not None:
    distraction_duration += end_time - distraction_start_time
if phone_usage_start_time is not None:
    phone_usage_duration += end_time - phone_usage_start_time
if attentive_start_time is not None:
    attentive_duration += end_time - attentive_start_time

total_duration = end_time - start_time

# Save detection logs to a JSON file
with open("detection_logs.json", "w") as f:
    json.dump(detection_logs, f, indent=4)

# Generate HTML report
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Drowsiness Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f4f4f4;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .good {{
            color: green;
        }}
        .warning {{
            color: orange;
        }}
        .danger {{
            color: red;
        }}
    </style>
</head>
<body>
    <h1>Driver Drowsiness Detection Report</h1>
    <h2>Session Summary</h2>
    <table>
        <tr>
            <th>Start Time</th>
            <td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}</td>
        </tr>
        <tr>
            <th>End Time</th>
            <td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}</td>
        </tr>
        <tr>
            <th>Total Duration</th>
            <td>{total_duration:.2f} seconds</td>
        </tr>
        <tr>
            <th>Attentive Duration</th>
            <td class="{'good' if (attentive_duration/total_duration) >= 0.8 else 'warning' if (attentive_duration/total_duration) >= 0.5 else 'danger'}">
                {attentive_duration:.2f} seconds ({(attentive_duration/total_duration*100):.1f}%)
            </td>
        </tr>
        <tr>
            <th>Drowsiness Duration</th>
            <td class="{'danger' if (drowsiness_duration/total_duration) >= 0.2 else 'warning' if (drowsiness_duration/total_duration) >= 0.1 else 'good'}">
                {drowsiness_duration:.2f} seconds ({(drowsiness_duration/total_duration*100):.1f}%)
            </td>
        </tr>
        <tr>
            <th>Distraction Duration</th>
            <td class="{'danger' if (distraction_duration/total_duration) >= 0.2 else 'warning' if (distraction_duration/total_duration) >= 0.1 else 'good'}">
                {distraction_duration:.2f} seconds ({(distraction_duration/total_duration*100):.1f}%)
            </td>
        </tr>
        <tr>
            <th>Phone Usage Duration</th>
            <td class="{'danger' if phone_usage_duration > 0 else 'good'}">
                {phone_usage_duration:.2f} seconds ({(phone_usage_duration/total_duration*100):.1f}%)
            </td>
        </tr>
    </table>
    <h2>Snapshot</h2>
    <p>Below is a snapshot taken during the session:</p>
    {f'<img src="{snapshot_path}" alt="Snapshot">' if snapshot_path else '<p>No snapshot available</p>'}
    <h2>Event Logs</h2>
    <table>
        <tr>
            <th>Event</th>
            <th>Timestamp</th>
            <th>Details</th>
        </tr>
"""

for log in detection_logs:
    event = log.get("event", "Unknown Event")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log.get("timestamp", 0)))
    details = ""
    if "confidence" in log:
        details = f"Confidence: {log['confidence']:.2f}"
    elif "duration" in log:
        details = f"Duration: {log['duration']:.2f} seconds"
    
    # Color code events in the log
    event_class = ""
    if event in ["eyes_closed", "eyes_closed_3_times", "yawning", "eyes_closed_prolonged"]:
        event_class = "drowsy-event"
    elif event == "not_looking_ahead":
        event_class = "distracted-event"
    elif event == "using_phone":
        event_class = "phone-event"
    elif event == "attentive_period":
        event_class = "attentive-event"
    
    html_content += f"""
        <tr class="{event_class}">
            <td>{event}</td>
            <td>{timestamp}</td>
            <td>{details}</td>
        </tr>
    """

html_content += """
    </table>
    <style>
        .drowsy-event {
            background-color: #ffdddd;
        }
        .distracted-event {
            background-color: #fff3cd;
        }
        .phone-event {
            background-color: #ffcccc;
        }
        .attentive-event {
            background-color: #ddffdd;
        }
    </style>
</body>
</html>
"""

# Save HTML content to a file
with open("report.html", "w") as f:
    f.write(html_content)

# Open the HTML report in the default web browser
webbrowser.open("report.html")