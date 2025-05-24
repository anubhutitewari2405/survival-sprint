#rgl_phase3s1
import cv2
import os
import time
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise IOError("Failed to load face cascade classifier")

# Create output directories
output_dir = "detected_faces"
movement_dir = os.path.join(output_dir, "movement_detected")
no_movement_dir = os.path.join(output_dir, "no_movement")

os.makedirs(movement_dir, exist_ok=True)
os.makedirs(no_movement_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Store unique IDs
tracked_ids = set()
prev_positions = {}
movement_threshold = 20  # Pixel threshold for movement detection

# CSV log file setup
csv_log_file = "face_detections_log.csv"
if not os.path.exists(csv_log_file):
    with open(csv_log_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Track_ID", "Face_Image_Path", "Movement_Status"])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLOv8 detection
        try:
            results = model(frame, verbose=False)[0]
        except Exception as e:
            print(f"Model inference error: {e}")
            continue

        # Prepare detections for DeepSORT
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # Update DeepSORT tracker
        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Tracker error: {e}")
            continue

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            if ltrb is None or len(ltrb) != 4:
                continue

            l, t, r, b = map(int, ltrb)
            tracked_ids.add(track_id)

            # Calculate movement
            cx, cy = (l + r) // 2, (t + b) // 2
            movement_status = "no_movement"
            if track_id in prev_positions:
                px, py = prev_positions[track_id]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist > movement_threshold:
                    movement_status = "movement_detected"
            prev_positions[track_id] = (cx, cy)

            # Draw bounding box and ID
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Face detection in cropped region
            person_crop = frame[t:b, l:r]
            if person_crop.size == 0:
                continue

            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            try:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            except Exception as e:
                print(f"Face detection error: {e}")
                continue

            for (fx, fy, fw, fh) in faces:
                face_img = person_crop[fy:fy + fh, fx:fx + fw]
                if face_img.size > 0:
                    timestamp = int(time.time())
                    save_dir = movement_dir if movement_status == "movement_detected" else no_movement_dir
                    filename = os.path.join(save_dir, f"face_id{track_id}_{timestamp}.jpg")

                    try:
                        cv2.imwrite(filename, face_img)
                        print(f"Saved face image: {filename}")

                        # Log to CSV
                        with open(csv_log_file, mode='a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([timestamp, track_id, filename, movement_status])
                    except Exception as e:
                        print(f"Failed to save face image: {e}")

        # Display info
        cv2.putText(frame, f"Total Persons: {len(tracked_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Save tracked IDs
    if tracked_ids:
        with open("tracked_ids.txt", "w") as f:
            for tid in sorted(tracked_ids):
                f.write(f"{tid}\n")
        print(f"Tracked IDs saved to tracked_ids.txt")

    print("Program ended")
