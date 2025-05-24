import cv2

for cam_id in [0, 1]:
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Camera {cam_id} not opened")
        continue
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f'Camera {cam_id}', frame)
    else:
        print(f"No frame from camera {cam_id}")
    cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
