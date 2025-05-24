import cv2

def find_cameras(max_id=10):
    available_cameras = []
    for cam_id in range(max_id):
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            print(f"Camera {cam_id} is available")
            available_cameras.append(cam_id)
            cap.release()
        else:
            print(f"Camera {cam_id} not available")
    return available_cameras

cams = find_cameras()
print("Available cameras:", cams)
