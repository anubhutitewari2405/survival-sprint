import cv2
import torch
import threading
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from keras.models import load_model
from stable_baselines3 import DQN
import time
import random
from playsound import playsound
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ------------------ Load Models ------------------
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

print("Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=30)

print("Loading movement classifier CNN model...")
cnn_model = load_model("movement_classifier.h5")

print("Loading dqn model...")
dqn_model = DQN.load("red_green_dqn_model.zip")

# ------------------ Tkinter Setup ------------------
root = Tk()
root.title("🟢🔴 Red-Green Light Game")
root.geometry("1280x720")
root.configure(bg="#e6f2ff")  # Light blue background

# ------------------ Title with Animation ------------------
# ------------------ Title with Animation ------------------
font_size = 24
direction = 1  # 1 for growing, -1 for shrinking

def animate_title():
    global font_size, direction
    font_size += direction
    if font_size >= 28:
        direction = -1
    elif font_size <= 24:
        direction = 1
    title_label.config(font=("Arial Black", font_size))
    root.after(300, animate_title)

title_label = Label(root, text="🕹️ Red-Green Light Game", font=("Arial Black", font_size), bg="#e6f2ff", fg="#004d99")
title_label.grid(row=0, column=0, columnspan=3, pady=10)
animate_title()


# ------------------ Light Indicator ------------------
green_img = ImageTk.PhotoImage(Image.open("green_light.jpg").resize((100, 100)))
red_img = ImageTk.PhotoImage(Image.open("red_light.jpg").resize((100, 100)))
light_label = Label(root, image=green_img, bg="#e6f2ff")
light_label.grid(row=1, column=1, pady=5)

status_label = Label(root, text="Initializing...", font=("Arial", 16, "bold"), bg="#e6f2ff", fg="blue")
status_label.grid(row=2, column=1)

# ------------------ Canvas for Video ------------------
canvas = Canvas(root, width=640, height=480, bg="black", bd=4, relief=RIDGE)
canvas.grid(row=3, column=1, padx=10, pady=10)

# ------------------ Leaderboard ------------------
leaderboard_frame = Frame(root, bg="#e6f2ff", bd=2, relief=RIDGE)
leaderboard_frame.grid(row=3, column=0, sticky=N, padx=10)

Label(leaderboard_frame, text="🏆 Leaderboard", font=("Arial", 14, "bold"), bg="#e6f2ff").pack(pady=5)
leaderboard_text = Text(leaderboard_frame, height=25, width=30, font=("Courier", 10), bg="#fff", fg="#000")
leaderboard_text.pack(padx=5, pady=5)

# ------------------ Graph Setup ------------------
graph_frame = Frame(root, bg="#e6f2ff", bd=2, relief=RIDGE)
graph_frame.grid(row=3, column=2, sticky=N, padx=10)

fig, ax = plt.subplots(figsize=(5, 4))
canvas_graph = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_graph.get_tk_widget().pack()
x_data, y_data = [], []

# ------------------ Game Logic ------------------
light_state = "green"
player_status = {}
frame = None
video_running = True
start_time = time.time()

# ------------------ Light Toggle Function ------------------
def toggle_light():
    global light_state
    while video_running:
        light_state = "green"
        light_label.config(image=green_img)
        status_label.config(text="🟢 Green Light! Run!", fg="green")
        playsound("green_light.wav", block=False)
        time.sleep(random.randint(4, 6))

        light_state = "red"
        light_label.config(image=red_img)
        status_label.config(text="🔴 Red Light! Stop!", fg="red")
        playsound("red_light.wav", block=False)
        time.sleep(random.randint(3, 5))

# ------------------ Leaderboard ------------------
def update_leaderboard():
    leaderboard_text.delete(1.0, END)
    leaderboard_text.insert(END, "Leaderboard:\n\n")
    for pid, status in sorted(player_status.items()):
        leaderboard_text.insert(END, f"Player {pid}: {status}\n")

# ------------------ Graph Update ------------------
def update_graph():
    current_time = time.time() - start_time
    alive_count = sum(1 for status in player_status.values() if status == "alive")
    x_data.append(current_time)
    y_data.append(alive_count)
    ax.clear()
    ax.plot(x_data, y_data, color='blue')
    ax.set_title("Active Players Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Alive Players")
    ax.set_ylim(0, max(1, max(y_data, default=1)))
    canvas_graph.draw()

# ------------------ Video Loop ------------------
def video_loop():
    global frame
    cap = cv2.VideoCapture(0)

    while video_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            class_name = model.names[int(cls)]
            if class_name == "person":
                detections.append([[x1, y1, x2, y2], conf, class_name])

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = [int(v) for v in track.to_ltrb()]
            player_crop = frame[t:b, l:r]

            if track_id not in player_status:
                player_status[track_id] = "alive"

            if light_state == "red" and player_status[track_id] == "alive":
                try:
                    resized = cv2.resize(player_crop, (64, 64))
                    normalized = resized / 255.0
                    input_data = np.expand_dims(normalized, axis=0)
                    prediction = cnn_model.predict(input_data)[0][0]
                    if prediction > 0.5:
                        player_status[track_id] = "eliminated"
                except:
                    continue

            color = (0, 255, 0) if player_status[track_id] == "alive" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"Player {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        update_leaderboard()
        update_graph()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
        canvas.imgtk = imgtk

    cap.release()

# ------------------ Start Threads ------------------
video_thread = threading.Thread(target=video_loop, daemon=True)
light_thread = threading.Thread(target=toggle_light, daemon=True)
video_thread.start()
light_thread.start()

# ------------------ Mainloop ------------------
root.mainloop()
video_running = False