#rgl_phase2
import threading
import time
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()

# Load sounds
sound_move = pygame.mixer.Sound("move_alert.wav")
sound_red_light = pygame.mixer.Sound("red_light.wav")
sound_green_light = pygame.mixer.Sound("green_light.wav")

# Constants
GAME_DURATION = 60  # seconds
LIGHT_INTERVAL = (5, 10)  # seconds for each light
MOVEMENT_THRESHOLD = 5000  # pixel difference threshold for movement detection


class RedGreenLightGameSingleCam:
    def __init__(self, window, camera_index=0):
        self.window = window
        self.cap = cv2.VideoCapture(camera_index)

        self.model = YOLO("yolov8n.pt")  # YOLO model
        self.tracker = DeepSort(max_age=30)

        self.green_light = True
        self.score_dict = {}  # player_id: True=active, False=eliminated
        self.previous_patches = {}  # track_id -> patch image

        self.running = False
        self.paused = False
        self.start_time = None
        self.next_switch = None

        # Tkinter GUI Setup
        self.window.title("Red Green Light Game - Single Camera")

        self.label = tk.Label(window, text=f"Camera {camera_index}")
        self.label.grid(row=0, column=0)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.grid(row=1, column=0)

        # Control Buttons
        self.btn_start = tk.Button(window, text="Start Game", command=self.start_game)
        self.btn_start.grid(row=2, column=0, sticky="w")

        self.btn_pause = tk.Button(window, text="Pause", command=self.pause_game, state="disabled")
        self.btn_pause.grid(row=2, column=0)

        self.btn_resume = tk.Button(window, text="Resume", command=self.resume_game, state="disabled")
        self.btn_resume.grid(row=2, column=0, sticky="e")

        # Status label
        self.status_label = tk.Label(window, text="Waiting to start...", font=("Arial", 16))
        self.status_label.grid(row=3, column=0)

        # Thread for game loop
        self.update_thread = None

    def start_game(self):
        if self.running:
            messagebox.showinfo("Info", "Game already running!")
            return
        self.running = True
        self.paused = False
        self.start_time = time.time()
        self.next_switch = time.time() + random.randint(*LIGHT_INTERVAL)
        self.score_dict.clear()
        self.previous_patches.clear()
        self.status_label.config(text="Green Light!")
        pygame.mixer.Sound.play(sound_green_light)
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_resume.config(state="disabled")
        self.update_thread = threading.Thread(target=self.game_loop, daemon=True)
        self.update_thread.start()

    def pause_game(self):
        if not self.running or self.paused:
            return
        self.paused = True
        self.status_label.config(text="Paused")
        self.btn_pause.config(state="disabled")
        self.btn_resume.config(state="normal")

    def resume_game(self):
        if not self.running or not self.paused:
            return
        self.paused = False
        self.status_label.config(text="Green Light!" if self.green_light else "Red Light!")
        self.next_switch = time.time() + random.randint(*LIGHT_INTERVAL)
        self.btn_pause.config(state="normal")
        self.btn_resume.config(state="disabled")

    def switch_light(self):
        self.green_light = not self.green_light
        if self.green_light:
            self.status_label.config(text="Green Light!")
            pygame.mixer.Sound.play(sound_green_light)
        else:
            self.status_label.config(text="Red Light!")
            pygame.mixer.Sound.play(sound_red_light)

    def has_moved(self, prev_patch, curr_patch):
        if prev_patch is None or curr_patch is None:
            return False
        if prev_patch.shape != curr_patch.shape:
            return True
        diff = cv2.absdiff(prev_patch, curr_patch)
        non_zero_count = np.sum(diff)
        return non_zero_count > MOVEMENT_THRESHOLD

    def game_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time >= self.next_switch:
                self.switch_light()
                self.next_switch = current_time + random.randint(*LIGHT_INTERVAL)

            elapsed = current_time - self.start_time
            if elapsed > GAME_DURATION:
                self.running = False
                total = len(self.score_dict)
                eliminated = sum(1 for k in self.score_dict if not self.score_dict[k])
                remaining = total - eliminated
                self.status_label.config(
                    text=f"Game Over! Total: {total} Eliminated: {eliminated} Remaining: {remaining}")
                messagebox.showinfo("Game Over",
                                    f"Game Over!\nTotal Players: {total}\nEliminated: {eliminated}\nRemaining: {remaining}")
                self.btn_start.config(state="normal")
                self.btn_pause.config(state="disabled")
                self.btn_resume.config(state="disabled")
                break

            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.predict(rgb_frame, classes=[0])  # Only person class

            # Format detections for tracker ([x1,y1,x2,y2], confidence, class_id)
            detections = []
            for r in results:
                for det in r.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det
                    detections.append(([x1, y1, x2, y2], conf, int(cls)))

            tracks = self.tracker.update_tracks(detections, frame=rgb_frame)

            moved_ids = set()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                l, t, r, b = map(int, track.to_tlbr())
                curr_patch = frame[t:b, l:r]
                prev_patch = self.previous_patches.get(tid)

                if not self.green_light:
                    if self.has_moved(prev_patch, curr_patch):
                        moved_ids.add(tid)
                        pygame.mixer.Sound.play(sound_move)

                self.previous_patches[tid] = curr_patch.copy()

                if tid not in self.score_dict:
                    self.score_dict[tid] = True
                if tid in moved_ids:
                    self.score_dict[tid] = False

                # Draw boxes and status on frame
                color = (0, 255, 0) if self.score_dict[tid] else (0, 0, 255)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                status_text = "ACTIVE" if self.score_dict[tid] else "ELIMINATED"
                cv2.putText(frame, f"ID {tid} {status_text}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            total = len(self.score_dict)
            eliminated = sum(1 for k in self.score_dict if not self.score_dict[k])
            remaining = total - eliminated

            # Overlay game info
            info_text = f"Total: {total} Eliminated: {eliminated} Remaining: {remaining}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            light_text = "GREEN LIGHT" if self.green_light else "RED LIGHT"
            light_color = (0, 255, 0) if self.green_light else (0, 0, 255)
            cv2.putText(frame, light_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_color, 3)

            # Convert frame for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            time.sleep(0.03)  # ~30 FPS

        self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    game = RedGreenLightGameSingleCam(root, camera_index=0)  # Adjust camera index as needed
    root.mainloop()