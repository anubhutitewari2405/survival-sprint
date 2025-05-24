#rgl_phase 1
import cv2
import threading
import tkinter as tk
from tkinter import messagebox
import random
import time
import numpy as np
import pygame
from PIL import Image, ImageTk
import datetime

# Initialize sound
pygame.mixer.init()
RED_SOUND = "red_light.wav"
GREEN_SOUND = "green_light.wav"

# Constants
GAME_DURATION = 30  # Total game duration in seconds
LIGHT_INTERVAL = (3, 7)  # Seconds between red/green switches
PRE_GAME_DELAY = 7  # Seconds to wait before starting after camera opens

class RedGreenGame:
    def __init__(self, root):
        self.root = root
        self.root.title("🔴🟢 Red-Green Light Game")
        self.root.geometry("800x700")

        # UI Elements
        self.start_button = tk.Button(root, text="Start Game", font=("Arial", 18), command=self.start_game)
        self.start_button.pack(pady=10)

        self.pause_button = tk.Button(root, text="Pause", font=("Arial", 14), command=self.toggle_pause, state='disabled')
        self.pause_button.pack(pady=5)

        self.status_label = tk.Label(root, text="Press 'Start Game' to Begin", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.score_label = tk.Label(root, text="Score: 0", font=("Arial", 14))
        self.score_label.pack()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.green_light = True
        self.score = 0
        self.running = False
        self.game_started = False
        self.paused = False
        self.background = None
        self.cap = None
        self.video_loop_id = None
        self.game_thread = None

    def play_sound(self, sound_file):
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except:
            print(f"Failed to play sound: {sound_file}")

    def start_game(self):
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.status_label.config(text="🎥 Opening camera...")
        self.score = 0
        self.score_label.config(text=f"Score: {self.score}")
        self.background = None

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access the camera")
            return

        self.running = True
        self.paused = False
        self.video_loop()
        self.game_thread = threading.Thread(target=self.game_loop)
        self.game_thread.start()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def switch_light(self):
        self.green_light = not self.green_light
        self.root.after(0, lambda: self.status_label.config(
            text="🟢 Green Light!" if self.green_light else "🔴 Red Light!"))
        self.play_sound(GREEN_SOUND if self.green_light else RED_SOUND)

    def game_loop(self):
        time.sleep(PRE_GAME_DELAY)
        self.root.after(0, lambda: self.status_label.config(text="🟢 Game Starting Now!"))
        self.game_started = True
        self.switch_light()
        start_time = time.time()
        next_switch = start_time + random.randint(*LIGHT_INTERVAL)

        while time.time() - start_time < GAME_DURATION:
            if not self.running:
                break
            if self.paused:
                time.sleep(0.1)
                continue
            if time.time() >= next_switch:
                self.switch_light()
                next_switch = time.time() + random.randint(*LIGHT_INTERVAL)
            time.sleep(1)

        self.running = False
        self.game_started = False
        self.root.after(0, lambda: self.status_label.config(text="🏁 Game Over!"))
        self.root.after(0, lambda: messagebox.showinfo("Game Over", f"Final Score: {self.score}"))
        self.save_score()
        if self.cap:
            self.cap.release()

    def save_score(self):
        with open("scores_log.txt", "a") as file:
            file.write(f"{datetime.datetime.now()} - Score: {self.score}\n")

    def video_loop(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.background is None:
                self.background = gray

            frame_delta = cv2.absdiff(self.background, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            movement_detected = False

            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                movement_detected = True

            if self.game_started and not self.green_light and movement_detected:
                self.score -= 1
                self.root.after(0, lambda: self.status_label.config(text="❌ Moved during Red Light! -1"))
                self.root.after(0, lambda: self.score_label.config(text=f"Score: {self.score}"))

            # Draw green/red light status on top-left
            color = (0, 255, 0) if self.green_light else (0, 0, 255)
            label = "Green Light" if self.green_light else "Red Light"
            cv2.rectangle(frame, (10, 10), (200, 60), color, -1)
            cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))

            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.video_loop_id = self.root.after(10, self.video_loop)

if __name__ == '__main__':
    root = tk.Tk()
    app = RedGreenGame(root)
    root.mainloop()