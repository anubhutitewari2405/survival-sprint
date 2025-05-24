#test dqn image input
import tensorflow as tf
import numpy as np
import random
import cv2
import time

# Load your trained CNN model
model = tf.keras.models.load_model('movement_classifier.h5')

# Load and preprocess image
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Red and green light image file paths
red_light_img = 'red_light.jpg'
green_light_img = 'green_light.jpg'

# Simulated environment using image input
class TrafficLightImageEnv:
    def __init__(self):
        self.actions = ["GO", "STOP"]
        self.states = {
            0: red_light_img,
            1: green_light_img
        }
        self.current_state = 0
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def reset(self):
        self.current_state = random.choice([0, 1])
        self.steps = 0
        self.done = False
        return load_image(self.states[self.current_state])

    def step(self, action):
        reward = 0
        if self.current_state == 0 and action == 1:
            reward = 1  # STOP on red
        elif self.current_state == 1 and action == 0:
            reward = 1  # GO on green
        else:
            reward = -1

        self.steps += 1
        self.current_state = random.choice([0, 1])

        if self.steps >= self.max_steps:
            self.done = True

        return load_image(self.states[self.current_state]), reward, self.done

    def render(self):
        print(f"Traffic Light: {'RED' if self.current_state == 0 else 'GREEN'}")

# Create environment
env = TrafficLightImageEnv()

# Start simulation
state = env.reset()
total_reward = 0

print("===== DQN Agent Simulation Start =====\n")
for step in range(10):
    env.render()

    # Expand dims to match input shape (1, 64, 64, 3)
    state_input = np.expand_dims(state, axis=0)

    # Predict Q-values
    q_values = model.predict(state_input, verbose=0)
    action = np.argmax(q_values[0])
    action_str = "GO" if action == 0 else "STOP"
    print(f"Step {step+1}: Action = {action_str}")

    # Take step
    state, reward, done = env.step(action)
    total_reward += reward
    print(f"Reward: {reward}\n")

    if done:
        break
    time.sleep(1)

print("===== Simulation End =====")
print(f"Total Reward: {total_reward}")
