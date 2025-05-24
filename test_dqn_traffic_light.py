import tensorflow as tf
import numpy as np
import random
import time


# Define a simple traffic light environment
class TrafficLightEnv:
    def __init__(self):
        self.actions = ["GO", "STOP"]
        self.state_space = ["RED", "GREEN"]
        self.current_state = 0  # 0 = RED, 1 = GREEN
        self.done = False
        self.steps = 0
        self.max_steps = 10

    def reset(self):
        self.current_state = random.choice([0, 1])  # Random initial light
        self.steps = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # One-hot encode the state: [RED, GREEN]
        state_vector = [0, 0]
        state_vector[self.current_state] = 1
        return np.array(state_vector, dtype=np.float32)

    def step(self, action):
        reward = 0
        if self.current_state == 0 and action == 1:
            reward = 1  # Correctly stopped at red light
        elif self.current_state == 1 and action == 0:
            reward = 1  # Correctly went at green light
        else:
            reward = -1  # Incorrect action

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        else:
            self.current_state = random.choice([0, 1])  # Change light randomly

        return self.get_state(), reward, self.done

    def render(self):
        print(f"Traffic Light: {'RED' if self.current_state == 0 else 'GREEN'}")


# Load the trained DQN model
model = tf.keras.models.load_model('movement_classifier.h5')

# Initialize environment
env = TrafficLightEnv()

# Start simulation
state = env.reset()
total_reward = 0

print("===== DQN Agent Simulation Start =====\n")
for step in range(10):
    env.render()

    # Reshape state for model: (1, input_dim)
    state_input = np.reshape(state, (1, 2))

    # Predict Q-values for current state
    q_values = model.predict(state_input, verbose=0)

    # Select action with highest Q-value
    action = np.argmax(q_values[0])

    action_str = "GO" if action == 0 else "STOP"
    print(f"Step {step + 1}: Action taken: {action_str}")

    # Take the action
    next_state, reward, done = env.step(action)
    total_reward += reward

    print(f"Reward: {reward}\n")
    state = next_state

    if done:
        break
    time.sleep(1)

print("===== Simulation End =====")
print(f"Total Reward: {total_reward}")
