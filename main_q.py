import pickle
from src.q_learning_training import *
from src.q_learning_assessment import *

# Define the initial set of sticks
initial_sticks = list(range(1, 13))

# Set Q-learning parameters
num_episodes = int(1e7)  # Total number of training episodes
num_episodes_for_assessment = int(1e5)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Decay rate for exploration probability (this is adjusted dynamically)

# Train the agent
q_table, rewards = train_agent(
    initial_sticks, num_episodes, alpha, gamma, epsilon, min_epsilon, decay_rate
)

# Save the Q-table to a file
with open("results/q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Q-table saved successfully.")

# Example of Q learning strategy
print("=====================================")
print(f"Assess the performance of the Q-Learning agent over {num_episodes_for_assessment:,} simulated games")
win_rate, wins, losses = simulate_games(q_table, num_episodes_for_assessment)
print(f"Win rate: {win_rate*100:.4f}% - Wins: {wins} - Losses: {losses}")
print("=====================================")

print("=====================================")
win_rate, wins, losses = simulate_random_games(num_episodes_for_assessment)
print(f"Assess the performance of the random strategy over {num_episodes_for_assessment:,} simulated games")
print(
    f"Random strategy - Win rate: {win_rate*100:.4f}% - Wins: {wins} - Losses: {losses}"
)
print("=====================================")
