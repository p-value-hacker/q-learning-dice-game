import itertools
import numpy as np
import random
from typing import List, Dict, Tuple


def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Update the Q-table using the Q-learning formula.
    """
    state = frozenset(state)
    next_state = frozenset(next_state)

    # Get the current Q-value
    current_q = q_table.get(state, {}).get(action, 0)

    # Calculate the maximum Q-value for the actions in the next state
    next_max_q = max(q_table.get(next_state, {}).values(), default=0)

    # Update the Q-value for the action in the current state
    new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
    q_table.setdefault(state, {})[action] = new_q


def adjust_epsilon(epsilon, min_epsilon, decay_rate):
    """
    Adjust the exploration rate over time.
    """
    return max(min_epsilon, epsilon * decay_rate)


def train_agent(
    initial_sticks: List[int],
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    min_epsilon: float,
    decay_rate: float,
):
    q_table = create_q_table()
    rewards: List[int] = []

    for episode in range(num_episodes):
        state: List[int] = initial_sticks.copy()
        total_reward: int = 0

        # Dynamically adjust epsilon based on the episode
        adjusted_epsilon: float = adjust_epsilon_dynamic(
            epsilon, num_episodes, episode, min_epsilon, decay_rate
        )

        while state:
            dice_roll: int = np.random.randint(2, 13)
            actions = valid_actions(state, dice_roll)
            if not actions:
                break  # No valid actions, game over

            action = choose_action(state, q_table, adjusted_epsilon, dice_roll)
            if action is None:
                break  # No action chosen, break the loop

            next_state = [s for s in state if s not in action]
            reward = 1 if not next_state else 0  # Reward 1 for winning, 0 otherwise
            total_reward += reward

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
            state = next_state

        rewards.append(total_reward)

        if episode % int(num_episodes / 10) == 0:
            print(
                f"Episode {episode:,}/{num_episodes:,} - Win Rate: {sum(rewards)/(episode + 1):.4f} - Epsilon: {adjusted_epsilon:.2f}"
            )

    return q_table, rewards


def adjust_epsilon_dynamic(
    epsilon, total_episodes, current_episode, min_epsilon, decay_rate
):
    """
    Adjust epsilon dynamically based on the progress of learning, with a minimum threshold and decay rate.
    """
    progress = current_episode / total_episodes
    if progress < 0.5:
        # Higher exploration in the early phase
        return max(min_epsilon, epsilon * decay_rate)
    else:
        # Gradually increase exploitation
        return max(min_epsilon, epsilon * (1 - progress) * decay_rate)


def valid_actions(sticks, dice_roll):
    """
    Returns a list of valid actions (stick combinations) based on the current sticks and the dice roll.
    An action is a combination of sticks that sums up to the dice roll.
    """
    valid = []
    max_combination_length = min(len(sticks), dice_roll)
    for L in range(1, max_combination_length + 1):
        for subset in itertools.combinations(sticks, L):
            if sum(subset) == dice_roll:
                valid.append(subset)
    return valid


def create_q_table():
    """
    Create a Q-table as a dictionary. The keys are states (represented as frozenset of remaining sticks)
    and the values are dictionaries mapping actions to Q-values.
    """
    q_table = {}
    return q_table


def choose_action(state, q_table, epsilon, dice_roll):
    """
    Choose an action based on the ε-greedy policy.
    """
    if np.random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        actions = valid_actions(list(state), dice_roll)
        if not actions:
            return None
        return random.choice(actions)
    else:
        # Exploitation: choose the best action based on Q-table
        state_actions = q_table.get(frozenset(state), {})
        if not state_actions:
            actions = valid_actions(list(state), dice_roll)
            return random.choice(actions) if actions else None

        max_q_value = max(state_actions.values())
        best_actions = [
            action for action, q in state_actions.items() if q == max_q_value
        ]
        return random.choice(best_actions)
