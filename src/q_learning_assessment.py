import random
import numpy as np
from src.q_learning_training import valid_actions


def simulate_games(q_table, num_games):
    wins = 0
    losses = 0

    for _ in range(num_games):
        state = set(range(1, 13))  # Initialize the state with all sticks
        while state:
            dice_roll = np.random.randint(2, 13)
            action = choose_best_action(state, q_table, dice_roll)
            
            if not action:  # No valid action, game over
                losses += 1
                break

            state -= set(action)  # Update state by removing selected sticks

            if not state:  # All sticks removed, win
                wins += 1
                break

    win_rate = wins / num_games
    return win_rate, wins, losses

def choose_best_action(state, q_table, dice_roll):
    """
    Choose the best action based on the Q-table for the current state and dice roll.
    """
    actions = valid_actions(list(state), dice_roll)
    if not actions:
        return None

    best_action = None
    best_q_value = float('-inf')

    state = frozenset(state)
    for action in actions:
        q_value = q_table.get(state, {}).get(action, 0)
        if q_value > best_q_value:
            best_q_value = q_value
            best_action = action

    return best_action


def simulate_random_games(num_games):
    wins = 0
    losses = 0

    for _ in range(num_games):
        state = set(range(1, 13))  # Initialize the state with all sticks
        while state:
            dice_roll = np.random.randint(2, 13)
            action = choose_random_action(state, dice_roll)
            
            if not action:  # No valid action, game over
                losses += 1
                break

            state -= set(action)  # Update state by removing selected sticks

            if not state:  # All sticks removed, win
                wins += 1
                break

    win_rate = wins / num_games
    return win_rate, wins, losses

def choose_random_action(state, dice_roll):
    """
    Randomly choose an action from the valid actions for the current state and dice roll.
    """
    actions = valid_actions(list(state), dice_roll)
    return random.choice(actions) if actions else None