import numpy as np
def find_next_state(state, action):
    if action == 0:
        if state - W >= 0:
            state -= W
    elif action == 1:
        if state + W < n_states:
            state += W
    elif action == 2:
        if ((state - 1) // W) == (state // W):
            state -=1
    elif action == 3:
        if ((state + 1) // W) == (state // W):
            state += 1
    return state
# Define the environment
H, W = 3,4
n_states = H*W  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_states = [0, 7]  # Goal state

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.5
discount_factor = 0.9
exploration_prob = 0.5
epochs = 1000

# Q-learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start from a random state

    while current_state not in goal_states:
        # Choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Simulate the environment (move to the next state)
        # For simplicity, move to the next state
        next_state = find_next_state(current_state,action)

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        reward = 1 if next_state in goal_states else -1

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])

        current_state = next_state  # Move to the next state

# After training, the Q-table represents the learned Q-values
print("Learned Q-table:")
print(Q_table)