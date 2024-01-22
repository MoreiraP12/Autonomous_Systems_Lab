# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import dill
from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from contest.util import nearestPoint, manhattanDistance
import csv
import uuid


from collections import namedtuple, deque

def create_team(first_index, second_index, is_red,
                first='DDQLAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


# Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



#################
# Team creation #
#################

TRAINING = True
BATCH_SIZE = 64
ACTION_MAPPING = {
    'North': 0,
    'East': 1,
    'South': 2,
    'West': 3,
    'Stop': 4
}

class DDQLAgent(CaptureAgent):

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            state = dill.load(f)

        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.target_net.load_state_dict(state['target_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

        # Load the ReplayBuffer state
        self.replay_buffer = state['replay_buffer']

    def save_model(self, file_path):
        state = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': self.replay_buffer
        }
        with open(file_path, 'wb') as f:
            dill.dump(state, f)

    def register_initial_state(self, game_state):
        (i_range, j_range) = game_state.get_walls().as_list()[-1]
        self.state_size = (i_range) * (j_range+1) # Define the size of the state representation
        CaptureAgent.register_initial_state(self, game_state)
        self.action_size = 5  # Define the number of actions
        self.replay_buffer = ReplayBuffer(10000)
        self.update_every = 10
        self.has_eaten_food = False
        self.avoiding_ghosts = False
        self.initial_position = game_state.get_agent_position(self.index)

        self.policy_net = QNetwork(self.state_size, self.action_size)
        self.target_net = QNetwork(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.step_count = 0
        self.cumulative_reward = 0
        self.training_id = str(uuid.uuid4())

        # Path to the saved model file
        model_save_path = "ddql_agent_state.pth"

        # Check if previously saved weights exist
        if os.path.exists(model_save_path):
            print("Loading previously saved model weights.")
            self.load_model(model_save_path)
        else:
            print("Initializing new model weights.")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def select_action(self, game_state, state, epsilon):
        """
        Select an action using the epsilon-greedy strategy.

        Args:
            state: The current state of the game.
            epsilon: The probability of choosing a random action (exploration rate).

        Returns:
            The selected action.
        """
        # Convert the state to a PyTorch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Choose action randomly if a randomly drawn number is less than epsilon
        if random.random() < epsilon:
            return random.choice(game_state.get_legal_actions(self.index))
        else:
            self.policy_net.eval()  # Set the network to evaluation mode
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            self.policy_net.train()  # Set the network back to training mode
            print("q_values: ", q_values)
            # Sort the Q-values in descending order and get their indices
            sorted_indices = torch.argsort(q_values, descending=True).squeeze().tolist()
            print("sorted: ", sorted_indices)
            # Iterate through the sorted indices to find a valid action
            for index in sorted_indices:
                if index < len(game_state.get_legal_actions(self.index)):
                    return game_state.get_legal_actions(self.index)[index]

            # If no valid action is found, return a random action as a fallback
            return random.choice(game_state.get_legal_actions(self.index))

    def update_model(self):
        # Implement the logic to update the model
        # This should involve sampling from the replay buffer and optimizing the model
        gamma = 0.95
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Convert batch arrays into PyTorch tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float)
        print(batch.action)

        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float)
        done_batch = torch.tensor(batch.done, dtype=torch.float)

        # Compute Q values for current states
        Q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V values for next states using target network
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_Q_values = reward_batch + (gamma * next_state_values * (1 - done_batch))

        # Compute loss and optimize the model
        loss = F.mse_loss(Q_values, expected_Q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
    
    # Function to convert action name to number
    def action_to_number(self, action):
        return ACTION_MAPPING.get(action, -1)  # Returns -1 for unknown actions

    # Function to convert number back to action name
    def number_to_action(self, number):
        for action, num in ACTION_MAPPING.items():
            if num == number:
                return action
        return None  # Return None or raise an error for invalid numbers

    def choose_action(self, game_state):
        state = self.get_state_representation(game_state)
        epsilon = self.get_epsilon()  # Assuming an epsilon-greedy strategy is used
        action = self.select_action(game_state, state, epsilon)
        action_number = self.action_to_number(action)

        # Update model in training mode
        if TRAINING:
            self.step_count += 1
            if self.step_count % self.update_every == 0:
                self.update_model()
                
        new_game_state = self.get_successor(game_state, action)

        next_state = self.get_state_representation(new_game_state)  # Assuming new_game_state is obtained after action
        reward = self.evaluate_action(game_state, new_game_state, action)  # Implement this method
        done = game_state.is_over()

        # Store the experience in the replay buffer
        self.replay_buffer.push(state, action_number, reward, next_state, done)
        if TRAINING:
            self.cumulative_reward += reward

        return action

    def evaluate_action(self, game_state, nextState, action):
        """
        Evaluate and return the reward for the given action.
        """

        # Get the agent's position in the current game state
        agent_position = game_state.get_agent_position(self.index)

        # Calculate rewards for different aspects and sum them up
        go_home_reward = self.calculate_carrying_food_go_home_reward(game_state, nextState)
        score_reward = self.calculate_score_reward(game_state, nextState)
        dist_to_food_reward = self.calculate_dist_to_food_reward(game_state, nextState, agent_position)
        enemies_reward = self.calculate_enemies_reward(game_state, nextState, agent_position)

        # Display individual rewards for debugging purposes
        rewards = {"enemies": enemies_reward, "go_home": go_home_reward, "dist_to_food_reward": dist_to_food_reward,
                   "score": score_reward}

        print("rewards:", rewards)
        # Return the sum of all rewards
        return sum(rewards.values())
    
    def calculate_carrying_food_go_home_reward(self, game_state, nextState):
        """
        Calculate a reward for moving closer to home when carrying food.

        Args:
            game_state (GameState): The current game state.
            nextState (GameState): The successor state after taking an action.

        Returns:
            float: The calculated reward.
        """
        original_agent_state = nextState.get_agent_state(self.index)
        amount_of_food_carrying = original_agent_state.num_carrying

        # Get the agent's position in the current and next state
        current_agent_position = game_state.get_agent_position(self.index)
        next_agent_position = nextState.get_agent_position(self.index)

        # Calculate the distance to home from the current and next state
        current_dist_to_home = self.get_maze_distance(self.initial_position, current_agent_position)
        next_dist_to_home = self.get_maze_distance(self.initial_position, next_agent_position)

        # Check if the agent has moved closer to home
        if next_dist_to_home < current_dist_to_home:
            # The reward is proportional to the amount of food being carried
            return amount_of_food_carrying * (current_dist_to_home - next_dist_to_home)
        else:
            # No reward or penalty if the agent didn't move closer to home
            return 0

    def calculate_score_reward(self, game_state, nextState):
        """
        Calculate the reward based on the change in score from the current state to the successor state.

        Args:
            game_state (GameState): The current game state.
            nextState (GameState): The successor state after taking an action.

        Returns:
            float: The calculated reward for the change in score.
        """
        score_reward = 0

        # Check if the score has increased
        if self.get_score(nextState) > self.get_score(game_state):
            # Calculate the difference in score
            diff = self.get_score(nextState) - self.get_score(game_state)

            # Update the score reward based on the team color
            score_reward += diff * 20 if self.red else -diff * 20

        return score_reward

    def calculate_dist_to_food_reward(self, game_state, nextState, agent_position):
        """
        Calculate the reward based on the change in distance to the nearest food.

        Args:
            game_state (GameState): The current game state.
            nextState (GameState): The successor state after taking an action.
            agent_position (tuple): The current position of the agent.

        Returns:
            float: The calculated reward for the change in distance to food.
        """
        # Get the list of coordinates of the agent's food in the current state
        current_foods = self.get_food(game_state).as_list()
        # Get the list of coordinates of the agent's food in the next state
        next_foods = self.get_food(nextState).as_list()

        # Get the minimum distance to food in the current state
        current_dist_to_food = min([self.get_maze_distance(agent_position, food) for food in current_foods])
        # Get the new agent position in the next state
        next_agent_position = nextState.get_agent_position(self.index)
        # Get the minimum distance to food in the next state
        next_dist_to_food = min([self.get_maze_distance(next_agent_position, food) for food in next_foods])

        # Calculate the change in distance to the nearest food
        dist_change = current_dist_to_food - next_dist_to_food
        
        # Assign a reward based on being closer to the food
        if dist_change > 0:
            dist_to_food_reward = 10  # positive reward for getting closer
        elif dist_change == 0:
            dist_to_food_reward = 0   # no reward if the distance is the same
        else:
            dist_to_food_reward = -5  # negative reward for getting farther away

        return dist_to_food_reward


    def calculate_enemies_reward(self, game_state, nextState, agent_position):
        """
        Calculate the reward based on the proximity to enemies (ghosts) in the current and next states.

        Args:
            game_state (GameState): The current game state.
            nextState (GameState): The successor state after taking an action.
            agent_position (tuple): The current position of the agent.

        Returns:
            float: The calculated reward for the proximity to enemies.
        """
        enemies_reward = 0

        # Get the states of enemies in the current state
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        # Get the positions of ghosts among enemies
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Check if there are ghosts in the current state
        if len(ghosts) > 0:
            # Get the minimum distance to a ghost in the current state
            min_dist_ghost = min([self.get_maze_distance(agent_position, g.get_position()) for g in ghosts])

            # Check if the agent is one step away from a ghost in the next state and going home
            if min_dist_ghost == 1:
                next_pos = nextState.get_agent_state(self.index).get_position()
                if next_pos == self.initial_position:
                    # Update the enemies_reward
                    enemies_reward = -50

        return enemies_reward
        

    def is_food_eaten(self, game_state, action):
        """
        Check if the agent eats food in the next state.
        """
        current_food_count = len(game_state.get_blue_food().as_list())
        new_state = self.get_successor(game_state, action)
        new_food_count = len(new_state.get_blue_food().as_list())

        return new_food_count < current_food_count
    
    def is_ghost_nearby(self, game_state, safety_distance=2):
        """
        Check if a ghost is nearby.
        """
        my_pos = game_state.get_agent_position(self.index)
        ghost_positions = [game_state.get_agent_position(self.index) for i in range(game_state.get_num_agents()) if i % 2 != 0]

        return any(manhattanDistance(my_pos, ghost_pos) <= safety_distance for ghost_pos in ghost_positions)

    def is_risky_move(self, game_state, action):
        """
        Determine if the action is risky (i.e., moves closer to a ghost).
        """
        my_pos = game_state.get_agent_position(self.index)
        new_state = self.get_successor(game_state, action)
        new_pos = new_state.get_agent_position(self.index)
        ghost_positions = [game_state.get_agent_position(self.index) for i in range(game_state.get_num_agents()) if i % 2 != 0]

        return any(manhattanDistance(new_pos, ghost_pos) <  manhattanDistance(my_pos, ghost_pos) for ghost_pos in ghost_positions)

    def is_returning_home(self, game_state, action):
        """
        Determine if the agent is returning home.
        """
        # Assuming the left side of the map is 'home'
        home_boundary_x = game_state.get_walls().width // 2
        new_state = self.get_successor(game_state, action)
        new_pos = new_state.get_agent_position(self.index)

        return new_pos[0] < home_boundary_x

    def vector_state_representation(self, game_state, agent_pos, ghosts_pos, food_grid, power_pellets, wall_position, edible_ghosts, score):
        # Agent Position
        (i_range, j_range) = game_state.get_walls().as_list()[-1]
        final_map = np.zeros((j_range, i_range))

        for i in range(0, i_range):
            for j in range(0, j_range):
                if (i, j) in [agent_pos]:
                    final_map[j][i] = 1
                if (i, j) in ghosts_pos and not edible_ghosts:
                    final_map[j][i] = 2
                if (i, j) in food_grid:
                    final_map[j][i] = 3
                if (i, j) in power_pellets:
                    final_map[j][i] = 4
                if (i, j) in wall_position:
                    final_map[j][i] = 5

        # Create the final row with the current score
        score_value = game_state.get_score()
        final_row = np.full((1, i_range), score_value)

        # Add the final row to the final_map
        final_map = np.concatenate((final_map, final_row), axis=0)

        return final_map.flatten()



    def get_state_representation(self, game_state):
        """
        Convert the game state into a neural network-friendly format.
        """
        # Extract necessary features from game_state
        # For example:
        agent_pos = game_state.get_agent_position(self.index)
        ghosts_pos = game_state.ghost_positions(self.index)
        food_grid = game_state.get_blue_food().as_list(True)
        power_pellets = game_state.get_blue_capsules()

        # Get the agent's state
        my_state = game_state.get_agent_state(self.index)
        edible_ghosts = my_state.scared_timer > 5

        wall_position = game_state.get_walls().as_list(True)
        done = game_state.is_over()
        score = game_state.get_score()

        # Convert features to a suitable format
        # Assuming a simple vector representation
        state_representation = self.vector_state_representation(
            game_state, agent_pos, ghosts_pos, food_grid, power_pellets, wall_position, edible_ghosts, score
        )

        return state_representation

    def get_epsilon(self):
        """
        Get the current value of epsilon for the epsilon-greedy strategy.
        """
        # Implement a decreasing epsilon strategy
        # For example, a simple linear decay:
        start_epsilon = 1.0
        end_epsilon = 0.1
        decay_rate = 0.001

        epsilon = max(end_epsilon, start_epsilon - decay_rate * self.step_count )
        self.step_count +=1
        return epsilon
    
    def save_cumulative_reward(self):
        # Ensure the directory exists
        os.makedirs('training_results', exist_ok=True)
        file_path = f'training_results/cumulative_rewards.csv'

        # Check if the file already exists and has content
        write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header only if needed
            if write_header:
                writer.writerow(['Training ID', 'Cumulative Reward'])

            # Append the data row
            writer.writerow([self.training_id, self.cumulative_reward])
    
    def final(self, state):
        file_path = "ddql_agent_state.pth"
        self.save_model(file_path)
        # Save cumulative reward to CSV
        if TRAINING:
            self.save_cumulative_reward()

