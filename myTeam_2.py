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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint
import pickle
import numpy as np
import json

import csv
import uuid
import os
import glob

#################
# Team creation #
#################

TRAINING_offensive = False
TRAINING_defensive = False
defensive_weights = 'training_results_4\defensive_agent_weights_613de35e-a3b5-4f6b-bbe5-579884aa87f0.json'
offensive_weights  = 'default_offensive_weights.json'
def create_team(firstIndex, secondIndex, isRed,
                first='OffensiveQLearningAgent', second='DefensiveQLearningAgent', **args):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class OffensiveQLearningAgent(CaptureAgent):
    """
    The OffensiveQLearningAgent is a strategic agent designed for playing an offensive role in the Capture the Flag 
    scenario of the Pac-Man game. It leverages Q-learning, a reinforcement learning technique, to make informed 
    decisions and optimize its offensive gameplay. 
    
    Here are the key features and strategies employed by the OffensiveQLearningAgent:

        1. Trained Weights Loading: The agent has the capability to load pre-trained weights from a file (trained_agent_weights.pkl). 
        This feature allows the agent to benefit from past learning experiences, enhancing its decision-making.
        
        2. Q-learning Algorithm Parameters: During training, the agent dynamically adjusts its exploration-exploitation trade-off. 
        The exploration rate (epsilon) is set to 0.15 for training, encouraging exploration, and 0 during gameplay, promoting exploitation 
        of learned knowledge. The learning rate (alpha) is set to 0.2, determining the impact of new information on the agent's existing knowledge.
        
        3. Action Selection: The agent dynamically selects actions based on its current knowledge and game state. It favors 
        returning home when carrying a significant amount of food or when close to victory conditions.
        
        4. Feature Extraction: Relevant features, such as proximity to ghosts, distance to nearby food, and the urgency to return home with food,
        are extracted to inform the agent's decision-making process.
        
        5. Weighted Q-value Computation: Q-values, representing the expected cumulative rewards for specific actions, 
        are calculated based on a weighted combination of features and learned weights. This forms the basis for the agent's decision-making.
        
        6. Weight Updating: The agent updates its knowledge by adjusting weights using the Q-learning update rule. This update considers 
        observed rewards, discount factors, and predicted future rewards.
        
        7. Reward Calculation: Rewards are calculated based on various factors, including proximity to enemies, progress towards returning home, 
        changes in the game score, and distance to the nearest food.
        
        8. Persistence and Learning: Trained weights are persisted to a file (trained_agent_weights.pkl) after each game, ensuring that the agent retains 
        and builds upon its learned knowledge over multiple game sessions.
        
        9. Ghost Position Inference: The agent utilizes Bayesian inference to estimate the positions of opponents, particularly ghosts, 
        based on observations and the passage of time.
        
    In summary, the OffensiveQLearningAgent demonstrates a strategic and adaptive approach to offensive gameplay, 
    employing Q-learning principles to optimize its actions in the dynamic environment of Capture the Flag. 
    The agent's continuous learning, feature-based decision-making, and integration with game-specific strategies 
    contribute to its effectiveness as an offensive player.

    """

    '''def load_weights(self):
        # Initialize weights to None
        weights = None

        # try:
        #     # Try to open the file and load weights from it
        #     with open('./trained_agent_weights.pkl', 'rb') as file:
        #         weights = pickle.load(file)

        # except (FileNotFoundError, IOError):
        # If the file is not found or an error occurs, provide default weights
        weights = {
            'bias': -9.1234412,
            'food_close': -2.983928392083,
            'ghosts_close': -3.65065432233,
            'food_eaten': 15.12232122121,
            'carrying_food_go_home': 1.822389123231
        }

        return weights'''
    def load_weights(self):
        if not offensive_weights:
            # Load weights from file or initialize to default values
            json_files = glob.glob('training_results_offensive/offensive_agent_weights_*.json')
            if not json_files:
                # Generate weights randomly
                food_eaten = random.uniform(-16, 16)  # Random number between -10 and 0
                carrying_food_go_home = random.uniform(-16, food_eaten)  # Smaller than food_defending_weight
                food_close = random.uniform(-16, carrying_food_go_home)  # Smallest of all
                ghosts_close = random.uniform(-16, food_close)
                bias = random.uniform(-16, ghosts_close)

                return {
                    'bias': bias,
                    'food_close': food_close,
                    'ghosts_close': ghosts_close,
                    'food_eaten': food_close,
                    'carrying_food_go_home': carrying_food_go_home
                }
            # Find the most recent file
            latest_file = max(json_files, key=os.path.getctime)
            
            # Extract training ID from the filename
            training_id = os.path.basename(latest_file).split('_')[3].split('.')[0]

            # Load the corresponding weights file
            weights_file = f'training_results_offensive/offensive_agent_weights_{training_id}.json'
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as file:
                    return json.load(file)
            else:
                # Generate weights randomly
                food_eaten = random.uniform(-16, 16)  # Random number between -10 and 0
                carrying_food_go_home = random.uniform(-16, food_eaten)  # Smaller than food_defending_weight
                food_close = random.uniform(-16, carrying_food_go_home)  # Smallest of all
                ghosts_close = random.uniform(-16, food_close)
                bias = random.uniform(-16, ghosts_close)

                return {
                    'bias': bias,
                    'food_close': food_close,
                    'ghosts_close': ghosts_close,
                    'food_eaten': food_close,
                    'carrying_food_go_home': carrying_food_go_home
                }
        else:
            with open(offensive_weights, 'r') as file:
                    return json.load(file)

    def register_initial_state(self, game_state):
        # Important variables related to the Q learning algorithm
        # When playing we don't want any exploration, strictly on policy
        if TRAINING_offensive:
            self.epsilon = 0.15
        else:
            self.epsilon = 0
        self.alpha = 0.2
        self.discount = 0.8
        self.weights = self.load_weights()

        self.initial_position = game_state.get_agent_position(self.index)
        self.legal_positions = game_state.get_walls().as_list(False)
        self.cumulative_reward = 0
        CaptureAgent.register_initial_state(self, game_state)

        # Initialize the Bayesian Inference for the ghost positions
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(game_state)}
        for enemy in self.get_opponents(game_state):
            self.obs[enemy][game_state.get_initial_agent_position(enemy)] = 1.0

    def run_home_action(self, game_state):
        """
        Choose the action that brings the agent closer to its initial position.

        Args:
            game_state (GameState): The current game state.

        Returns:
            str: The chosen action.
        """
        # Initialize the best distance to a large value
        best_dist = 10000

        for action in game_state.get_legal_actions(self.index):
            # Get the successor state after taking the current action
            successor = self.get_successor(game_state, action)

            # Get the agent's position in the successor state
            pos2 = successor.get_agent_position(self.index)

            # Calculate the distance from the initial position to the new position
            dist = self.get_maze_distance(self.initial_position, pos2)

            # Update the best action if the new distance is smaller
            if dist < best_dist:
                bestAction = action
                best_dist = dist

        # Return the best action that brings the agent closer to its initial position
        return bestAction

    def choose_action(self, game_state):
        """
        Choose an action based on the Q-values and exploration-exploitation strategy.

        Args:
            game_state (GameState): The current game state.

        Returns:
            str: The chosen action.
        """
        action = None
        legal_actions = game_state.get_legal_actions(self.index)

        # Return None if no legal actions are available
        if len(legal_actions) == 0:
            return None

        # If the agent has collected enough food to win, always return home
        # food_left = len(self.get_food(game_state).as_list())
        # if food_left <= 2:
        #     return self.run_home_action(game_state)

        # If the agent is carrying a significant amount of food, prioritize returning home
        # original_agent_state = game_state.get_agent_state(self.index)
        # if original_agent_state.num_carrying > 3:
        #     return self.run_home_action(game_state)

        # If in training mode, update weights based on the current state and actions
        if TRAINING_offensive:
            for action in legal_actions:
                self.update_weights(game_state, action)

        # Determine whether to exploit or explore based on the epsilon value
        if not util.flipCoin(self.epsilon):
            # Exploit: Choose the action with the highest Q-value
            action = self.compute_action_from_q_values(game_state)
        else:
            # Explore: Randomly choose an action
            action = random.choice(legal_actions)

        return action

    def is_ghost_within_steps(self, agentPos, ghostPos, steps, walls):
        """
        Check if a ghost is within a specified number of steps from the agent.

        Args:
            agentPos (tuple): The current position of the agent.
            ghostPos (tuple): The position of the ghost.
            steps (int): The maximum number of steps allowed.
            walls (Grid): The grid representing the walls.

        Returns:
            bool: True if the ghost is within the specified number of steps, False otherwise.
        """
        # Calculate the distance between the agent and the ghost
        distance = self.get_maze_distance(agentPos, ghostPos)

        # Check if the distance is within the specified number of steps
        return distance <= steps

    def get_num_of_ghost_in_proximity(self, game_state, action):
        """
        Get the number of ghosts in proximity after taking a specific action.

        Args:
            game_state (GameState): The current game state.
            action (str): The chosen action to take.

        Returns:
            int: The number of ghosts in proximity.
        """
        # Extract the grid of food and wall locations and get the ghost locations
        food = self.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_idx = [i for i in self.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Update ghost positions using Bayesian inference if no ghosts are currently visible
        max_vals = list()
        if len(ghosts) == 0:
            for e_idx in enemies_idx:
                self.observe(e_idx, game_state)
                self.elapse_time(e_idx, game_state)
                belief_dist_e = self.obs[e_idx]
                max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                max_vals.append(max_position)
            ghosts = list(set(max_vals))

        # Get the agent's position after taking the specified action
        agentPosition = game_state.get_agent_position(self.index)
        x, y = agentPosition
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Count the number of ghosts within 3 steps from the new position
        return sum(self.is_ghost_within_steps((next_x, next_y), g, 3, walls) for g in ghosts)

    def calculate_carrying_food_go_home_feature(self, game_state, agent_position, action):
        """
        Calculate a feature indicating the desirability of going near home when carrying food.

        Args:
            game_state (GameState): The current game state.
            agent_position (tuple): The current position of the agent.

        Returns:
            float: The calculated feature value.
        """
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        original_agent_state = game_state.get_agent_state(self.index)
        amount_of_food_carrying = original_agent_state.num_carrying
        # print("Food carrying: ", amount_of_food_carrying )
        # print("Distance home: ", ((game_state.get_walls().width / 3) -  self.get_maze_distance(self.initial_position, agent_position)))

        return amount_of_food_carrying / -(
                    (game_state.get_walls().width / 3) - self.get_maze_distance(self.initial_position,
                                                                                (next_x, next_y)))

    def get_features(self, game_state, action):
        """
        Compute and return a set of features that describe the game state after taking a given action.

        Args:
            game_state (GameState): The current game state.
            action (str): The chosen action to take.

        Returns:
            util.Counter: A set of features as a Counter.
        """
        # Initialize an empty Counter to store the features
        features = util.Counter()

        # Compute the location of the agent after taking the action
        agent_position = game_state.get_agent_position(self.index)
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Set a bias feature with a constant value
        features["bias"] = 1.0

        features["score"] = 1.0
        # Get the number of ghosts in proximity and set a feature accordingly
        features["ghosts_close"] = self.get_num_of_ghost_in_proximity(game_state, action)

        features["food_eaten"] = 1.0

        # Calculate the distance to the closest food and set a feature accordingly
        dist = self.closest_food((next_x, next_y), self.get_food(game_state), game_state.get_walls())
        if dist is not None:
            features["food_close"] = float(dist) / (game_state.get_walls().width * game_state.get_walls().height)

        # Calculate the carrying_food_go_home feature and set the corresponding feature
        features["carrying_food_go_home"] = self.calculate_carrying_food_go_home_feature(game_state, agent_position,
                                                                                         action)

        return features

    def closest_food(self, pos, food, walls):
        frontier = [(pos[0], pos[1], 0)]
        expanded = set()
        while frontier:
            pos_x, pos_y, dist = frontier.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))

            if food[pos_x][pos_y]:
                return dist

            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                frontier.append((nbr_x, nbr_y, dist + 1))
        return None

    def elapse_time(self, enemy, game_state):
        """
        Args:
            enemy: The opponent for which the belief is updated.
            game_state (GameState): The current game state.
        """
        # Define a lambda function to calculate possible next positions
        possible_positions = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over the previous positions and their probabilities
        for prev_pos, prev_prob in self.obs[enemy].items():
            # Calculate the new possible positions for the enemy
            new_obs = util.Counter(
                {pos: 1.0 for pos in possible_positions(prev_pos[0], prev_pos[1]) if pos in self.legal_positions})

            # Normalize the new observation probabilities
            new_obs.normalize()

            # Update the overall belief distribution with the new probabilities
            for new_pos, new_prob in new_obs.items():
                all_obs[new_pos] += new_prob * prev_prob

        # Check for any food eaten by opponents
        foods = self.get_food_you_are_defending(game_state).as_list()
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()).as_list() if self.get_previous_observation() else []

        # If the number of foods has decreased, adjust the belief distribution
        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(game_state))

        # Update the agent's belief about the opponent's positions
        self.obs[enemy] = all_obs

    def observe(self, enemy, game_state):
        """
        Observes and updates the agent's belief about the opponent's position.

        Args:
            enemy: The opponent for which the belief is updated.
            game_state (GameState): The current game state.
        """
        # Get distance observations for all agents
        all_noise = game_state.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = game_state.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(game_state.teams) if value]
        team_pos = [game_state.get_agent_position(team) for team in team_idx]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over all legal positions on the board
        for pos in self.legal_positions:
            # Check if any teammate is close to the current position
            team_dist = [team for team in team_pos if team is not None and util.manhattanDistance(team, pos) <= 5]

            if team_dist:
                # If a teammate is close, set the probability of this position to 0
                all_obs[pos] = 0.0
            else:
                # Calculate the true distance between Pacman and the current position
                true_distance = util.manhattanDistance(my_pos, pos)

                # Get the probability of observing the noisy distance given the true distance
                pos_prob = game_state.get_distance_prob(true_distance, noisy_distance)

                # Update the belief distribution with the calculated probability
                all_obs[pos] = pos_prob * self.obs[enemy][pos]

        # Check if there are any non-zero probabilities in the belief distribution
        if all_obs.totalCount():
            # Normalize the belief distribution if there are non-zero probabilities
            all_obs.normalize()

            # Update the agent's belief about the opponent's positions
            self.obs[enemy] = all_obs

    def get_q_value(self, game_state, action):
        features = self.get_features(game_state, action)
        # print("features: ", features, "value: ", features * self.weights)
        return features * self.weights

    def update(self, game_state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.get_features(game_state, action)
        oldValue = self.get_q_value(game_state, action)
        futureQValue = self.compute_value_from_q_values(game_state)
        difference = (reward + self.discount * futureQValue) - oldValue
        # for each feature i
        for feature in features:
            if feature not in self.weights:
                self.weights[feature] = 0  # Initialize with a default value, like 0
            newWeight = np.clip(self.alpha * difference * features[feature], -16, 16)
            self.weights[feature] += newWeight
        # Update cumulative reward if in training mode
        if TRAINING_defensive:
            self.cumulative_reward += reward
        # print(self.weights)

    def update_weights(self, game_state, action):
        """
        Update the weights based on the reward received after taking a specific action.

        Args:
            game_state (GameState): The current game state.
            action (str): The chosen action to update weights.
        """
        # Get the successor state after taking the specified action
        nextState = self.get_successor(game_state, action)

        # Get the reward for the transition from the current state to the successor state
        reward = self.get_reward(game_state, nextState)

        # Update the weights based on the current and successor states, the chosen action, and the reward
        self.update(game_state, action, nextState, reward)

    def get_reward(self, game_state, nextState):
        """
        Calculate and return the total reward for a given state transition.

        Args:
            game_state (GameState): The current game state.
            nextState (GameState): The successor state after taking an action.

        Returns:
            float: The total reward for the state transition.
        """
        # Get the agent's position in the current game state
        agent_position = game_state.get_agent_position(self.index)

        # Calculate rewards for different aspects and sum them up
        go_home_reward = self.calculate_carrying_food_go_home_reward(nextState)
        score_reward = self.calculate_score_reward(game_state, nextState)
        dist_to_food_reward = self.calculate_dist_to_food_reward(game_state, nextState, agent_position)
        enemies_reward = self.calculate_enemies_reward(game_state, nextState, agent_position)

        # Display individual rewards for debugging purposes
        rewards = {"enemies": enemies_reward, "go_home": go_home_reward, "dist_to_food_reward": dist_to_food_reward,
                   "score": score_reward}
        print("REWARDS:", rewards)

        # Return the sum of all rewards
        return sum(rewards.values())

    def calculate_carrying_food_go_home_reward(self, nextState):
        """
        Calculate a feature indicating the desirability of going near home when carrying food.

        Args:
            game_state (GameState): The current game state.
            agent_position (tuple): The current position of the agent.

        Returns:
            float: The calculated feature value.
        """

        original_agent_state = nextState.get_agent_state(self.index)
        amount_of_food_carrying = original_agent_state.num_carrying

        agent_position = nextState.get_agent_position(self.index)
        # print("Food carrying: ", amount_of_food_carrying )
        # print("Distance home: ", ((game_state.get_walls().width / 3) -  self.get_maze_distance(self.initial_position, agent_position)))

        return amount_of_food_carrying / -(
                    (nextState.get_walls().width / 3) - self.get_maze_distance(self.initial_position, agent_position))

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
        dist_to_food_reward = 0

        # Get the list of coordinates of the agent's food in the current state
        my_foods = self.get_food(game_state).as_list()

        # Get the minimum distance to food in the current state
        dist_to_food = min([self.get_maze_distance(agent_position, food) for food in my_foods])

        # Check if the agent reached a food in the next state
        if dist_to_food == 1:
            # Get the list of coordinates of the agent's food in the next state
            next_foods = self.get_food(nextState).as_list()

            # Check if one food was eaten in the next state
            if len(my_foods) - len(next_foods) == 1:
                # Update the dist_to_food_reward
                dist_to_food_reward += 20

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
    def save_cumulative_reward(self, training_id):
        # Ensure the directory exists
        os.makedirs('training_results_offensive', exist_ok=True)
        file_path = f'training_results_offensive/offensive_cumulative_rewards.csv'

        # Check if the file already exists and has content
        write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['Training ID', 'Cumulative Reward'])
            writer.writerow([training_id, self.cumulative_reward])

    def final(self, state):
        CaptureAgent.final(self, state)
        # Generate a unique identifier for the training session
        if TRAINING_offensive:
            training_id = str(uuid.uuid4())
            self.save_cumulative_reward(training_id)
            # Define the file path for saving the weights
            weights_path = f'training_results_offensive/offensive_agent_weights_{training_id}.json'
            # Save the weights to the specified file
            with open(weights_path, 'w') as file:
                json.dump(self.weights, file)

    def compute_value_from_q_values(self, game_state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        allowedActions = game_state.get_legal_actions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.compute_action_from_q_values(game_state)
        return self.get_q_value(game_state, bestAction)

    def compute_action_from_q_values(self, game_state):

        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None

        actionVals = {}
        bestQValue = float('-inf')
        #print("=============================")
        for action in legal_actions:
            #print("Action: ", action)
            target_q_value = self.get_q_value(game_state, action)
            actionVals[action] = target_q_value
            if target_q_value > bestQValue:
                bestQValue = target_q_value
        bestActions = [k for k, v in actionVals.items() if v == bestQValue]
        # random tie-breaking
        return random.choice(bestActions)

class ReflexCaptureBaseAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def register_initial_state(self, game_state):
        self.initial_position = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.walls = game_state.get_walls()

        self.legal_positions = game_state.get_walls().as_list(False)
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(game_state)}
        for enemy in self.get_opponents(game_state):
            self.obs[enemy][game_state.get_initial_agent_position(enemy)] = 1.0
    def is_red_team(self, game_state):
        """
        Check if the agent is on the red team.

        Args:
            game_state (GameState): The current game state.

        Returns:
            bool: True if the agent is on the red team, False otherwise.
        """
        return self.index in game_state.red_team
    
    def get_zone_map(self, game_state, is_red_team):
        """
        Create a binary 2D array representing the safe zone and danger zone.

        Args:
            game_state (GameState): The current game state.
            is_red_team (bool): True if your team is the red team, False if blue.

        Returns:
            List[List[int]]: A 2D array with 0's indicating your safe zone and 1's indicating the enemy's zone.
        """
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2

        # Create a 2D array initialized to 0
        zone_map = [[0 for _ in range(height)] for _ in range(width)]

        # Fill the enemy's half with 1's
        start_x = mid_x if is_red_team else 0
        end_x = width if is_red_team else mid_x
        for x in range(start_x, end_x):
            for y in range(height):
                if not walls[x][y]:  # Only mark non-wall positions
                    zone_map[x][y] = 1

        return zone_map
    def elapse_time(self, enemy, game_state):

        # Define a lambda function to calculate possible next positions
        possible_positions = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over the previous positions and their probabilities
        for prev_pos, prev_prob in self.obs[enemy].items():
            # Calculate the new possible positions for the enemy
            new_obs = util.Counter(
                {pos: 1.0 for pos in possible_positions(prev_pos[0], prev_pos[1]) if pos in self.legal_positions})

            # Normalize the new observation probabilities
            new_obs.normalize()

            # Update the overall belief distribution with the new probabilities
            for new_pos, new_prob in new_obs.items():
                all_obs[new_pos] += new_prob * prev_prob

        # Check for any food eaten by opponents
        foods = self.get_food_you_are_defending(game_state).as_list()
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()).as_list() if self.get_previous_observation() else []

        # If the number of foods has decreased, adjust the belief distribution
        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(game_state))

        # Update the agent's belief about the opponent's positions
        self.obs[enemy] = all_obs

    def observe(self, enemy, game_state):
        """
        Updates beliefs based on the distance observation and Pacman's position.
        """
        # Get distance observations for all agents
        all_noise = game_state.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = game_state.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(game_state.teams) if value]
        team_pos = [game_state.get_agent_position(team) for team in team_idx]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over all legal positions on the board
        for pos in self.legal_positions:
            # Check if any teammate is close to the current position
            team_dist = [team for team in team_pos if util.manhattanDistance(team, pos) <= 5]

            if team_dist:
                # If a teammate is close, set the probability of this position to 0
                all_obs[pos] = 0.0
            else:
                # Calculate the true distance between Pacman and the current position
                true_distance = util.manhattanDistance(my_pos, pos)

                # Get the probability of observing the noisy distance given the true distance
                pos_prob = game_state.get_distance_prob(true_distance, noisy_distance)

                # Update the belief distribution with the calculated probability
                all_obs[pos] = pos_prob * self.obs[enemy][pos]

        # Check if there are any non-zero probabilities in the belief distribution
        if all_obs.totalCount():
            # Normalize the belief distribution if there are non-zero probabilities
            all_obs.normalize()

            # Update the agent's belief about the opponent's positions
            self.obs[enemy] = all_obs
        # else:
        # If no valid observations, initialize the belief distribution
        # self.initialize(enemy, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a) for the agent.

        Args:
            game_state (GameState): The current game state.

        Returns:
            str: The chosen action.
        """
        actions = game_state.get_legal_actions(self.index)

        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        # Select the best action based on the highest Q-value
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.initial_position, pos2)
                if dist < best_dist:
                    bestAction = action
                    best_dist = dist
            return bestAction

        return random.choice(bestActions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor state after taking an action.

        Args:
            game_state (GameState): The current game state.
            action (str): The action to be taken.

        Returns:
            GameState: The successor game state.
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
        Computes a linear combination of features and feature weights.

        Args:
            game_state (GameState): The current game state.
            action (str): The action to be evaluated.

        Returns:
            float: The evaluated value.
        """
        features = self.get_features(game_state, action)
        weights = self.getWeights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state.

        Args:
            game_state (GameState): The current game state.
            action (str): The action for which features are computed.

        Returns:
            util.Counter: A counter containing the features.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successorScore'] = self.get_score(successor)
        return features

    def getWeights(self, game_state, action):
        """
        Returns a dictionary of feature weights for the given game state and action.

        Args:
            game_state (GameState): The current game state.
            action (str): The action for which weights are provided.

        Returns:
            dict: A dictionary containing feature weights.
        """
        return {'successorScore': 1.0}


class DefensiveQLearningAgent(CaptureAgent):
    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8, **args):
        super().__init__(index, **args)
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.weights = self.load_weights()  # load or initialize weights
        # Initialize the initial amount of food
        self.current_food_defending = None
        # Additional attributes for cumulative reward and unique identifier
        self.cumulative_reward = 0
        self.training_id = str(uuid.uuid4())  # Generate a unique identifier
    def get_food_you_are_defending(self, game_state):
        """
        Returns the food you're meant to protect (i.e., that your opponent is
        supposed to eat). This is in the form of a matrix where m[x][y]=true if
        there is food at (x,y) that your opponent can eat.
        """
        if self.red:
            return game_state.get_red_food().as_list()
        else:
            return game_state.get_blue_food().as_list()
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Set the initial amount of food in your home territory
        self.current_food_defending = self.get_food_you_are_defending(game_state)
        #Initialize the Bayesian Inference for the ghost positions
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(game_state)}
        for enemy in self.get_opponents(game_state):
            self.obs[enemy][game_state.get_initial_agent_position(enemy)] = 1.0
        self.legal_positions = game_state.get_walls().as_list(False)
        self.entry_points = self.identify_entry_points(game_state)
        self.current_patrol_index = 0
    def identify_entry_points(self, game_state):
        """Identify potential entry points in the territory."""
        mid_x = game_state.get_walls().width // 2
        if self.red:
            entry_points = [(mid_x - 1, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(mid_x - 1, y)]
        else:
            entry_points = [(mid_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(mid_x, y)]
        return entry_points
    def elapse_time(self, enemy, game_state):
        """
        Args:
            enemy: The opponent for which the belief is updated.
            game_state (GameState): The current game state.
        """
        # Define a lambda function to calculate possible next positions
        possible_positions = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over the previous positions and their probabilities
        for prev_pos, prev_prob in self.obs[enemy].items():
            # Calculate the new possible positions for the enemy
            new_obs = util.Counter(
                {pos: 1.0 for pos in possible_positions(prev_pos[0], prev_pos[1]) if pos in self.legal_positions})

            # Normalize the new observation probabilities
            new_obs.normalize()

            # Update the overall belief distribution with the new probabilities
            for new_pos, new_prob in new_obs.items():
                all_obs[new_pos] += new_prob * prev_prob

        # Check for any food eaten by opponents
        foods = self.get_food_you_are_defending(game_state)
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()) if self.get_previous_observation() else []

        # If the number of foods has decreased, adjust the belief distribution
        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(game_state))

        # Update the agent's belief about the opponent's positions
        self.obs[enemy] = all_obs

    def observe(self, enemy, game_state):
        """
        Observes and updates the agent's belief about the opponent's position.

        Args:
            enemy: The opponent for which the belief is updated.
            game_state (GameState): The current game state.
        """
        # Get distance observations for all agents
        all_noise = game_state.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = game_state.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(game_state.teams) if value]
        team_pos = [game_state.get_agent_position(team) for team in team_idx]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over all legal positions on the board
        for pos in self.legal_positions:
            # Check if any teammate is close to the current position
            team_dist = [team for team in team_pos if team is not None and util.manhattanDistance(team, pos) <= 5]

            if team_dist:
                # If a teammate is close, set the probability of this position to 0
                all_obs[pos] = 0.0
            else:
                # Calculate the true distance between Pacman and the current position
                true_distance = util.manhattanDistance(my_pos, pos)

                # Get the probability of observing the noisy distance given the true distance
                pos_prob = game_state.get_distance_prob(true_distance, noisy_distance)

                # Update the belief distribution with the calculated probability
                all_obs[pos] = pos_prob * self.obs[enemy][pos]

        # Check if there are any non-zero probabilities in the belief distribution
        if all_obs.totalCount():
            # Normalize the belief distribution if there are non-zero probabilities
            all_obs.normalize()

            # Update the agent's belief about the opponent's positions
            self.obs[enemy] = all_obs

    def load_weights(self):
        if not defensive_weights:
            # Load weights from file or initialize to default values
            json_files = glob.glob('training_results/defensive_agent_weights_*.json')
            if not json_files:
                # Generate weights randomly
                food_defending_weight = random.uniform(-10, 0)  # Random number between -10 and 0
                invaderDistance_weight = random.uniform(-10, food_defending_weight)  # Smaller than food_defending_weight
                closest_entry_point_weight = random.uniform(-10, invaderDistance_weight)  # Smallest of all

                return {
                    'food_defending': food_defending_weight,
                    'invaderDistance': invaderDistance_weight,
                    'closest_entry_point': closest_entry_point_weight
                }
            # Find the most recent file
            latest_file = max(json_files, key=os.path.getctime)
            
            # Extract training ID from the filename
            training_id = os.path.basename(latest_file).split('_')[3].split('.')[0]

            # Load the corresponding weights file
            weights_file = f'training_results/defensive_agent_weights_{training_id}.json'
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as file:
                    return json.load(file)
            else:
                # Generate weights randomly
                food_defending_weight = random.uniform(-16, 0)  # Random number between -10 and 0
                invaderDistance_weight = random.uniform(-16, food_defending_weight)  # Smaller than food_defending_weight
                closest_entry_point_weight = random.uniform(-16, invaderDistance_weight)  # Smallest of all

                return {
                    'food_defending': food_defending_weight,
                    'invaderDistance': invaderDistance_weight,
                    'closest_entry_point': closest_entry_point_weight
                }
        else:
            with open(defensive_weights, 'r') as file:
                    return json.load(file)

    def getQValue(self, state, action):
        features = self.get_features(state, action)
        #if features['food_defending']!=20:
        #    print('here')
        return sum(self.weights[f] * features[f] for f in features)
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
    def update_weights(self, game_state, action):
        """
        Update the weights based on the reward received after taking a specific action.

        Args:
            game_state (GameState): The current game state.
            action (str): The chosen action to update weights.
        """
        # Get the successor state after taking the specified action
        nextState = self.get_successor(game_state, action)

        # Get the reward for the transition from the current state to the successor state
        reward = self.get_reward(game_state, nextState)

        # Update the weights based on the current and successor states, the chosen action, and the reward
        self.update(game_state, action, nextState, reward)
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        # If in training mode, update weights based on the current state and actions
        if TRAINING_defensive:
            for action in actions:
                self.update_weights(game_state, action)
        if len(actions) == 0:
            return None
        # Filter out actions that lead out of home territory
        safe_actions = [a for a in actions if not self.is_leaving_home_territory(game_state, a)]
        if not safe_actions:
            safe_actions = actions  # fallback if no safe actions
        enemies_pos = self.get_enemies_pos2(game_state, Directions.STOP)  # Using STOP as a placeholder action
        #if not enemies_pos:  # If no enemies visible, switch to patrolling
        #    return self.select_patrol_action(game_state)
        if util.flipCoin(self.epsilon):
            return random.choice(safe_actions)
        else:
            return max(safe_actions, key=lambda action: self.getQValue(game_state, action))
    def select_patrol_action(self, game_state):
        """Select an action to patrol the entry points."""
        my_pos = game_state.get_agent_position(self.index)
        target_entry_point = self.entry_points[self.current_patrol_index]

        # Update patrol index if the current point is reached
        if my_pos == target_entry_point:
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.entry_points)
            target_entry_point = self.entry_points[self.current_patrol_index]

        # Choose action to move towards the target entry point
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        min_distance = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_state(self.index).get_position()
            if not self.is_leaving_home_territory(successor, action):  # Check if action stays in safe zone
                distance = self.get_maze_distance(successor_pos, target_entry_point)
                if distance < min_distance:
                    best_action = action
                    min_distance = distance

        return best_action

    def update(self, state, action, nextState, reward):
        correction = reward + self.gamma * self.compute_value_from_q_values(nextState) - self.getQValue(state, action)
        features = self.get_features(state, action)
        for feature in features:
            # Check for NaN or Inf in correction and feature value
            if not np.isnan(correction) and not np.isnan(features[feature]):
                weight_update = self.alpha * correction * features[feature]
                # Clip the weight update to prevent extreme changes
                weight_update = np.clip(weight_update, -16, 16)  # Adjust the range as necessary
                self.weights[feature] += weight_update
                # Round and check for NaN
                self.weights[feature] = round(self.weights[feature], 3)
                if np.isnan(self.weights[feature]):
                    self.weights[feature] = 0  # Reset to zero or some default value in case of NaN
        # Update cumulative reward if in training mode
        if TRAINING_defensive:
            self.cumulative_reward += reward
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

    def compute_value_from_q_values(self, state):
        legal_actions = state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        return max(self.getQValue(state, action) for action in legal_actions)
    # Add methods to check if leaving home territory and to get distance to safe zone
    def is_leaving_home_territory(self, game_state, action):
        red_team_bool = self.is_red_team(game_state)
        zone_map = self.get_zone_map(game_state, red_team_bool)
        my_pos = game_state.get_agent_position(self.index)
        # Get the agent's position after taking the specified action
        x, y = my_pos
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        if zone_map[next_x][next_y]:
            return True
        else:
            return False
    def get_enemies_pos2(self,game_state,action):
        my_pos = game_state.get_agent_position(self.index)
        # Get the agent's position after taking the specified action
        x, y = my_pos
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        my_pos = (next_x, next_y)
        #search if a ghost is near by the border
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_idx = [i for i in self.get_opponents(game_state)]
        #ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]
        #check if there is any visible enemy
        enemies_pos = [a.get_position() for a in enemies if a.get_position() is not None]
        return enemies_pos
    def get_enemies_pos(self, game_state, action):
        #this function retrieves if there is any invader visible by Bayes or by my agent
        my_pos = game_state.get_agent_position(self.index)
        # Get the agent's position after taking the specified action
        x, y = my_pos
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        my_pos = (next_x, next_y)
        #search if a ghost is near by the border
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_idx = [i for i in self.get_opponents(game_state)]
        #ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]
        #check if there is any visible enemy
        enemies_pos = [a.get_position() for a in enemies if a.get_position() is not None]
        enemies_pos_b = []
        if len(enemies_pos) == 0:
            #if there are no visible enemies, try to infer them using the Bayes location
            red_team_bool = self.is_red_team(game_state)
            zone_map = self.get_zone_map(game_state, red_team_bool)
            max_vals = list()
            for e_idx in enemies_idx:
                self.observe(e_idx, game_state)
                self.elapse_time(e_idx, game_state)
                belief_dist_e = self.obs[e_idx]
                #Previous approach
                #max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                #max_vals.append(max_position)
                #New approach: get top 4 danger zones in the map and set them as "danger zones" possibly ghosts
                # Filter and sort items with values greater than 0
                filtered_sorted_items = sorted(
                    [(position, prob) for position, prob in belief_dist_e.items() if prob > 0],
                    key=lambda item: item[1], 
                    reverse=True
                )
                # Get the top 4 or fewer items
                top_items = filtered_sorted_items[:min(4, len(filtered_sorted_items))]
                for item in top_items:
                    (col,row) = item[0]
                    #only append enemies when they are in the danger zone
                    if zone_map[col][row]==1:
                        max_vals.append(item[0])  # Append each item in top_items to max_vals'''
            enemies_pos_b = list(set(max_vals))

        if len(enemies_pos) != 0:
            return enemies_pos
        else:
            return enemies_pos_b
    def is_red_team(self, game_state):
        """
        Check if the agent is on the red team.

        Args:
            game_state (GameState): The current game state.

        Returns:
            bool: True if the agent is on the red team, False otherwise.
        """
        return self.index in game_state.red_team
    
    def get_zone_map(self, game_state, is_red_team):
        """
        Create a binary 2D array representing the safe zone and danger zone.

        Args:
            game_state (GameState): The current game state.
            is_red_team (bool): True if your team is the red team, False if blue.

        Returns:
            List[List[int]]: A 2D array with 0's indicating your safe zone and 1's indicating the enemy's zone.
        """
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2

        # Create a 2D array initialized to 0
        zone_map = [[0 for _ in range(height)] for _ in range(width)]

        # Fill the enemy's half with 1's
        start_x = mid_x if is_red_team else 0
        end_x = width if is_red_team else mid_x
        for x in range(start_x, end_x):
            for y in range(height):
                if not walls[x][y]:  # Only mark non-wall positions
                    zone_map[x][y] = 1

        return zone_map
    def get_invader_distance(self, game_state, action):
        #this function retrieves if there is any invader visible by Bayes or by my agent
        my_pos = game_state.get_agent_position(self.index)
        # Get the agent's position after taking the specified action
        x, y = my_pos
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        my_pos = (next_x, next_y)
        enemies_pos = self.get_enemies_pos(game_state, action)
        if len(enemies_pos)==0:
            #assume they are very far away if I don't see anything
            return 10000

        red_team_bool = self.is_red_team(game_state)
        zone_map = self.get_zone_map(game_state, red_team_bool)
        enemies_in_my_home = [0 if zone_map[int(e_pos[0])][int(e_pos[1])] == 1 else 1 for e_pos in enemies_pos]
        if len(enemies_in_my_home)==0:
            return 10000
        else:
            min_dist = 10000
            min_dist_en = None
            for enemy_pos in enemies_pos:
                dist = self.get_maze_distance(my_pos, enemy_pos)
                if dist<min_dist:
                    min_dist = dist
                    min_dist_en = enemy_pos
            return min_dist
        
    def get_enemy_distance_to_my_home(self, game_state, action):
        if self.get_invader_distance(game_state, action)!=10000:
            #there are enemies in my home
            return 0, None
        else:
            red_team_bool = self.is_red_team(game_state)
            zone_map = self.get_zone_map(game_state, red_team_bool)
            enemies_pos = self.get_enemies_pos(game_state, action)
            if len(enemies_pos) == 0:
                # I don't see any enemy but there maybe some near to my home
                return 3, None
            # Determine the x-coordinate of the border that divides the map
            border_x = (game_state.get_walls().width // 2) - 1 if self.red else (game_state.get_walls().width // 2)

            #there are no enemies in my home
            posible_desired_doors=[]
            if red_team_bool:
                enemy_boarder_idx = border_x+1
            else:
                enemy_boarder_idx = border_x-1
            #search the doors where the enemy can enter my home
            y_s = zone_map[enemy_boarder_idx]
            for y in range(len(y_s)):
                if y_s[y]==1:
                    posible_desired_doors.append((enemy_boarder_idx,y))
            #search for the door where the enemy is getting closer
            min_dist = 10000
            all_possible_pos_and_dists = []
            for door in posible_desired_doors:
                for enemy in enemies_pos:
                    dist = self.get_maze_distance(door, enemy)
                    #get the door that the enemies are getting closer
                    all_possible_pos_and_dists.append((dist, door))
            # Find the tuple with the minimum distance
            min_dist_tuple = min(all_possible_pos_and_dists, key=lambda x: x[0])

            # Extract the door from the tuple with the minimum distance
            min_dist = min_dist_tuple[0]
            most_possible_door = min_dist_tuple[1] 
            return min_dist, most_possible_door
    def check_food_difference(self,food1,food2):
        num_f_1 = 0
        num_f_2 = 0
        for i in range(len(food1)):
            for j in range(len(food1[i])):
                if food1[i][j] == True:
                    num_f_1 = num_f_1 + 1
                if food2[i][j] == True:
                    num_f_2 = num_f_2 + 1
        difference = abs(num_f_1 - num_f_2)
        if difference > 0:
            return True
        else:
            return False
    def is_ghost_within_steps(self, agentPos, ghostPos, steps, walls):
        """
        Check if a ghost is within a specified number of steps from the agent.

        Args:
            agentPos (tuple): The current position of the agent.
            ghostPos (tuple): The position of the ghost.
            steps (int): The maximum number of steps allowed.
            walls (Grid): The grid representing the walls.

        Returns:
            bool: True if the ghost is within the specified number of steps, False otherwise.
        """
        # Calculate the distance between the agent and the ghost
        distance = self.get_maze_distance(agentPos, ghostPos)

        # Check if the distance is within the specified number of steps
        return distance <= steps
    def get_eating_enemy_reward(self, game_state, nextState):
        """
        Calculates the reward for eating an enemy invader in the agent's territory.

        Args:
            game_state (GameState): The current game state.
            nextState (GameState): The game state after taking an action.

        Returns:
            float: The reward for eating an enemy.
        """
        reward = 0.0
        my_pos = nextState.get_agent_position(self.index)
        current_invaders = self.get_invaders(game_state)
        next_invaders = self.get_invaders(nextState)

        # Check if the number of invaders has decreased, implying an enemy was eaten
        if len(next_invaders) < len(current_invaders):
            # Check if the agent's position coincides with one of the invader's last known positions
            for invader in current_invaders:
                if my_pos == invader.get_position():
                    reward += 1.0  # Assign a significant positive reward for eating an invader
                    break
        if reward > 0: 
            return reward 
        else: return -1.0
        

    def get_invaders(self, game_state):
        """
        Identifies invaders in the agent's territory.

        Args:
            game_state (GameState): The current game state.

        Returns:
            list: A list of enemy agent states who are invaders.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        return invaders

    def get_num_of_ghost_in_proximity(self, game_state, action):
        """
        Get the number of ghosts in proximity after taking a specific action.

        Args:
            game_state (GameState): The current game state.
            action (str): The chosen action to take.

        Returns:
            int: The number of ghosts in proximity.
        """
        # Extract the grid of food and wall locations and get the ghost locations
        food = self.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_idx = [i for i in self.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Update ghost positions using Bayesian inference if no ghosts are currently visible
        red_team_bool = self.is_red_team(game_state)
        zone_map = self.get_zone_map(game_state, red_team_bool)
        max_vals = list()
        if len(ghosts) == 0:
            for e_idx in enemies_idx:
                self.observe(e_idx, game_state)
                self.elapse_time(e_idx, game_state)
                belief_dist_e = self.obs[e_idx]
                #Previous approach
                #max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                #max_vals.append(max_position)
                #New approach: get top 4 danger zones in the map and set them as "danger zones" possibly ghosts
                # Filter and sort items with values greater than 0
                filtered_sorted_items = sorted(
                    [(position, prob) for position, prob in belief_dist_e.items() if prob > 0],
                    key=lambda item: item[1], 
                    reverse=True
                )
                # Get the top 4 or fewer items
                top_items = filtered_sorted_items[:min(4, len(filtered_sorted_items))]
                for item in top_items:
                    (col,row) = item[0]
                    #only append enemies when they are in the danger zone
                    if zone_map[col][row]==1:
                        max_vals.append(item[0])  # Append each item in top_items to max_vals'''
            ghosts = list(set(max_vals))

        # Get the agent's position after taking the specified action
        agentPosition = game_state.get_agent_position(self.index)
        x, y = agentPosition
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Count the number of ghosts within 1 steps from the new position
        return sum(self.is_ghost_within_steps((next_x, next_y), g, 1, walls) for g in ghosts)
    def get_distance_to_closest_entry_point(self, game_state, action):
        my_pos = self.get_successor(game_state, action).get_agent_state(self.index).get_position()
        distances = [self.get_maze_distance(my_pos, entry) for entry in self.entry_points]
        return min(distances) if distances else 0

    def get_features(self, game_state, action):
        # Define and extract features relevant for defense
        features = util.Counter()
        invader_distance = self.get_invader_distance(game_state, action)
        # distance to the nearest invader, the closest the best
        features['invaderDistance'] = invader_distance
        # get my amount of food
        features['food_defending'] = len(self.get_food_you_are_defending(game_state))
        features['closest_entry_point'] = self.get_distance_to_closest_entry_point(game_state, action)

        return features
    def get_eaten_reward(self, nextState):
        my_pos = nextState.get_agent_position(self.index)
        enemies = [nextState.get_agent_state(i) for i in self.get_opponents(nextState)]
        enemies_idx = [i for i in self.get_opponents(nextState)]
        is_scared = True if nextState.get_agent_state(self.index).scared_timer > 0 else False
        ghosts = [a.get_position() for a in enemies if a.is_pacman and a.get_position() is not None and is_scared ]
        if len(ghosts) == 0:
            #no ghosts near to me for the nex state
            return 0.6
        else:
            for en_idx in enemies_idx:
                en_pos = nextState.get_agent_position(en_idx)
                if en_pos == my_pos:
                    # a ghost can eat me in the next state
                    return -0.6
            #no ghosts near to me for the nex state
            return 0.6
    def get_reward(self, game_state, nextState):
        # Existing reward calculation
        reward = 0.0
        current_food = self.get_food_you_are_defending(game_state)
        future_food = self.get_food_you_are_defending(nextState)
        
        if self.check_food_difference(current_food, future_food):
            reward += -0.4
        else:
            reward += 0.4

        get_eaten_reward = self.get_eaten_reward(nextState)
        reward += get_eaten_reward
        # Add the reward for eating an enemy invader
        reward += self.get_eating_enemy_reward(game_state, nextState)

        # Adjust the reward based on the game score
        score = nextState.data.score  # This retrieves the current score
        score_adjustment = -score / 10.0  # Example: scale down the score's impact
        reward += score_adjustment

        return reward

    def final(self, state):
        # Save weights to file
        CaptureAgent.final(self, state)
        #with open('defensive_agent_weights.pkl', 'wb') as file:
        #    pickle.dump(self.weights, file)

        # Save cumulative reward to CSV
        if TRAINING_defensive:
            self.save_cumulative_reward()

        # Save weights to a specific directory with the training ID
        weights_path = f'training_results/defensive_agent_weights_{self.training_id}.json'
        with open(weights_path, 'w') as file:
            json.dump(self.weights, file)

        text_to_write = 'invaderDistance = {0}, food_defending = {1}, closest_entry_point = {2}'.format(self.weights['invaderDistance'],self.weights['food_defending'],self.weights['closest_entry_point'])
        with open('weights_tracking.txt', 'a') as file:
            # Write the string to the file
            file.write(text_to_write+'\n')