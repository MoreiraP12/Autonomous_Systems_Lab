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

#################
# Team creation #
#################

TRAINING = False
def create_team(firstIndex, secondIndex, isRed,
                first='OffensiveQLearningAgent', second='DefensiveReflexCaptureAgent', **args):
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
    def load_weights(self):
        '''
        Load the trained weights from a file, or provide default weights if the file is not found.

        Returns:
            dict: A dictionary containing the loaded or default weights.
        '''
        # Initialize weights to None
        weights = None
        
        # try:
        #     # Try to open the file and load weights from it
        #     with open('./trained_agent_weights.pkl', 'rb') as file:
        #         weights = pickle.load(file)
            
        # except (FileNotFoundError, IOError):
            # If the file is not found or an error occurs, provide default weights
        weights = {
                    'bias':-9.1234412,
                    'food_close': -2.983928392083,
                    'ghosts_close': -3.65065432233,
                    'food_eaten': 15.12232122121,
                    'carrying_food_go_home': 1.822389123231
                }
        
        return weights

    
    def register_initial_state(self, game_state):
        #Important variables related to the Q learning algorithm
        #When playing we don't want any exploration, strictly on policy
        if TRAINING: self.epsilon = 0.15
        else: 
            self.epsilon = 0
        self.alpha = 0.2
        self.discount = 0.8
        self.weights = self.load_weights()

        
        self.initial_position = game_state.get_agent_position(self.index)
        self.legal_positions = game_state.get_walls().as_list(False)

        CaptureAgent.register_initial_state(self, game_state)

        #Initialize the Bayesian Inference for the ghost positions
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
        if TRAINING:
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

        return amount_of_food_carrying / -((game_state.get_walls().width / 3) -  self.get_maze_distance(self.initial_position, (next_x, next_y)))

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
        features["carrying_food_go_home"] = self.calculate_carrying_food_go_home_feature(game_state, agent_position, action)

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
        #print("features: ", features, "value: ", features * self.weights)
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
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight
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
        rewards = {"enemies": enemies_reward, "go_home": go_home_reward, "dist_to_food_reward": dist_to_food_reward, "score": score_reward}
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

        return amount_of_food_carrying / -((nextState.get_walls().width / 3) -  self.get_maze_distance(self.initial_position, agent_position))




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
        
    def final(self, state):
        CaptureAgent.final(self, state)
        with open('trained_agent_weights.pkl', 'wb') as file:
            pickle.dump(self.weights, file)

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
        print("=============================")
        for action in legal_actions:
            print("Action: ", action)
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


class DefensiveReflexCaptureAgent(ReflexCaptureBaseAgent):
    """
    The DefensiveReflexCaptureAgent is responsible for defending the team's side of the map in the Pac-Man game.
    It uses a combination of strategies to defend against enemy Pac-Man invaders.

    More specifically:
    1. Patrol Behavior: The agent patrols strategic points along the border of its territory, with a focus on chokepoints.
       This ensures that it covers critical areas and intercepts invaders.

    2. Belief Distribution: When no visible invaders are present, the agent uses a belief distribution to estimate
       the potential locations of enemy Pac-Man agents. It adjusts its patrol behavior based on these estimations to
       maintain effective defense. This Bayesian inference approach allows the agent to make probabilistic estimates
       about where the enemy Pac-Man agents might be located. So its enables the agent to make informed decisions about
       its defensive actions, even when it cannot directly see the enemy Pac-Man invaders.

    3. Chokepoint Identification: The agent analyzes the map layout to identify chokepoints, which are key areas for
       interception. It strategically positions itself near these chokepoints to increase its defensive capabilities.

    4. Feature-Based Decision Making: The agent makes decisions based on a set of features computed from the game state.
       These features include information about the presence of invaders, distances to invaders, and more.

    5. Weighted Features: The agent assigns weights to these features to prioritize certain aspects of its decision-making.
       For example, it may prioritize staying on defense, targeting nearby invaders, and avoiding stopping or reversing
       whenever possible.

    Overall, the DefensiveReflexCaptureAgent is designed to be a formidable defender, ensuring that the team's territory
    remains secure by intercepting and repelling enemy Pac-Man invaders.
    """

    def register_initial_state(self, game_state):
        """
        Initializes the agent's state and patrol points.

        Args:
            game_state (GameState): The current game state.
        """
        # Initialize the agent's state
        super().register_initial_state(game_state)

        # Get the patrol points for the agent
        self.patrol_points = self.get_patrol_points(game_state)
        self.current_patrol_point = 0  # Index of the current patrol point

    def get_patrol_points(self, game_state):
        """
        This method calculates a list of patrol points that the agent should visit to effectively defend its side of the
        map. It first determines the x-coordinate of the border, considering whether the agent is on the red or blue
        team. It then adjusts this coordinate to ensure a safe distance from the border. Finally, it calls
        'identify_chokepoints' to create a list of patrol points that focus on chokepoints, which are key areas where
        enemy invaders are likely to pass through.

        Args:
            game_state (GameState): The current game state.

        Returns:
            list: List of patrol points.
        """
        # Calculate the x-coordinate for the patrol area
        border_x = (game_state.get_walls().width // 2) - 1
        if not self.red:
            border_x += 1  # Adjust for blue team

        # Adjust x-coordinate to stay within safe distance from the border
        patrol_x = border_x - 1 if self.red else border_x + 1

        # Create patrol points focusing on chokepoints
        points = self.identify_chokepoints(game_state, patrol_x)
        return points

    def identify_chokepoints(self, game_state, patrol_x):
        """
        This method analyzes the layout of the game map to identify chokepoints that are strategically important for
        patrolling. Chokepoints are locations where gaps exist in the walls along the border. Depending on whether
        the agent is on the red or blue team, it searches for these gaps in the walls and records the positions of
        chokepoints in the form of (x, y) coordinates. These chokepoints are critical for effective defensive patrol.

        Args:
            game_state (GameState): The current game state.
            patrol_x (int): The x-coordinate of the patrol area.

        Returns:
            list: List of identified chokepoints.
        """
        # Initialize a list to store the identified chokepoints
        points = []

        # Get the height and width of the game map
        wall_matrix = game_state.get_walls()
        height = wall_matrix.height
        width = wall_matrix.width

        # Identify tiles that have gaps in the walls along the border
        if self.red:
            # If the agent is on the red team, search for gaps on the left side of the map
            for y in range(1, height - 1):
                if not wall_matrix[patrol_x][y]:
                    if not wall_matrix[patrol_x + 1][y]:
                        points.append((patrol_x, y))
        else:
            # If the agent is on the blue team, search for gaps on the right side of the map
            for y in range(1, height - 1):
                if not wall_matrix[patrol_x][y]:
                    if not wall_matrix[patrol_x - 1][y]:
                        points.append((patrol_x, y))

        return points

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a) for the agent.

        Args:
            game_state (GameState): The current game state.

        Returns:
            str: The chosen action.
        """
        # Get the legal actions the agent can take
        actions = game_state.get_legal_actions(self.index)

        # Get information about enemy agents
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        # Identify visible invaders
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]

        if len(invaders) == 0:
            # When no visible invaders are present, use belief distribution for patrol actions
            return self.patrol_based_on_belief(game_state, actions)
        else:
            # Use existing logic for when invaders are detected
            return super().choose_action(game_state)

    def patrol_based_on_belief(self, game_state, actions):
        """
        Adjust patrol behavior based on belief distribution of invader locations,ensuring the agent stays on its side of
        the map.

        Args:
            game_state (GameState): The current game state.
            actions (list): List of legal actions the agent can take.

        Returns:
            str: The chosen action for patrolling.
        """
        # Get the agent's current position
        myPos = game_state.get_agent_position(self.index)

        # Initialize variables for tracking the best action and its distance
        best_action = None
        min_dist = float('inf')

        # Determine the x-coordinate of the border that divides the map
        border_x = (game_state.get_walls().width // 2) - 1 if self.red else (game_state.get_walls().width // 2)

        # Identify the most probable invader location based on belief distribution
        most_probable_invader_loc = None
        highest_prob = 0.0
        for enemy in self.get_opponents(game_state):
            for pos, prob in self.obs[enemy].items():
                if prob > highest_prob and not game_state.has_wall(*pos):
                    # Ensure the position is on your side of the map
                    if (self.red and pos[0] <= border_x) or (not self.red and pos[0] >= border_x):
                        highest_prob = prob
                        most_probable_invader_loc = pos

        # If a probable invader location is identified on the agent's side, move towards it
        if most_probable_invader_loc:
            for action in actions:
                successor = self.get_successor(game_state, action)
                nextPos = successor.get_agent_state(self.index).get_position()
                # Ensure the agent doesn't cross into the opposing side
                if (self.red and nextPos[0] <= border_x) or (not self.red and nextPos[0] >= border_x):
                    dist = self.get_maze_distance(nextPos, most_probable_invader_loc)
                    if dist < min_dist:
                        best_action = action
                        min_dist = dist
        else:
            # Default to standard patrol behavior if no probable location is identified
            return self.patrol_border(game_state, actions)

        return best_action if best_action is not None else random.choice(actions)

    def patrol_border(self, game_state, actions):
        """
        Move towards the current patrol point, and update to the next point as needed.

        Args:
            game_state (GameState): The current game state.
            actions (list): List of legal actions the agent can take.

        Returns:
            str: The chosen action for patrolling.
        """
        # Get the agent's current position
        myPos = game_state.get_agent_position(self.index)

        # Get the current patrol point
        patrol_point = self.patrol_points[self.current_patrol_point]

        # Check if reached the current patrol point
        if myPos == patrol_point:
            # Update to the next patrol point in the list, looping back if necessary
            self.current_patrol_point = (self.current_patrol_point + 1) % len(self.patrol_points)
            patrol_point = self.patrol_points[self.current_patrol_point]

        # Choose an action to move towards the patrol point
        best_action = None
        min_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            nextPos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(nextPos, patrol_point)
            if dist < min_dist:
                best_action = action
                min_dist = dist

        # Return the chosen action for patrolling
        return best_action if best_action is not None else random.choice(actions)

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

        # Get the successor state after taking the specified action
        successor = self.get_successor(game_state, action)

        # Get the agent's state in the successor state
        myState = successor.get_agent_state(self.index)
        myPos = myState.get_position()

        # Compute whether the agent is on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.is_pacman:
            features['onDefense'] = 0

        # Compute the distance to visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]

        enemies_idx = [i for i in self.get_opponents(successor) if successor.get_agent_state(i).is_pacman]

        # Calculate Bayesian probability to see an enemy in further positions
        if len(enemies_idx) > 0:
            if len(invaders) > 0:
                dists = [self.get_maze_distance(myPos, a.get_position()) for a in invaders]
                features['invaderDistance'] = min(dists)
                features['numInvaders'] = len(invaders)
            else:
                dists = []
                for e_idx in enemies_idx:
                    self.observe(e_idx, game_state)
                    self.elapse_time(e_idx, game_state)
                    belief_dist_e = self.obs[e_idx]
                    max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                    dists.append(self.get_maze_distance(myPos, max_position))
                features['invaderDistance'] = min(dists)
                features['numInvaders'] = len(enemies_idx)

        # Check if the action is STOP and set the 'stop' feature
        if action == Directions.STOP:
            features['stop'] = 1

        # Check if the action is a reverse action and set the 'reverse' feature
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, game_state, action):
        """
        Define and return a set of weights for the features to influence the agent's decision-making.

        Args:
            game_state (GameState): The current game state.
            action (str): The chosen action to take.

        Returns:
            dict: A dictionary of feature weights.
        """
        return {
            'numInvaders': -1000,  # Weight for the number of invaders (penalize more invaders)
            'onDefense': 100,  # Weight for being on defense (favor being on defense)
            'invaderDistance': -10,  # Weight for the distance to invaders (penalize longer distances)
            'stop': -100,  # Weight for choosing the STOP action (strongly penalize STOP)
            'reverse': -2  # Weight for choosing reverse actions (penalize reverse actions)
        }
