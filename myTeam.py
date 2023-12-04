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
    def load_weights(self):
        '''
        load the trained weights 
        '''
        weights = None
        try:
            with open('./trained_agent_weights.pkl', 'rb') as file:
                weights = pickle.load(file)
          
        except (FileNotFoundError, IOError):
                weights = {
                        'bias':  -10.23232322332,
                        'food_close': -2.983928392083,
                        'ghosts_close"': -34.9843372922,
                        'food_eaten': 12.12232122121,
                        'carrying_food_go_home': 1.02389123231
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
        best_dist = 10000
        for action in self.legal_actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.initial_position, pos2)
            if dist < best_dist:
                bestAction = action
                best_dist = dist
        return bestAction
    
    def choose_action(self, game_state):
        """
            Picks among the actions with the highest Q(s,a).
        """
        action = None
        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None

        #If you ate enough food to win the best action is always returning home (no worries about enemies)
        food_left = len(self.get_food(game_state).as_list())
        
        if food_left <= 2:
            return self.run_home_action(game_state)
        
        original_agent_state = game_state.get_agent_state(self.index)
        if original_agent_state.num_carrying  > 3:
            return self.run_home_action(game_state)
        
        if TRAINING:
            for action in legal_actions:
                self.update_weights(game_state, action)
        if not util.flipCoin(self.epsilon):
            # exploit
            action = self.compute_action_from_q_values(game_state)
        else:
            # explore
            action = random.choice(legal_actions)
        return action


    def is_ghost_within_steps(self, agentPos, ghostPos, steps, walls):
        # This function checks if a ghost is within 'steps' distance from the agent
        distance = self.get_maze_distance(agentPos, ghostPos)
        return distance <= steps
    
    def get_num_of_ghost_in_proximity(self, game_state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_idx = [i for i in self.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None]

        # Here is where the probability of seeing a ghost in blind locations comes into play
        # The agent will always aproximate the ghost location, he will never going to be blind
        max_vals = list()
        if len(ghosts) == 0:
            for e_idx in enemies_idx:
                self.observe(e_idx, game_state)
                self.elapse_time(e_idx, game_state)
                belief_dist_e = self.obs[e_idx]
                max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                max_vals.append(max_position)
            ghosts = list(set(max_vals))

        agentPosition = game_state.get_agent_position(self.index)
        x, y = agentPosition
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        return sum(self.is_ghost_within_steps((next_x, next_y), g, 3, walls) for g in ghosts)
    
    def calculate_carrying_food_go_home_feature(self, game_state, agent_position):
        midpoint_x = game_state.get_walls().width // 2

        # Adjust border_x based on the team
        border_x = midpoint_x - 1 if self.red else midpoint_x
        border_x = max(0, min(border_x, game_state.get_walls().width - 1))

        # Calculate the center y-coordinate of the map
        center_y = game_state.get_walls().height // 2

        # Go near home if carrying food
        original_agent_state = game_state.get_agent_state(self.index)

        desired_y, best_dist = self.find_desired_y(game_state, agent_position, border_x)
        x_agent, _ = agent_position
        base_distance = best_dist if x_agent > border_x - 1 else 0

        amount_of_food_carrying = original_agent_state.num_carrying
        carrying_food_go_home_feature = (amount_of_food_carrying + 2 * ((100 - 1 - base_distance))) / 10

        print(base_distance)
        print(amount_of_food_carrying)

        return carrying_food_go_home_feature

    def find_desired_y(self, game_state, agent_position, border_x):
        desired_y = 0
        best_dist = 999999
        for i in range(1, game_state.get_walls().height - 1):
            if (border_x, i) in self.legal_positions:
                dist = self.get_maze_distance(agent_position, (border_x, i))
                if dist < best_dist:
                    desired_y = i
        return desired_y, best_dist
    
    def get_features(self, game_state, action):
        features = util.Counter()

        # compute the location of pacman after he takes the action
        agent_position = game_state.get_agent_position(self.index)
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        features["bias"] = 1.0
        features["ghosts_close"] =  self.get_num_of_ghost_in_proximity(game_state, action)
        if features["ghosts_close"] == 0:
            features["food_eaten"] = 1.0

        dist = self.closest_food((next_x, next_y),  self.get_food(game_state), game_state.get_walls())
        if dist is not None:
            features["food_close"] = float(dist) / (game_state.get_walls().width * game_state.get_walls().height)

        features["carrying_food_go_home"] = self.calculate_carrying_food_go_home_feature(game_state, agent_position)

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
        # else:
        # If no valid observations, initialize the belief distribution
        # self.initialize(enemy, game_state)

    def get_q_value(self, game_state, action):
        # features vector
        features = self.get_features(game_state, action)
        print(features)
        print(self.weights)
        print(features * self.weights)

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
        nextState = self.get_successor(game_state, action)
        reward = self.get_reward(game_state, nextState)
        self.update(game_state, action, nextState, reward)
    

    def get_reward(self, game_state, nextState):
        agent_position = game_state.get_agent_position(self.index)
        go_home_reward = self.calculate_go_home_reward(game_state, agent_position)
        score_reward = self.calculate_score_reward(game_state, nextState)
        dist_to_food_reward = self.calculate_dist_to_food_reward(game_state, nextState, agent_position)
        enemies_reward = self.calculate_enemies_reward(game_state, nextState, agent_position)

        rewards = {"enemies": enemies_reward, "go_home": go_home_reward, "dist_to_food_reward": dist_to_food_reward, "score": score_reward}
        print(rewards)
        return sum(rewards.values())

    def calculate_go_home_reward(self, game_state, agent_position):
        midpoint_x = game_state.get_walls().width // 2
        border_x = midpoint_x - 1 if self.red else midpoint_x
        border_x = max(0, min(border_x, game_state.get_walls().width - 1))
        center_y = game_state.get_walls().height // 2
        original_agent_state = game_state.get_agent_state(self.index)
        return (original_agent_state.num_carrying * 3) * (10 - self.get_maze_distance(agent_position, (border_x, center_y)))

    def calculate_score_reward(self, game_state, nextState):
        score_reward = 0
        if self.get_score(nextState) > self.get_score(game_state):
            diff = self.get_score(nextState) - self.get_score(game_state)
            score_reward += diff * 20 if self.red else -diff * 20
        return score_reward

    def calculate_dist_to_food_reward(self, game_state, nextState, agent_position):
        dist_to_food_reward = 0
        my_foods = self.get_food(game_state).as_list()
        dist_to_food = min([self.get_maze_distance(agent_position, food) for food in my_foods])

        if dist_to_food == 1:
            next_foods = self.get_food(nextState).as_list()
            if len(my_foods) - len(next_foods) == 1:
                dist_to_food_reward += 20

        return dist_to_food_reward

    def calculate_enemies_reward(self, game_state, nextState, agent_position):
        enemies_reward = 0
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        if len(ghosts) > 0:
            min_dist_ghost = min([self.get_maze_distance(agent_position, g.get_position()) for g in ghosts])
            if min_dist_ghost == 1:
                next_pos = nextState.get_agent_state(self.index).get_position()
                if next_pos == self.initial_position:
                    enemies_reward = -500

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
        for action in legal_actions:
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
