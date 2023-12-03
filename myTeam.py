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
                first='ApproxQLearningOffense', second='DefensiveReflexAgent', **args):
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


class ApproxQLearningOffense(CaptureAgent):

    def register_initial_state(self, gameState):
        if TRAINING:
            self.epsilon = 0.1
        else: 
            self.epsilon = 0
        
        self.alpha = 0.2
        self.discount = 0.8

        # try:
        #     with open('./trained_agent_weights.pkl', 'rb') as file:
        #         self.weights = pickle.load(file)
        #         print(self.weights)
          
        # except (FileNotFoundError, IOError):
        self.weights = {
                        'closest-food': -3.67136772,
                        'bias': -9.2819232,
                        '#-of-ghosts-3-step-away': -3.12376152,
                        'eats-food': 15.81648863,
                        #'carrying_food_go_home': 30
                        }

        self.start = gameState.get_agent_position(self.index)
        self.features_extractor = features_extractor(self)
        CaptureAgent.register_initial_state(self, gameState)
        self.walls = gameState.get_walls()

        self.legal_positions = gameState.get_walls().as_list(False)
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(gameState)}
        for enemy in self.get_opponents(gameState):
            self.obs[enemy][gameState.get_initial_agent_position(enemy)] = 1.0

        self.total_initial_food = len(self.get_food(gameState).as_list())

    def elapse_time(self, enemy, gameState):

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
        foods = self.get_food_you_are_defending(gameState).as_list()
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()).as_list() if self.get_previous_observation() else []

        # If the number of foods has decreased, adjust the belief distribution
        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(gameState))

        # Update the agent's belief about the opponent's positions
        self.obs[enemy] = all_obs

    def observe(self, enemy, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.
        """
        # Get distance observations for all agents
        all_noise = gameState.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = gameState.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(gameState.teams) if value]
        team_pos = [gameState.get_agent_position(team) for team in team_idx]

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
                pos_prob = gameState.get_distance_prob(true_distance, noisy_distance)

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
        # self.initialize(enemy, gameState)



    def choose_action(self, gameState):
        """
            Picks among the actions with the highest Q(s,a).
        """
        legal_actions = gameState.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None

        #If you ate enough food to win the best action is always returning home (no worries about enemies)
        food_left = len(self.get_food(gameState).as_list())
        if food_left <= 2:
            best_dist = 10000
            for action in legal_actions:
                successor = self.get_successor(gameState, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    bestAction = action
                    best_dist = dist
            return bestAction
        
        original_agent_state = gameState.get_agent_state(self.index)
        food_carrying = original_agent_state.num_carrying 

        if food_carrying > 3:
            best_dist = 10000
            for action in legal_actions:
                successor = self.get_successor(gameState, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    bestAction = action
                    best_dist = dist
            return bestAction
        
        action = None
        if TRAINING:
            for action in legal_actions:
                self.updateWeights(gameState, action)
        if not util.flipCoin(self.epsilon):
            # exploit
            action = self.compute_action_from_q_values(gameState)
        else:
            # explore
            action = random.choice(legal_actions)
        return action

    def get_q_value(self, gameState, action):
        # features vector
        features = self.features_extractor.get_features(gameState, action)
        print("======================")
        print(self.weights)
        print(features)
        print(features * self.weights)

        return features * self.weights

    def update(self, gameState, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.features_extractor.get_features(gameState, action)
        oldValue = self.get_q_value(gameState, action)
        futureQValue = self.get_value(nextState)
        difference = (reward + self.discount * futureQValue) - oldValue
        # for each feature i
        for feature in features:
            if feature not in self.weights:
                self.weights[feature] = 0  # Initialize with a default value, like 0
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight
        # print(self.weights)

    def updateWeights(self, gameState, action):
        nextState = self.get_successor(gameState, action)
        reward = self.getReward(gameState, nextState)
        self.update(gameState, action, nextState, reward)
    



    def getReward(self, gameState, nextState):
        agentPosition = gameState.get_agent_position(self.index)
        midpoint_x = gameState.get_walls().width // 2

        # Adjust border_x based on the team
        if self.red:  # For the red team
            border_x = midpoint_x - 1
        else:  # For the blue team
            border_x = midpoint_x

        # Ensure border_x is within valid range
        border_x = max(0, min(border_x, gameState.get_walls().width - 1))

        # Calculate the center y-coordinate of the map
        center_y = gameState.get_walls().height // 2

        #go near home if you're carrying food
        original_agent_state = gameState.get_agent_state(self.index)
        go_home = (original_agent_state.num_carrying*3) * (10-self.get_maze_distance(agentPosition, (border_x, center_y)))

            
        score = 0
        if self.get_score(nextState) > self.get_score(gameState):
            diff = self.get_score(nextState) - self.get_score(gameState)
            if self.red:  # For the red team
                score += diff * 20
            else:  # For the blue team
                score += -diff * 20
        

        distToFood_reward = 0
        # check if food eaten in nextState
        myFoods = self.get_food(gameState).as_list()
        distToFood = min([self.get_maze_distance(agentPosition, food) for food in myFoods])
        # I am 1 step away, will I be able to eat it?
        if distToFood == 1:
            nextFoods = self.get_food(nextState).as_list()
            if len(myFoods) - len(nextFoods) == 1:
                distToFood_reward += 20

        enemies_reward = 0
        # check if I am eaten
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        if len(ghosts) > 0:
            minDistGhost = min([self.get_maze_distance(agentPosition, g.get_position()) for g in ghosts])
            if minDistGhost == 1:
                nextPos = nextState.get_agent_state(self.index).get_position()
                if nextPos == self.start:
                    # I die in the next state
                    enemies_reward = -500
        
        # check if an enemy is close
        #numGhostsInProximity = sum(self.isGhostWithinSteps((next_x, next_y), g, 3, walls) for g in ghosts)

        print({"enemies":enemies_reward, "go_home": go_home, "distToFood_reward": distToFood_reward, "score": score})
        return enemies_reward + go_home + distToFood_reward + score

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        with open('trained_agent_weights.pkl', 'wb') as file:
            pickle.dump(self.weights, file)
        # did we finish training?

    def get_successor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def compute_value_from_q_values(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        allowedActions = gameState.get_legal_actions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.compute_action_from_q_values(gameState)
        return self.get_q_value(gameState, bestAction)

    def compute_action_from_q_values(self, gameState):
        
        legal_actions = gameState.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None
        
        actionVals = {}
        bestQValue = float('-inf')
        for action in legal_actions:
            target_q_value = self.get_q_value(gameState, action)
            actionVals[action] = target_q_value
            if target_q_value > bestQValue:
                bestQValue = target_q_value
        bestActions = [k for k, v in actionVals.items() if v == bestQValue]
        # random tie-breaking
        return random.choice(bestActions)


    def get_value(self, gameState):
        return self.compute_value_from_q_values(gameState)


class features_extractor:

    def __init__(self, agentInstance):
        self.agentInstance = agentInstance

    def isGhostWithinSteps(self, agentPos, ghostPos, steps, walls):
        # This function checks if a ghost is within 'steps' distance from the agent
        distance = self.agentInstance.get_maze_distance(agentPos, ghostPos)
        return distance <= steps

    def get_features(self, gameState, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.agentInstance.get_food(gameState)
        walls = gameState.get_walls()
        enemies = [gameState.get_agent_state(i) for i in self.agentInstance.get_opponents(gameState)]
        enemies_idx = [i for i in self.agentInstance.get_opponents(gameState)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None]

        # Here is where the probability of seeing a ghost in blind locations comes into play
        # The agent will always aproximate the ghost location, he will never going to be blind
        max_vals = list()
        if len(ghosts) == 0:
            for e_idx in enemies_idx:
                self.agentInstance.observe(e_idx, gameState)
                self.agentInstance.elapse_time(e_idx, gameState)
                belief_dist_e = self.agentInstance.obs[e_idx]
                max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                max_vals.append(max_position)
            ghosts = list(set(max_vals))


        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        agentPosition = gameState.get_agent_position(self.agentInstance.index)
        x, y = agentPosition
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        #TO DO: only worry about ghosts if scarred

        numGhostsInProximity = sum(self.isGhostWithinSteps((next_x, next_y), g, 3, walls) for g in ghosts)
        #numGhostsInProximity = sum((x, y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)
        features["#-of-ghosts-3-step-away"] = numGhostsInProximity

    
        if not features["#-of-ghosts-3-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.closest_food((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)


        #carrying to much food should go home
        agentPosition = gameState.get_agent_position(self.agentInstance.index)
        midpoint_x = gameState.get_walls().width // 2

        # Adjust border_x based on the team
        if self.agentInstance.red:  # For the red team
            border_x = midpoint_x - 1
        else:  # For the blue team
            border_x = midpoint_x

        # Ensure border_x is within valid range
        border_x = max(0, min(border_x, gameState.get_walls().width - 1))

        # Calculate the center y-coordinate of the map
        center_y = gameState.get_walls().height // 2

        #go near home if you're carrying food
        original_agent_state = gameState.get_agent_state(self.agentInstance.index)

        desired_y = 0
        best_dist = 999999
        for i in range(1, gameState.get_walls().height-1):
            if (border_x,i) in self.agentInstance.legal_positions:
                dist = self.agentInstance.get_maze_distance(agentPosition, (border_x,i))
                if dist < best_dist:
                    desired_y = i

        (x_agent,y_agent) = agentPosition
        if x_agent > border_x-1:
            base_distance = best_dist
        else:
            base_distance = 0
        amount_of_food_carrying = original_agent_state.num_carrying
        
        features["carrying_food_go_home"] = (amount_of_food_carrying + 2*((100 -1 - base_distance)))/10
        print(agentPosition)
        print(base_distance)
        print(amount_of_food_carrying)
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



##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def register_initial_state(self, gameState):
        self.start = gameState.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, gameState)
        self.walls = gameState.get_walls()

        self.legal_positions = gameState.get_walls().as_list(False)
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(gameState)}
        for enemy in self.get_opponents(gameState):
            self.obs[enemy][gameState.get_initial_agent_position(enemy)] = 1.0

    def elapse_time(self, enemy, gameState):

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
        foods = self.get_food_you_are_defending(gameState).as_list()
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()).as_list() if self.get_previous_observation() else []

        # If the number of foods has decreased, adjust the belief distribution
        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(gameState))

        # Update the agent's belief about the opponent's positions
        self.obs[enemy] = all_obs

    def observe(self, enemy, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.
        """
        # Get distance observations for all agents
        all_noise = gameState.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = gameState.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(gameState.teams) if value]
        team_pos = [gameState.get_agent_position(team) for team in team_idx]

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
                pos_prob = gameState.get_distance_prob(true_distance, noisy_distance)

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
        # self.initialize(enemy, gameState)

    def choose_action(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        food_left = len(self.get_food(gameState).as_list())

        if food_left <= 2:
            best_dist = 9999
            for action in actions:
                successor = self.get_successor(gameState, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    bestAction = action
                    best_dist = dist
            return bestAction

        return random.choice(bestActions)

    def get_successor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def get_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(gameState, action)
        features['successorScore'] = self.get_score(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def choose_action(self, gameState):
        actions = gameState.get_legal_actions(self.index)
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]

        if len(invaders) == 0:
            # Use belief distribution to determine patrol actions when no visible invaders
            return self.patrol_based_on_belief(gameState, actions)
        else:
            # Existing logic for when invaders are detected
            return super().choose_action(gameState)

    def patrol_based_on_belief(self, gameState, actions):
        """
        Adjust patrol behavior based on belief distribution of invader locations,
        ensuring the agent stays on its side of the map.
        """
        myPos = gameState.get_agent_position(self.index)
        best_action = None
        min_dist = float('inf')

        # Border x-coordinate for dividing the map
        border_x = (gameState.get_walls().width // 2) - 1 if self.red else (gameState.get_walls().width // 2)

        # Identify the most probable invader location based on belief distribution
        most_probable_invader_loc = None
        highest_prob = 0.0
        for enemy in self.get_opponents(gameState):
            for pos, prob in self.obs[enemy].items():
                if prob > highest_prob and not gameState.has_wall(*pos):
                    # Ensure the position is on your side of the map
                    if (self.red and pos[0] <= border_x) or (not self.red and pos[0] >= border_x):
                        highest_prob = prob
                        most_probable_invader_loc = pos

        # If a probable invader location is identified on your side, move towards it
        if most_probable_invader_loc:
            for action in actions:
                successor = self.get_successor(gameState, action)
                nextPos = successor.get_agent_state(self.index).get_position()
                # Ensure the agent doesn't cross into the opposing side
                if (self.red and nextPos[0] <= border_x) or (not self.red and nextPos[0] >= border_x):
                    dist = self.get_maze_distance(nextPos, most_probable_invader_loc)
                    if dist < min_dist:
                        best_action = action
                        min_dist = dist
        else:
            # Default to standard patrol behavior if no probable location is identified
            return self.patrol_border(gameState, actions)

        return best_action if best_action is not None else random.choice(actions)

    def register_initial_state(self, gameState):
        super().register_initial_state(gameState)
        self.patrol_points = self.get_patrol_points(gameState)
        self.current_patrol_point = 0  # Index of the current patrol point

    def get_patrol_points(self, gameState):
        """
        Generate a list of strategic patrol points along the border, focusing on chokepoints.
        """
        # Calculate the x-coordinate for the patrol area
        border_x = (gameState.get_walls().width // 2) - 1
        if not self.red:
            border_x += 1  # Adjust for blue team

        # Adjust x-coordinate to stay within safe distance from the border
        patrol_x = border_x - 1 if self.red else border_x + 1

        # Create patrol points focusing on chokepoints
        points = self.identify_chokepoints(gameState, patrol_x)
        return points

    def identify_chokepoints(self, gameState, patrol_x):
        """
        Analyze the map layout to find chokepoints for patrolling.
        """
        points = []
        wall_matrix = gameState.get_walls()
        height = wall_matrix.height
        width = wall_matrix.width

        # Identify tiles that have gaps in the walls along the border
        if self.red:
            for y in range(1, height - 1):
                if not wall_matrix[patrol_x][y]:
                    if not wall_matrix[patrol_x + 1][y]:
                        points.append((patrol_x, y))
        else:
            for y in range(1, height - 1):
                if not wall_matrix[patrol_x][y]:
                    if not wall_matrix[patrol_x - 1][y]:
                        points.append((patrol_x, y))

        return points

    def patrol_border(self, gameState, actions):
        """
        Move towards the current patrol point, and update to the next point as needed.
        """
        myPos = gameState.get_agent_position(self.index)
        patrol_point = self.patrol_points[self.current_patrol_point]

        # Check if reached the current patrol point
        if myPos == patrol_point:
            self.current_patrol_point = (self.current_patrol_point + 1) % len(self.patrol_points)
            patrol_point = self.patrol_points[self.current_patrol_point]

        # Choose action to move towards the patrol point
        best_action = None
        min_dist = float('inf')
        for action in actions:
            successor = self.get_successor(gameState, action)
            nextPos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(nextPos, patrol_point)
            if dist < min_dist:
                best_action = action
                min_dist = dist

        return best_action if best_action is not None else random.choice(actions)

    def get_features(self, gameState, action):
        features = util.Counter()
        successor = self.get_successor(gameState, action)

        myState = successor.get_agent_state(self.index)
        myPos = myState.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.is_pacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]

        enemies_idx = [i for i in self.get_opponents(successor) if successor.get_agent_state(i).is_pacman]
        # Bayes probability to see an enemy in further positions
        if len(enemies_idx) > 0:
            if len(invaders) > 0:
                dists = [self.get_maze_distance(myPos, a.get_position()) for a in invaders]
                features['invaderDistance'] = min(dists)
                features['numInvaders'] = len(invaders)
            else:
                dists = []
                for e_idx in enemies_idx:
                    self.observe(e_idx, gameState)
                    self.elapse_time(e_idx, gameState)
                    belief_dist_e = self.obs[e_idx]
                    max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                    dists.append(self.get_maze_distance(myPos, max_position))
                features['invaderDistance'] = min(dists)
                features['numInvaders'] = len(enemies_idx)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}