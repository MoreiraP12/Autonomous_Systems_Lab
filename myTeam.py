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
                first='ApproxQLearningOffense', second='DefensiveReflexAgent', numTraining=0, **args):
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
        self.epsilon = 0.3
        self.alpha = 0.2
        self.discount = 0.85
        self.episodesSoFar = 0

        try:
            with open('./trained_agent_weights.pkl', 'rb') as file:
                print("Opened File")
                self.weights = pickle.load(file)
                print(self.weights)
          
        except (FileNotFoundError, IOError):
            print("Didn't find file")
            self.weights = {'closest-food': -3.099192562140742,
                            'bias': -9.280875042529367,
                            '#-of-ghosts-3-step-away': -3,
                            'eats-food': 15.127808437648863,
                            'score': 20}

        self.start = gameState.get_agent_position(self.index)
        self.features_extractor = features_extractor(self)
        CaptureAgent.register_initial_state(self, gameState)
        self.walls = gameState.get_walls()

        self.legal_positions = gameState.get_walls().as_list(False)
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(gameState)}
        for enemy in self.get_opponents(gameState):
            self.obs[enemy][gameState.get_initial_agent_position(enemy)] = 1.0

        self.total_food = len(self.get_food(gameState).as_list())

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

    def get_enemy_position_img(self, enemy):
        # Get the belief distribution about the opponent's positions
        belief_distribution = self.obs[enemy]  # Assuming 'enemy' is defined somewhere in your code

        # Get the shape of the game board
        board_shape = self.walls.width, self.walls.height

        # Convert the belief distribution to a 2D numpy array
        belief_array = np.zeros(board_shape)
        for pos, prob in belief_distribution.items():
            belief_array[pos[0]][pos[1]] = prob

        return np.transpose(np.array(belief_array))

    def choose_action(self, gameState):
        """
            Picks among the actions with the highest Q(s,a).
        """
        legalActions = gameState.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None

        foodLeft = len(self.get_food(gameState).as_list())

        if foodLeft <= 2:
            bestDist = 9999
            for action in legalActions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        action = None
        if TRAINING:
            for action in legalActions:
                self.updateWeights(gameState, action)
        if not util.flipCoin(self.epsilon):
            # exploit
            action = self.getPolicy(gameState)
        else:
            # explore
            action = random.choice(legalActions)
        return action

    def getQValue(self, gameState, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # features vector
        features = self.features_extractor.getFeatures(gameState, action)
        return features * self.weights

    def update(self, gameState, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.features_extractor.getFeatures(gameState, action)
        oldValue = self.getQValue(gameState, action)
        futureQValue = self.getValue(nextState)
        difference = (reward + self.discount * futureQValue) - oldValue
        # for each feature i
        for feature in features:
            if feature not in self.weights:
                self.weights[feature] = 0  # Initialize with a default value, like 0
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight
        # print(self.weights)

    def updateWeights(self, gameState, action):
        nextState = self.getSuccessor(gameState, action)
        reward = self.getReward(gameState, nextState)
        self.update(gameState, action, nextState, reward)

    def getReward(self, gameState, nextState):
        reward = 0
        agentPosition = gameState.get_agent_position(self.index)

        if self.get_score(nextState) > self.get_score(gameState):
            diff = self.get_score(nextState) - self.get_score(gameState)
            if self.red:  # For the red team
                reward = diff * 20
            else:  # For the blue team
                reward = -diff * 20

        # check if food eaten in nextState
        myFoods = self.get_food(gameState).as_list()
        distToFood = min([self.get_maze_distance(agentPosition, food) for food in myFoods])
        # I am 1 step away, will I be able to eat it?
        if distToFood == 1:
            nextFoods = self.get_food(nextState).as_list()
            if len(myFoods) - len(nextFoods) == 1:
                reward = 20

        # check if I am eaten
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        if len(ghosts) > 0:
            minDistGhost = min([self.get_maze_distance(agentPosition, g.get_position()) for g in ghosts])
            if minDistGhost == 1:
                nextPos = nextState.get_agent_state(self.index).get_position()
                if nextPos == self.start:
                    # I die in the next state
                    reward = -50

        return reward

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        with open('trained_agent_weights.pkl', 'wb') as file:
            pickle.dump(self.weights, file)
        # did we finish training?

    def getSuccessor(self, gameState, action):
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

    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        allowedActions = gameState.get_legal_actions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.getPolicy(gameState)
        return self.getQValue(gameState, bestAction)

    def computeActionFromQValues(self, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = gameState.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None
        actionVals = {}
        bestQValue = float('-inf')
        for action in legalActions:
            targetQValue = self.getQValue(gameState, action)
            actionVals[action] = targetQValue
            if targetQValue > bestQValue:
                bestQValue = targetQValue
        bestActions = [k for k, v in actionVals.items() if v == bestQValue]
        # random tie-breaking
        return random.choice(bestActions)

    def getPolicy(self, gameState):
        return self.computeActionFromQValues(gameState)

    def getValue(self, gameState):
        return self.computeValueFromQValues(gameState)


class features_extractor:

    def __init__(self, agentInstance):
        self.agentInstance = agentInstance

    def isGhostWithinSteps(self, agentPos, ghostPos, steps, walls):
        # This function checks if a ghost is within 'steps' distance from the agent
        distance = self.agentInstance.get_maze_distance(agentPos, ghostPos)
        return distance <= steps

    def getFeatures(self, gameState, action):
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

        # count the number of ghosts 3-steps away
        numGhostsInProximity = sum(self.isGhostWithinSteps((next_x, next_y), g, 3, walls) for g in ghosts)
        #numGhostsInProximity = sum((x, y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)
        features["#-of-ghosts-3-step-away"] = numGhostsInProximity

    
        if not features["#-of-ghosts-3-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)


        features["score"] = self.agentInstance.get_score(gameState)
        #print(features)

        return features

    def closestFood(self, pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
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

        foodLeft = len(self.get_food(gameState).as_list())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
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
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights


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
            # Patrol near the center of the border when there are no invaders
            return self.patrol_border(gameState, actions)
        else:
            # Existing logic for when invaders are detected
            return super().choose_action(gameState)

    def patrol_border(self, gameState, actions):
        # Calculate the map's midpoint
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

        # Get the current position of the agent
        myPos = gameState.get_agent_position(self.index)

        # Check if the agent is at the center border position
        if myPos == (border_x, center_y):
            # If at the center, move one step north if possible, else move south
            if (border_x, center_y + 1) in actions and not gameState.has_wall(border_x, center_y + 1):
                return Directions.NORTH
            else:
                return Directions.SOUTH
        elif myPos == (border_x, center_y + 1):
            # If one step north of center, move back to center
            return Directions.SOUTH
        elif myPos == (border_x, center_y - 1):
            # If one step south of center, move back to center
            return Directions.NORTH
        else:
            # If not near the center, move towards the center
            best_action = None
            min_dist = float('inf')
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                nextPos = successor.get_agent_state(self.index).get_position()
                dist = self.get_maze_distance(nextPos, (border_x, center_y))
                if dist < min_dist:
                    best_action = action
                    min_dist = dist

            return best_action if best_action is not None else random.choice(actions)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

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