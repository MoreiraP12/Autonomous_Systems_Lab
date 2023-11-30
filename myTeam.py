# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import random
import util

import matplotlib.pyplot as plt
import numpy as np

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from matplotlib import pyplot as plt


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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
        
        self.walls = game_state.get_walls()

        self.legal_positions = game_state.get_walls().as_list(False)
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(game_state)}
        for enemy in self.get_opponents(game_state):
            self.obs[enemy][game_state.get_initial_agent_position(enemy)] = 1.0
    

    def elapse_time(self, enemy, gameState):
        # Define a lambda function to calculate possible next positions
        possible_positions = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        
        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over the previous positions and their probabilities
        for prev_pos, prev_prob in self.obs[enemy].items():
            # Calculate the new possible positions for the enemy
            new_obs = util.Counter({pos: 1.0 for pos in possible_positions(prev_pos[0], prev_pos[1]) if pos in self.legal_positions})
            
            # Normalize the new observation probabilities
            new_obs.normalize()

            # Update the overall belief distribution with the new probabilities
            for new_pos, new_prob in new_obs.items():
                all_obs[new_pos] += new_prob * prev_prob

        # Check for any food eaten by opponents
        foods = self.get_food_you_are_defending(gameState).as_list()
        prev_foods = self.get_food_you_are_defending(self.get_previous_observation()).as_list() if self.get_previous_observation() else []

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
        team_pos = [gameState.get_agent_position(team) for team in self.get_team(gameState)]
        
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
        #else:
            # If no valid observations, initialize the belief distribution
            # self.initialize(enemy, gameState)
    

    def get_enemy_position_img(self,enemy):
        # Get the belief distribution about the opponent's positions
        belief_distribution = self.obs[enemy]  # Assuming 'enemy' is defined somewhere in your code

        # Get the shape of the game board
        board_shape = self.walls.width, self.walls.height

        # Convert the belief distribution to a 2D numpy array
        belief_array = np.zeros(board_shape)
        for pos, prob in belief_distribution.items():
            belief_array[pos[0]][pos[1]] = prob

        return np.transpose(np.array(belief_array))
    
    def danger_zone_img(self, enemy1_index, enemy2_index, game_state):

        my_state = game_state.get_agent_state(self.index)
        scared = True if my_state.scared_timer > 5 else False

        enemy1_state = game_state.get_agent_state(enemy1_index)
        enemy2_state = game_state.get_agent_state(enemy2_index)
        enemiesScared = True if (enemy1_state.scared_timer > 5 or enemy2_state.scared_timer > 5) else False

        # Get the shape of the game board
        board_shape = self.walls.width, self.walls.height
        
        is_red = False if 1 in self.get_team(game_state) else True

        if scared:
            return [[1 for _ in range(board_shape[1])] for _ in range(board_shape[0])]
        elif enemiesScared: 
            return [[0 for _ in range(board_shape[1])] for _ in range(board_shape[0])]
        else:
            width_of_zeros = int(board_shape[1] * 0.5)

            if is_red:
                matrix = [[0] * width_of_zeros + [1] * (board_shape[1] - width_of_zeros) for _ in range(len(board_shape))]
            else:
                matrix = [[0] * (board_shape[1] - width_of_zeros) + [1] * width_of_zeros for _ in range(len(board_shape))]

            return matrix

    

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
        
        enemy1 = 1
        enemy2 = 3

        #Layer 1: Bayesian Inference over Enemey Location
        self.observe(enemy1, game_state)
        self.observe(enemy2, game_state)
        self.elapse_time(enemy1, game_state)
        self.elapse_time(enemy2, game_state)
        if enemy1!=1 or enemy2!=3:
            print('here')
        enemy_1_pos_img = np.uint8(np.array(self.get_enemy_position_img(enemy1))*255)
        enemy_1_pos_img = enemy_1_pos_img[::-1]#The rows are reversed
        time_left = game_state.data.timeleft
        plt.imsave(f'agents\custom\images\{str(time_left)}enemy_1_pos_img.png', enemy_1_pos_img, cmap='gray')
        enemy_2_pos_img = np.uint8(np.array(self.get_enemy_position_img(enemy2))*255)[::-1]#The rows are reversed
        plt.imsave(f'agents\custom\images\{str(time_left)}enemy_2_pos_img.png', enemy_2_pos_img, cmap='gray')
        
        #Layer 2: Danger Zone based on scared
        danger_zone_img = np.array(self.danger_zone_img(enemy1, enemy2, game_state))#transpose because it comes in a col,row form
        plt.imsave(f'agents\custom\images\{str(time_left)}danger_zone.png', danger_zone_img, cmap='gray')
        
        #Layer 3: Food
        food =  self.get_food_you_are_defending(game_state).as_list()
        food_img = np.zeros_like(enemy_1_pos_img).astype(np.uint8)
        for col, row in food:
            food_img[row, col] = 1
        food_img = food_img[::-1]#The rows are reversed
        plt.imsave(f'agents\custom\images\{str(time_left)}Food.png', food_img, cmap='gray')

        #Layer 4: Enemy food
        food_enemy =  self.get_food(game_state).as_list()
        food_enemy_img = np.zeros_like(enemy_1_pos_img).astype(np.uint8)
        for col, row in food_enemy:
            food_enemy_img[row, col] = 1
        food_enemy_img = food_enemy_img[::-1]
        plt.imsave(f'agents\custom\images\{str(time_left)}Food_enemy.png', food_enemy_img, cmap='gray')

        #Layer 5: Agent Position
        agent_pos =  game_state.get_agent_position(self.index)
        agent_pos_img = np.zeros_like(enemy_1_pos_img).astype(np.uint8)
        agent_pos_img[agent_pos[1],agent_pos[0]] = 1
        agent_pos_img = agent_pos_img[::-1]
        plt.imsave(f'agents\custom\images\{str(time_left)}Agent_pos.png', agent_pos_img, cmap='gray')
        
        #Layer 6: Walls
        walls_img =  np.array(game_state.get_walls().data).T.astype(np.uint8)[::-1]
        # Save using matplotlib
        plt.imsave(f'agents\custom\images\{str(time_left)}walls.png', walls_img, cmap='gray')
        
        #Layer 7: Capsules
        capsules = self.get_capsules(game_state)
        capsules_img = np.zeros_like(enemy_1_pos_img).astype(np.uint8)
        for col, row in capsules:
            capsules_img[row, col] = 1
        capsules_img = capsules_img[::-1]
        plt.imsave(f'agents\custom\images\{str(time_left)}capsules_img.png', capsules_img, cmap='gray')

        #Layer 8: Capsules the enemy is trying to get
        capsules_defending =  self.get_capsules_you_are_defending(game_state)
        capsules_defending_img = np.zeros_like(enemy_1_pos_img).astype(np.uint8)
        for col, row in capsules_defending:
            capsules_defending_img[row, col] = 1
        capsules_defending_img = capsules_defending_img[::-1]
        plt.imsave(f'agents\custom\images\{str(time_left)}capsules_defending_img.png', capsules_defending_img, cmap='gray')

        #Get the enemies location
        most_prob_e1_loc = np.unravel_index(np.argmax(enemy_1_pos_img), enemy_1_pos_img.shape)
        most_prob_e2_loc = np.unravel_index(np.argmax(enemy_2_pos_img), enemy_2_pos_img.shape)
        distances_e_2_myfood = []
        distances_e_2_myfood = []

        for col, row in food:
            # Append tuple of (distance, (row, col)) for each food location
            distances_e_2_myfood.append((util.manhattanDistance(most_prob_e1_loc, (row, col)), (row, col)))
            distances_e_2_myfood.append((util.manhattanDistance(most_prob_e2_loc, (row, col)), (row, col)))

        # Find the tuple with the minimum distance
        min_distance_tuple = min(distances_e_2_myfood, key=lambda x: x[0])

        # Extract the minimum distance and the corresponding point of the enemies to my food
        min_e_distance = min_distance_tuple[0]
        min_distance_tgt_point = min_distance_tuple[1]

        #get the agent location in the map
        agent_pos_map_layer = np.unravel_index(np.argmax(agent_pos_img), agent_pos_img.shape)
        #get the distance of my agent from the target point of my enemy
        my_dist_to_e_tgt = util.manhattanDistance(agent_pos_map_layer, min_distance_tgt_point)

        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        print("REFLEX ##################")

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
    def get_successors(self, position):
        """Get successor positions from the given position."""
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = 0, 0
            if action == Directions.NORTH:
                dy = 1
            elif action == Directions.SOUTH:
                dy = -1
            elif action == Directions.EAST:
                dx = 1
            elif action == Directions.WEST:
                dx = -1

            next_position = (x + dx, y + dy)
            if not self.walls[next_position[0]][next_position[1]]:  # Check for walls
                successors.append(next_position)
        return successors
    def calculate_min_distance_target_point(self, game_state):
        # Compute the most probable enemy locations
        enemy1, enemy2 = self.get_opponents(game_state)
        enemy_1_pos_img = self.get_enemy_position_img(enemy1)
        enemy_2_pos_img = self.get_enemy_position_img(enemy2)
        most_prob_e1_loc = np.unravel_index(np.argmax(enemy_1_pos_img), enemy_1_pos_img.shape)
        most_prob_e2_loc = np.unravel_index(np.argmax(enemy_2_pos_img), enemy_2_pos_img.shape)

        # Compute the closest food point to the enemies
        food = self.get_food_you_are_defending(game_state).as_list()
        distances_e_2_myfood = []
        for col, row in food:
            distances_e_2_myfood.append((util.manhattanDistance(most_prob_e1_loc, (row, col)), (row, col)))
            distances_e_2_myfood.append((util.manhattanDistance(most_prob_e2_loc, (row, col)), (row, col)))

        min_distance_tuple = min(distances_e_2_myfood, key=lambda x: x[0])
        return min_distance_tuple[1]
    def a_star_search(self, start, goal):
        """Perform A* search from start to goal."""

        def heuristic(a, b):
            """Manhattan distance heuristic."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        frontier = util.PriorityQueue()
        frontier.push(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.isEmpty():
            current = frontier.pop()

            if current == goal:
                break

            for next in self.get_successors(current):
                new_cost = cost_so_far[current] + 1  # Assuming uniform cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    frontier.push(next, priority)
                    came_from[next] = current

        # Reconstruct path
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


    
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def choose_action(self, game_state):
        print("OFFENSIVE ##################")

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

    def choose_action(self, game_state):
        # Calculate the target point
        target = self.calculate_min_distance_target_point(game_state)

        # A* pathfinding towards the target
        my_pos = game_state.get_agent_position(self.index)
        path_to_target = self.a_star_search(my_pos, target)

        # Choose the next step on the path
        if path_to_target:
            next_step = path_to_target[0]
            return self.get_action_from_position(my_pos, next_step)
        else:
            return random.choice(game_state.get_legal_actions(self.index))

    def get_action_from_position(self, current_position, next_position):
        """Get the action to move from current_position to next_position."""
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]

        if dx == 1 and dy == 0:
            return Directions.EAST
        elif dx == -1 and dy == 0:
            return Directions.WEST
        elif dx == 0 and dy == 1:
            return Directions.NORTH
        elif dx == 0 and dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP

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