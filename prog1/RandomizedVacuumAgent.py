"""
RandomizedVacuumAgent.py
Generates environments for the vacuum agent world.
"""

import numpy as np
from copy import deepcopy
from enum import Enum
from collections import OrderedDict

__author__ = "Chris Campell"
__version__ = "9/3/2017"


class Environment(Enum):
    PRESET = 0
    RANDOM = 1


class Actions(Enum):
    MOVE_RIGHT = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_UP = 3
    SUCK = 4


class Agents(Enum):
    BLIND_AGENT = 0


class BlindAgent:
    environment = None
    location = (None, None)
    performance_score = None

    def __init__(self, environment, location=None):
        self.environment = environment
        if location is not None:
            self.location = location
        else:
            self.location = get_random_agent_position(world=self.environment.get_world())
        self.performance_score = 0

    def move_up(self):
        if self.environment.is_valid_action(action=Actions.MOVE_UP, location=self.location):
            self.location = (self.location[0]-1, self.location[1])
        self.performance_score -= 1

    def move_right(self):
        if self.environment.is_valid_action(action=Actions.MOVE_RIGHT, location=self.location):
            self.location = (self.location[0], self.location[1] + 1)
        self.performance_score -= 1

    def move_down(self):
        if self.environment.is_valid_action(action=Actions.MOVE_DOWN, location=self.location):
            self.location = (self.location[0] + 1, self.location[1])
        self.performance_score -= 1

    def move_left(self):
        if self.environment.is_valid_action(action=Actions.MOVE_LEFT, location=self.location):
            self.location = (self.location[0], self.location[1] - 1)
        self.performance_score -= 1

    def suck(self):
        if self.environment.is_dirty(location=self.location):
            self.environment.make_clean(location=self.location)
            print("BlindAgent-suck: The agent succeeded in cleaning (%d, %d)." % (self.location[0], self.location[1]))
            self.performance_score += 100
        else:
            print("BlindAgent-suck: Location (%d, %d) was already clean, no reward."
                  % (self.location[0], self.location[1]))


class ReflexAgent:
    environment = None
    location = (None, None)
    performance_score = None
    # at point (3,11) the new desired point is (2,3) Left-Upper Quad
    right_upper_quad_policy = [(1,8),(0,8),(0,9),(0,10),(0,11),(1,11),(1,10),(1,9),(2,9),(2,10),(2,11),(3,11),(2,3)]
    left_upper_quad_policy = [(1,3),(0,3),(0,2),(0,1),(0,0),(1,0),(1,1),(1,2),(2,2),(2,1),(2,0),(8,2)]
    lef_lower_quad_policy = [(8,2),(8,1),(8,0),(9,0),(9,1),(9,2),(10,2),(10,1),(10,0),(11,0),(11,1),(11,2),(8,9)]

    def __init__(self, environment, location):
        self.environment = environment
        self.location = location
        self.performance_score = 0

    def get_agent_quadrant_location(self):
        quadrant = None
        # Get which quadrant the agent is in:
        if self.location[0] < 12 and self.location[0] > 7:
            # The agent is in the bottom half of the map.
            if self.location[1] < 4 and self.location[1] >=0:
                # The agent is in the left-lower quadrant:
                quadrant = 'Left-Lower'
            elif self.location[1] < 12 and self.location[1] > 7:
                # The agent is in the right-lower quadrant:
                quadrant = 'Right-Lower'
            else:
                # The agent is not in a quadrant, but is in lower half of map.
                quadrant = 'NA-Lower'
        elif self.location[0] < 4 and self.location[0] >= 0:
            # The agent is in the upper half of the map.
            if self.location[1] < 4 and self.location[1] >=0:
                # The agent is in the left-upper quadrant:
                quadrant = 'Left-Upper'
            elif self.location[1] < 12 and self.location[1] > 7:
                # The agent is in the right-upper quadrant:
                quadrant = 'Right-Upper'
            else:
                # Not in a quadrant, is in upper half of map.
                quadrant = 'NA-Upper'
        else:
            print("Error, agent location not in any quadrant.")
        return quadrant

    def manhattan_distance(self, p_vec, q_vec):
        return np.sum(np.fabs(np.array(p_vec) - np.array(q_vec)))

    def get_policy_for_target_loc(self, loc):
        # Given the agents location and the desire to be at the provided loc, return the optimal move.
        # Get the agents possible moves:
        possible_moves = OrderedDict()
        if self.environment.is_valid_action(action=Actions.MOVE_UP, location=self.location):
            updated_loc = self.location
            updated_loc = (updated_loc[0] - 1, updated_loc[1])
            possible_moves[Actions.MOVE_UP] = updated_loc
        if self.environment.is_valid_action(action=Actions.MOVE_RIGHT, location=self.location):
            updated_loc = self.location
            updated_loc = (updated_loc[0], updated_loc[1] + 1)
            possible_moves[Actions.MOVE_RIGHT] = updated_loc
        if self.environment.is_valid_action(action=Actions.MOVE_LEFT, location=self.location):
            updated_loc = self.location
            updated_loc = (updated_loc[0], updated_loc[1] - 1)
            possible_moves[Actions.MOVE_LEFT] = updated_loc
        if self.environment.is_valid_action(action=Actions.MOVE_DOWN, location=self.location):
            updated_loc = self.location
            updated_loc = (updated_loc[0] + 1, updated_loc[1])
            possible_moves[Actions.MOVE_DOWN] = updated_loc
        man_dist = OrderedDict()
        for action, new_loc in possible_moves.items():
            man_dist[action] = self.manhattan_distance(new_loc, loc)
        # Return the minimum manhattan distance between the possible locations and the target.
        return min(man_dist, key=man_dist.get)

    def get_movement_policy(self):
        """
        get_movement_policy: Returns the policy (an action in every state) that the agent should follow to maximize rewards.
            The policy is determined only by the current percepts.
        :return:
        """
        policy = {}
         # Determine which quadrant the agent is in:
        agent_quadrant = self.get_agent_quadrant_location()
        if agent_quadrant == 'Right-Upper':
            desired_loc = (2,8)
            # move_policy = [(1,8),(0,8),(0,9),()]
        elif agent_quadrant == 'Left-Upper':
            desired_loc = (2,3)
        elif agent_quadrant == 'Right-Lower':
            desired_loc = (8,9)
        else:
            desired_loc = (8,2)

    def move_direction(self, policy=None):
        """
        move_direction: Moves along the pre-determined route, follows a policy if provided.
        :param policy:
        :return:
        """
        # Move using the current percepts only.
        desired_move = None
        # Determine which quadrant the agent is in:
        agent_quadrant = self.get_agent_quadrant_location()
        if agent_quadrant == 'Right-Upper':
            desired_loc = (2,8)
            if self.location == desired_loc:
                # Follow the right quadrant policy in accordance to table:
                if len(self.right_upper_quad_policy) != 0:
                    desired_loc = self.right_upper_quad_policy[0]
                    self.right_upper_quad_policy = self.right_upper_quad_policy[1:]
                else:
                    # Policy end, new policy at left-upper quad.
                    desired_loc = (2,3)
                    self.right_upper_quad_policy = None
        elif agent_quadrant == 'Left-Upper':
            desired_loc = (2,3)
            if self.location == desired_loc:
                # Follow left quadrant policy in accordance to table:
                if len(self.left_upper_quad_policy) != 0:
                    desired_loc = self.left_upper_quad_policy[0]
                    self.left_upper_quad_policy = self.left_upper_quad_policy[1:]
                else:
                    # Policy end, new policy at lower-left quad.
                    desired_loc = (8,2)
                    self.left_upper_quad_policy = None
        elif agent_quadrant == 'Right-Lower':
            desired_loc = (8,9)
        else:
            desired_loc = (8,2)
            if self.location == desired_loc:
                # Follow left-lower quad policy:
                if len(self.left_lower_quad_policy) != 0:
                    desired_loc = self.lef_lower_quad_policy[0]
                    self.left_lower_quad_policy = self.left_lower_quad_policy[1:]
                else:
                    # Policy end, new policy at lower-right quad.
                    desired_loc = (8,9)
                    self.left_lower_quad_policy = None
        return self.get_policy_for_target_loc(loc=desired_loc)
        '''
        # Count the number of moveable squares in each direction:
        valid_agent_moves = {Actions.MOVE_UP: 0, Actions.MOVE_RIGHT: 0,
                             Actions.MOVE_DOWN: 0, Actions.MOVE_LEFT: 0}

        initial_location = self.location
        while self.environment.is_valid_action(action=Actions.MOVE_UP, location=initial_location):
            valid_agent_moves[Actions.MOVE_UP] += 1
            initial_location = (initial_location[0] - 1, initial_location[1])

        initial_location = self.location
        while self.environment.is_valid_action(action=Actions.MOVE_RIGHT, location=initial_location):
            valid_agent_moves[Actions.MOVE_RIGHT] += 1
            initial_location = (initial_location[0], initial_location[1] + 1)

        initial_location = self.location
        while self.environment.is_valid_action(action=Actions.MOVE_LEFT, location=initial_location):
            valid_agent_moves[Actions.MOVE_LEFT] += 1
            initial_location = (initial_location[0], initial_location[1] - 1)

        initial_location = self.location
        while self.environment.is_valid_action(action=Actions.MOVE_DOWN, location=initial_location):
            valid_agent_moves[Actions.MOVE_DOWN] += 1
            initial_location = (initial_location[0] + 1, initial_location[1])

        most_possible_moves = 0
        desired_move = None
        for action, num_moves in valid_agent_moves.items():
            if num_moves > most_possible_moves:
                most_possible_moves = num_moves
                desired_move = action
        return desired_move
        '''

    def move_up(self):
        if self.environment.is_valid_action(action=Actions.MOVE_UP, location=self.location):
            self.location = (self.location[0]-1, self.location[1])
        self.performance_score -= 1

    def move_right(self):
        if self.environment.is_valid_action(action=Actions.MOVE_RIGHT, location=self.location):
            self.location = (self.location[0], self.location[1] + 1)
        self.performance_score -= 1

    def move_down(self):
        if self.environment.is_valid_action(action=Actions.MOVE_DOWN, location=self.location):
            self.location = (self.location[0] + 1, self.location[1])
        self.performance_score -= 1

    def move_left(self):
        if self.environment.is_valid_action(action=Actions.MOVE_LEFT, location=self.location):
            self.location = (self.location[0], self.location[1] - 1)
        self.performance_score -= 1

    def suck(self):
        if self.environment.is_dirty(location=self.location):
            self.environment.make_clean(location=self.location)
            print("ReflexAgent-suck: The agent succeeded in cleaning (%d, %d)." % (self.location[0], self.location[1]))
            self.performance_score += 100
        else:
            print("ReflexAgent-suck: Location (%d, %d) was already clean, no reward."
                  % (self.location[0], self.location[1]))

class VacuumWorld:
    world = None
    dirt_locations = None

    def __init__(self, world):
        self.world = world
        self.dirt_locations = self.initialize_dirt(world=world)

    def initialize_dirt(self, world):
        """
        initialize_dirt: Creates a deep copy of the provided world and initializes dirt in
            valid locations with 50% probability of occurrence.
        :param world: The world representing the environment.
        :return dirt_positions: A list of lists where an 'x' indicates a dirty room.
        """
        dirt_positions = deepcopy(world)
        for i, row in enumerate(world):
            for j, element in enumerate(row):
                if element == 1:
                    is_dirty = np.random.randint(low=0, high=2)
                    if is_dirty:
                        dirt_positions[i][j] = 2
        return dirt_positions

    def is_dirty(self, location):
        """
        is_dirty: Returns whether or not the given location is dirty.
        :param location: A tuple (x, y) or (i, j) that indicates the position to test for cleanliness.
        :return boolean: True if the location is dirty (contains a 2), False otherwise.
        """
        if self.dirt_locations[location[0]][location[1]] == 2:
            return True
        else:
            return False

    def make_clean(self, location):
        """
        make_clean: Removes the dirt in the specified location as a result of an agent's action.
        :param location: The location for which the dirt is to be removed.
        :return None: Upon completion; the dirt_locations map will have been updated.
        """
        self.dirt_locations[location[0]][location[1]] = 1

    def is_valid_action(self, action, location):
        """
        is_valid_action: Returns if the desired action is possible in the context of the environment.
        :param action: One of the actions possible to the agent, as described in class Actions.
        :param location: The location of the agent in the environment.
        :return is_valid: True if the action is valid given the agent's location and desired action; False otherwise.
        """
        if action == Actions.MOVE_UP:
            if location[0] == 0:
                # Top of the world, can't move up.
                return False
            else:
                new_y = location[0] - 1
                if self.world[new_y][location[1]] == 0:
                    return False
                else:
                    return True
        elif action == Actions.MOVE_RIGHT:
            if location[1] == len(self.world[location[0]]) - 1:
                # Far right edge of the world, can't move right.
                return False
            else:
                new_x = location[1] + 1
                if self.world[location[0]][new_x] == 0:
                    return False
                else:
                    return True
        elif action == Actions.MOVE_DOWN:
            if location[0] == len(self.world) - 1:
                # Bottom of the world, can't move down.
                return False
            else:
                new_y = location[0] + 1
                if self.world[new_y][location[1]] == 0:
                    return False
                else:
                    return True
        elif action == Actions.MOVE_LEFT:
            if location[1] == 0:
                # Far left edge of the world, can't move left.
                return False
            else:
                new_x = location[1] - 1
                if self.world[location[0]][new_x] == 0:
                    return False
                else:
                    return True
        elif action == Actions.SUCK:
            # Our agent can always suck independent of position.
            return True
        else:
            print("is_valid_action: Error, action to check for validity is not recognized.")

    def get_world(self):
        return self.world

    def print_world(self, agent_location):
        world_state = deepcopy(self.dirt_locations)
        world_state[agent_location[0]][agent_location[1]] = 9
        world_string = ""
        for i, row in enumerate(world_state):
            world_string = world_string + str(row) + "\n"
            # for j, element in enumerate(row):
        print(world_string)


def get_random_agent_position(world):
    """
    get_random_agent_position: Returns the ith and jth index of a random valid starting location for the agent.
    :param world: A list of lists representing the world. A '1' indicates a valid location; '0' indicates otherwise.
    :return (i, j):
        i: The ith index of a randomly selected start position.
        j: The jth index of a randomly selected start position.
    """
    valid_room = False
    i = None
    j = None

    while not valid_room:
        i = np.random.randint(low=0, high=len(world))
        j = np.random.randint(low=0, high=len(world[i]))
        if world[i][j] == 1:
            valid_room = True
    return i, j





def run_simulation(environment, agent, step_count):
    print("Running Simulation with Agent: %s" % type(agent))
    agent.performance_score = 0
    # Main control loop:
    for i in range(step_count):
        if environment.is_dirty(location=agent.location):
            print("Main-CTRL: The agent is attempting to clean location (%d, %d)."
                  % (agent.location[0], agent.location[1]))
            agent.suck()
            print("Main-CTRL: The agent's performance score is now: %d." % agent.performance_score)
        else:
            if type(agent) is BlindAgent:
                # Move in a random direction:
                rand_int = np.random.randint(low=0, high=4)
                direction = Actions(rand_int)
            elif type(agent) is ReflexAgent:
                # Move in a deliberate direction.
                # The only available information is the percepts, use this to create a policy and then execute it.
                policy = {}
                # policy = {state: action}
                policy = agent.get_movement_policy()
                direction = agent.move_direction()
            if direction == Actions.MOVE_LEFT:
                print("Main-CTRL: The agent is attempting to move LEFT.")
                agent.move_left()
            elif direction == Actions.MOVE_UP:
                print("Main-CTRL: The agent is attempting to move UP.")
                agent.move_up()
            elif direction == Actions.MOVE_RIGHT:
                print("Main-CTRL: The agent is attempting to move RIGHT.")
                agent.move_right()
            elif direction == Actions.MOVE_DOWN:
                print("Main-CTRL: The agent is attempting to move DOWN.")
                agent.move_down()
            else:
                print("main: Control Loop directed agent to move randomly in an unsupported way (action not possible).")
            print("Main-CTRL: The agent's performance score is now: %d." % agent.performance_score)
    print("Main: Simulation finished for step-count: %d. Agent %s's performance: %d." \
          % (step_count, type(agent), agent.performance_score))
    return agent.performance_score


def main(environment_type, agent_type, num_repeats, step_count):
    environment = None
    if environment_type is Environment.PRESET:
        m = 12
        n = 12
        world = [[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                 [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]
    elif environment_type is Environment.RANDOM:
        print("Random environment not yet implemented.")
        exit(-1)
    else:
        print("Environment request not recognized!")
        exit(-1)
    # Perform the experiment 'num_repeats' times:
    previous_location = False
    seed_location = None
    performance_scores_blind = []
    performance_scores_reflex = []
    for i in range(num_repeats):
        # Initialize the environment (re-randomize dirt)
        environment = VacuumWorld(world=world)
        print("Main: Initialized Environment: Vacuum World.")
        if previous_location:
            blind_roomba = BlindAgent(environment=environment, location=seed_location)
        else:
            blind_roomba = BlindAgent(environment=environment, location=None)
            reflex_roomba = ReflexAgent(environment=deepcopy(environment), location=blind_roomba.location)
            seed_location = blind_roomba.location
            previous_location = True
        print("Main: Initialized Agent: Blind Roomba at location (%d, %d)."
              % (blind_roomba.location[0], blind_roomba.location[1]))
        print("Main: Initialized Agent: Reflex Roomba at location (%d, %d)."
              % (reflex_roomba.location[0], reflex_roomba.location[1]))
        print("Main: The location of the agents is dirty? %s" % str(environment.is_dirty(blind_roomba.location)))
        print("Main: Agent Blind Roomba's performance score is: %d." % blind_roomba.performance_score)
        print("Main: Agent Reflex Roomba's performance score is: %d." % reflex_roomba.performance_score)
        print("Main: Agent Blind Roomba's world:")
        environment.print_world(blind_roomba.location)
        print("Main: Agent Reflex Roomba's world:")
        reflex_roomba.environment.print_world(reflex_roomba.location)

        performance_score_blind = run_simulation(environment=environment, agent=blind_roomba, step_count=step_count)
        performance_score_reflex = run_simulation(environment=reflex_roomba.environment,
                                                   agent=reflex_roomba, step_count=step_count)
        performance_scores_blind.append(performance_score_blind)
        performance_scores_reflex.append(performance_score_reflex)
    print("Main: Experiment Results BlindAgent: %s" % performance_scores_blind)
    print("Main: Experiment Results ReflexAgent: %s" % performance_scores_reflex)
    print("Main: Average Performance Score BlindAgent (%d runs, %d steps): %.2f."
          % (num_repeats, step_count, np.average(performance_scores_blind)))
    print("Main: Average Performance Score ReflexAgent (%d runs, %d steps): %.2f."
          % (num_repeats, step_count, np.average(performance_scores_reflex)))


if __name__ == '__main__':
    step_count = 200
    main(environment_type=Environment.PRESET, agent_type=Agents.BLIND_AGENT, num_repeats=20, step_count=step_count)

