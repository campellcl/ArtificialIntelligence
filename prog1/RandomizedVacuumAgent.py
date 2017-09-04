"""
RandomizedVacuumAgent.py
Generates environments for the vacuum agent world.
"""

import numpy as np
from copy import deepcopy
from enum import Enum

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
    # Main control loop:
    for i in range(step_count):
        if environment.is_dirty(location=agent.location):
            print("Main-CTRL: The agent is attempting to clean location (%d, %d)."
                  % (agent.location[0], agent.location[1]))
            agent.suck()
            print("Main-CTRL: The agent's performance score is now: %d." % agent.performance_score)
        else:
            # Move in a random direction:
            rand_int = np.random.randint(low=0, high=4)
            direction = Actions(rand_int)
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
    print("Main: Simulation finished for step-count: %d. Agent performance: %d." \
          % (step_count, agent.performance_score))
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
    performance_scores = []
    for i in range(num_repeats):
        # Initialize the environment (re-randomize dirt)
        environment = VacuumWorld(world=world)
        print("Main: Initialized Environment: Vacuum World.")
        if previous_location:
            blind_roomba = BlindAgent(environment=environment, location=seed_location)
        else:
            blind_roomba = BlindAgent(environment=environment, location=None)
            seed_location = blind_roomba.location
            previous_location = True
        print("Main: Initialized Agent: Blind Roomba at location (%d, %d)."
              % (blind_roomba.location[0], blind_roomba.location[1]))
        print("Main: The location of the agent is dirty? %s" % str(environment.is_dirty(blind_roomba.location)))
        print("Main: The agent's performance score is: %d." % blind_roomba.performance_score)
        environment.print_world(blind_roomba.location)
        performance_score = run_simulation(environment=environment, agent=blind_roomba, step_count=step_count)
        performance_scores.append(performance_score)
    print("Main: Experiment Results: %s" % performance_scores)
    print("Main: Average Performance Score (%d runs, %d steps): %.2f."
          % (num_repeats, step_count, np.average(performance_scores)))


if __name__ == '__main__':
    step_count = 200
    main(environment_type=Environment.PRESET, agent_type=Agents.BLIND_AGENT, num_repeats=20, step_count=step_count)

