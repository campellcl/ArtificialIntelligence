"""
UniformCostAgent.py
Represents a UniformCost agent in the provided environment.
"""
from enum import Enum
from queue import PriorityQueue


class UniformCostAgent:
    """
    UniformCostAgent
    Represents a UniformCost agent in the provided environment.
    """
    location = None
    pets_in_car = None
    pets_on_street = None
    environment = None
    frontier = None
    explored = set()

    def __init__(self, location, pets_in_car, pets_on_street, environment_instance):
        self.location = location
        self.pets_in_car = pets_in_car
        self.pets_on_street = pets_on_street
        self.environment = environment_instance
        self.frontier = PriorityQueue()
        self.frontier.put_nowait((0, {'agent_loc': location, 'pets_in_car': pets_in_car,
                                      'pets_on_street': pets_on_street, 'parent': None}))

    class Actions(Enum):
        MOVE_UP = 0
        MOVE_RIGHT = 1
        MOVE_DOWN = 2
        MOVE_LEFT = 3

    def uniform_cost_search(self):
        is_failure = False
        if self.frontier.empty():
            return None
        node = self.frontier.get_nowait()
        if self.environment.goal_test(num_pets_in_car=len(node[1]['pets_in_car']),
                                      num_pets_in_street=len(node[1]['pets_in_street'])):
            # TODO: Modify node to keep track of solution history.
            return node
        self.explored.add(node)
        for action in self.Actions:




    def execute_action(self, action):
        """
        execute_action: Performs the requested action and updates the environment to reflect the ground truth.
        :param action: The desired action (a member of the Actions class).
        :return status: The result of executing the action (success or failure).
            :type bool: True if action was executed, false otherwise.
        """
        is_valid_action = False
        if action in self.Actions:
            if self.environment.is_valid_action(action=action):
                is_valid_action = True
                # Perform the action
                if action == UniformCostAgent.Actions.MOVE_UP:
                    self.move_up()
                elif action == UniformCostAgent.Actions.MOVE_RIGHT:
                    self.move_right()
                elif action == UniformCostAgent.Actions.MOVE_DOWN:
                    self.move_down()
                elif action == UniformCostAgent.Actions.MOVE_LEFT:
                    self.move_left()
                else:
                    print("execute_action: Requested action %s not available to agent." % action)
                # Update the environment.
                self.environment.agent_location = self.location
        return is_valid_action

    def move_up(self):
        if self.execute_action(action=UniformCostAgent.Actions.MOVE_UP):
            print("move_up: The agent moved up.")
        else:
            print("move_up: The agent failed while attempting to move up.")

    def move_right(self):
        if self.execute_action(action=UniformCostAgent.Actions.MOVE_RIGHT):
            print("move_right: The agent moved right.")
        else:
            print("move_right: The agent failed while attempting to move right.")

    def move_down(self):
        if self.execute_action(action=UniformCostAgent.Actions.MOVE_DOWN):
            print("move_down: The agent moved down.")
        else:
            print("move_down: The agent failed while attempting to move down.")

    def move_left(self):
        if self.execute_action(action=UniformCostAgent.Actions.MOVE_LEFT):
            print("move_left: The agent moved left.")
        else:
            print("move_left: The agent failed while attempting to move left.")
