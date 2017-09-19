"""
PetDetectiveProblem.py
Represents the PetDetective problem, subclasses aima's Problem class.
"""

from prog2.aima.search import Problem

__author__ = "Chris Campell"
__version__ = "9/19/2017"


class PetDetectiveProblem(Problem):

    def __init__(self, initial_state, goal_state):
        super().__init__(initial=initial_state, goal=goal_state)
        # TODO: perform additional self.init as required.

    def actions(self, state):
        """
        actions: Returns the actions that can be executed in the given state.
        :param state: The state from which possible actions are to be determined.
        :return actions: A list of possible actions to be executed.
        """
        # TODO: method body.
        pass

    def result(self, state, action):
        """
        result: Returns the state that results from executing the given action in the given state. The returned action
            must be one of self.actions(state).
        :param state:
        :param action:
        :return resultant_state: The state that results from executing the given action in the provided state.
        """
        # TODO: method body.
        pass

    def goal_test(self, state):
        """
        goal_test: Returns true if the state is a goal.
        :param state:
        :return is_goal: True if the provided state is a goal state, false otherwise.
        """
        # TODO: method body.
        pass

    def path_cost(self, c, state1, action, state2):
        """
        path_cost: Returns the cost of a solution path that arrives at state2 from state1 via action, assuming cost 'c'
            to arrive at state1.
        :param c:
        :param state1:
        :param action:
        :param state2:
        :return path_cost: The cost of a solution path that arrives at state2 from state1 assuming cost 'c'.
        """
        # TODO: method body.
        pass
