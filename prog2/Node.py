"""
Node.py
Implementation of a Node for AI's programming assignment two. Node must maintain state information as well as path cost.
Additionally, the Node must be hashable (for use with a PriorityQueue) and support efficient membership testing.
"""

__author__ = "Chris Campell"
__version__ = "9/20/2017"


class Node:
    state = None
    path_cost = None

    def __init__(self, state, path_cost, problem=None, node=None, action=None):
        self.state = state
        self.path_cost = path_cost
        self.problem = problem
        self.node = node
        self.action = action

    def __hash__(self):
        # TODO: Verify method works.
        pets_in_car_hashable = frozenset(self.state['pets_in_car'])
        pets_in_street_hashable = tuple(self.state['pets_in_street'].items())
        hashable_tuple = (self.path_cost, self.state['agent_loc'], pets_in_car_hashable, pets_in_street_hashable)
        return hash(hashable_tuple)

    def __eq__(self, other):
        # TODO: Verify method works.
        if isinstance(other, Node):
            if self.path_cost == other.path_cost:
                if self.__hash__() == other.__hash__():
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def get_state(self):
        return self.state

    def get_path_cost(self):
        return self.path_cost

