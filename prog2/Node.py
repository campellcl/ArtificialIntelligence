"""
Node.py
Implementation of a Node for AI's programming assignment two. Node must maintain state information as well as path cost.
Additionally, the Node must be hashable (for use with a PriorityQueue) and support efficient membership testing.
"""

from functools import total_ordering
from line_profiler import LineProfiler

__author__ = "Chris Campell"
__version__ = "9/20/2017"

def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner

class Node:
    state = None
    path_cost = None
    hashed_value = None

    def __init__(self, state, path_cost, problem=None, node=None, action=None):
        self.state = state
        self.path_cost = path_cost
        self.problem = problem
        self.node = node
        self.action = action
        self.hashed_value = None

    def __hash__(self):
        pets_in_car_hashable = frozenset(self.state['pets_in_car'])
        pets_in_street_hashable = tuple(self.state['pets_in_street'].items())
        # Warning: Do not include path cost as part of the state hashing (this is a node attribute).
        hashable_tuple = (self.state['agent_loc'], pets_in_car_hashable, pets_in_street_hashable)
        self.hashed_value = hash(hashable_tuple)
        return hash(hashable_tuple)

    def __eq__(self, other):
        # TODO: Ensure this still works after modifying from hash-based comparator.
        if isinstance(other, Node):
            if self.hashed_value is not None:
                if other.hashed_value is not None:
                    return self.hashed_value == other.hashed_value
                else:
                    return self.hashed_value == other.__hash__()
            else:
                if other.hashed_value is not None:
                    return self.__hash__() == other.hashed_value
                else:
                    return self.__hash__() == other.__hash__()
        else:
            return NotImplemented

    def _is_valid_operand(self, other):
        """
        _is_valid_operand: Determines if the target for comparison has the necessary attributes
            used to determine ordering.
        :param other: The desired object to compare against hopefully a node.
        :return NotImplemented: An error indicating that the other object is not a node and therefore cannot be compared
            to one.
        :return boolean: True if the node to compare against has a path cost for valid comparison, false otherwise.
        :source docs: https://docs.python.org/3/library/functools.html#functools.total_ordering
        :source SO: https://stackoverflow.com/questions/33764584/typeerror-unorderable-types-node-node
        """
        if not isinstance(other, Node):
            return NotImplemented
        else:
            return hasattr(other, 'path_cost')

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.action < other.action

    def __le__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.action <= other.action

    def __ne__(self, other):
        # TODO: Ensure this still works after modifying from hash-based comparator.
        if self._is_valid_operand(other):
            if self.state['agent_loc'] != other.state['agent_loc']:
                if self.state['pets_in_car'] != other.state['pets_in_car']:
                    if self.state['pets_in_street'] != other.state['pets_in_street']:
                        return True
            return False
        else:
            return NotImplemented

    def __gt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        else:
            return self.action > other.action

    def __ge__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        else:
            return self.action >= other.action

    def __repr__(self):
        return "(Action Performed: %s, PC: %d, State: %s)" % (self.action, self.path_cost, self.state)

    def get_state(self):
        return self.state

    def get_path_cost(self):
        return self.path_cost
