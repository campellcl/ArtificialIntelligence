"""
Frontier.py
Custom frontier implementation utilizes two data structures to take advantage of runtime efficiency.
"""

from sortedcontainers import SortedListWithKey, SortedSet

class Frontier:
    index = 0
    sorted_by_heuristic = None
    members = None

    def __init__(self, key=NotImplemented):
        self.key = key
        self.sorted_by_heuristic = SortedListWithKey(key=key)
        self.members = set()

    def add(self, node):
        # is the node in the frontier already?
        if node not in self.members:
            self.members.add(node)
            self.sorted_by_heuristic.add(node)

    def pop(self):
        node_to_remove = self.sorted_by_heuristic.pop(idx=0)
        self.members.remove(node_to_remove)
        return node_to_remove

    def __contains__(self, node):
        return node in self.members

    def __len__(self):
        return len(self.members)

    def __iter__(self):
        return self.sorted_by_heuristic.__iter__()

    def __next__(self):
        if self.index == len(self.sorted_by_heuristic) - 1:
            raise StopIteration
        else:
            self.index += 1
            return self.sorted_by_heuristic.pop(idx=self.index)

    def __repr__(self):
        return self.sorted_by_heuristic.__repr__()
