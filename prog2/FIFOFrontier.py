"""
FIFOFrontier.py
Implementation of a FIFO Queue with efficent hash-based membership testing.
"""

__author__ = "Chris Campell"
__version__ = '9/28/2017'

"""
Frontier.py
Custom frontier implementation utilizes two data structures to take advantage of runtime efficiency.
"""

class FIFOFrontier:
    index = 0
    sorted_by_fifo = None
    members = None

    def __init__(self, key=NotImplemented):
        self.key = key
        self.sorted_by_fifo = []
        self.members = set()

    def add(self, node):
        # is the node in the frontier already?
        if node not in self.members:
            self.members.add(node)
            self.sorted_by_fifo.append(node)

    def pop(self):
        node_to_remove = self.sorted_by_fifo[0]
        self.sorted_by_fifo.remove(node_to_remove)
        self.members.remove(node_to_remove)
        return node_to_remove

    def __contains__(self, node):
        return node in self.members

    def __len__(self):
        return len(self.members)

    def __iter__(self):
        return self.sorted_by_fifo.__iter__()

    def __next__(self):
        if self.index == len(self.sorted_by_fifo) - 1:
            raise StopIteration
        else:
            self.index += 1
            return self.sorted_by_fifo[self.index]

    def __repr__(self):
        return self.sorted_by_fifo.__repr__()
