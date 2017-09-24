"""
p2.py
Implementation of Artificial Intelligence's second programming assignment.
"""
import pandas as pd
from queue import PriorityQueue
from sortedcontainers import SortedList
from prog2.PetDetectiveProblem import PetDetectiveProblem
from prog2.Node import Node
import time

__author__ = "Chris Campell"
__version__ = "9/19/2017"


def extract_agent_location(game_board):
    """
    extract_agent_location: Helper class method during initialization. Returns the agent location when indexed from
        the upper left of the board toward the bottom right.
    :param game_board: The provided string during initialization that represents the game board in array form.
    :return (i, j): A tuple where the ith index represents the row on the board, and the jth index represents the
        column on the board where the agent is located.
    """
    player_position = (None, None)
    for i, row in enumerate(game_board):
        for j, element in enumerate(row):
            if game_board[i][j] == '^':
                return i, j
    if player_position[0] is None or player_position[1] is None:
        return None


def get_street_pet_locations(game_board):
    """
    get_street_pet_locations: Returns a dictionary of the pets (lowercase char __repr__) that are still on the game-
        board, and their associated locations.
    :param game_board: The provided string during initialization that represents the game board in array form.
    :return pet_locations: A dictionary of pet locations of the form...
        :key pet: The lowercase letter representing the pet.
        :var location: The location of the pet in cartesian coordinate form (x, y).
    """
    pet_locations = {}
    for i, row in enumerate(game_board):
        for j, ele in enumerate(row):
            if ele.isalpha():
                if ele.islower():
                    pet_locations[ele] = (i, j)
    return pet_locations


def get_solution_from_node(goal_node):
    """
    get_solution_from_node: Returns the solution to the PetDetectiveProblem subclass in the form of a string comprised
        of as a series of actions {'u','r','d','l'}.
    :param goal_node: The node which constitutes the goal state (instance of Node.py).
    :return solution_string: The string of actions constituting a solution to the problem subclass.
    """
    solution_string = goal_node.action
    node_child = goal_node.node
    while node_child is not None:
        if node_child.action is not None:
            solution_string = solution_string + node_child.action
        node_child = node_child.node

    return solution_string[::-1]


def get_hashable_state_representation(state):
    """
    get_hashable_state_representation: Returns a hashable/immutable representation of the provided state.
    :param state: The current state of a node in the simulation.
    :return hashable_rep: The hashable representation of the provided state.
    """
    hashable_rep = None
    pets_in_street_hashable = tuple(state['pets_in_street'].items())
    pets_in_car_hashable = frozenset(state['pets_in_car'])
    hashable_rep = (('agent_loc', state['agent_loc']),
                    ('pets_in_car', pets_in_car_hashable),
                    ('pets_in_street', pets_in_street_hashable))
    return hashable_rep


def uniform_cost_search(problem_subclass):
    num_nodes_expanded = 0
    # Define a node of the form: (path_cost, state)
    node = Node(state=problem_subclass.initial, path_cost=0)
    frontier = PriorityQueue()
    # Add the initial node to the frontier:
    frontier.put_nowait(node)
    print("Just Added Node (State: %s, Action: %s, PC: %d) to Frontier." % (node.state, node.action, node.path_cost))
    print("Frontier Now: %s" % frontier.queue)
    # Initialize the explored set:
    explored = set()
    while True:
        if frontier.empty():
            # Failure, no solution.
            print("CRITICAL: Frontier now empty. No solution possible. Search Failure.")
            return None, num_nodes_expanded
        node = frontier.get_nowait()
        print("Just Removed Node (State: %s, Action: %s, PC: %d) from Frontier for expansion."
              % (node.state, node.action, node.path_cost))
        print("Frontier Now Length %d: %s" % (len(frontier.queue), frontier.queue))
        if problem_subclass.goal_test(node.state):
            print("CRITICAL: Reached goal state! Returning solution...")
            solution_string = get_solution_from_node(goal_node=node)
            return solution_string, num_nodes_expanded
        # Modify the node's state to be hashable for a set:
        # hashable_state = node.__hash__()
        explored.add(node)
        print("Just added Node (State: %s, Action: %s, PC: %d) to Explored (if not already in set)."
              % (node.state, node.action, node.path_cost))
        print("Explored now: %s" % explored)
        for action in problem_subclass.actions(node.state):
            resultant_state = problem_subclass.result(state=node.state, action=action)
            path_cost = problem_subclass.path_cost(c=1,state1=node.state,
                                                              action=action,state2=resultant_state)
            child_node = Node(state=resultant_state, path_cost=path_cost + node.path_cost,
                              problem=problem_subclass, node=node, action=action)
            num_nodes_expanded += 1
            # child_node.problem.print_world(child_node.state)
            print("Generated new child_node (State: %s, Action Performed: %s, PC: %d) for consideration."
                  % (child_node.state, child_node.action, child_node.path_cost))
            print("Agent moved FROM: %s TO: %s with action %s"
                  % (node.state['agent_loc'], child_node.state['agent_loc'], action))
            if child_node not in explored:
                # The child node is not explored.
                if child_node not in frontier.queue:
                    # The child node is not explored, and is not in the frontier.
                    # Add the child node to the frontier for expansion.
                    frontier.put_nowait(child_node)
                    print("Just Added Node (State: %s, Action: %s, PC: %d) to Frontier."
                          % (child_node.state, child_node.action, child_node.path_cost))
                    print("Frontier Now: %s" % frontier.queue)
                else:
                    # The child node is not explored, and is in the frontier.
                    # Does the child in the frontier have a higher path cost:
                    for index, frontier_node in enumerate(frontier.queue):
                        if frontier_node == child_node:
                            if frontier_node.path_cost > child_node.path_cost:
                                # The frontier's copy of the node has a higher path-cost.
                                # Replace the frontier's copy with the new copy with lower path cost.
                                frontier.queue[index] = child_node
            else:
                # The child node is explored.
                print("The generated child node was already in Explored. Generating a new one...")
            '''
            if child_node not in explored or child_node not in frontier.queue:
                frontier.put_nowait(child_node)
            elif child_node in frontier:
                # check to see if frontier child_node has higher path cost:
                for node in frontier.queue:
                    if node[1]['state'] == child_node[1]['state']:
                        if node[0] > child_node[0]:
                            frontier.queue.remove(node)
                            frontier.put_nowait(child_node)
            '''

def main(training_file, solution_file):
    # Read in required information:
    df_problems = None
    df_solutions = None
    # Read puzzles for specified training_file:
    with open(training_file, 'r') as fp:
        df_problems = pd.read_csv(fp)
    # Read solutions for specified training_file:
    with open(solution_file, 'r') as fp:
        df_solutions = pd.read_csv(fp)
    for i, row in enumerate(df_problems.iterrows()):
        try:
            solution_row = df_solutions.iloc[0]
            # Solution already exists:
            print("Solution for Problem Set: %s, Puzzle ID: %d already exists." % (training_file, i))
        except IndexError:
            print("Solution for Problem Set: %s, Puzzle ID: %d does not exist. Attempting to solve..." % (training_file, i))
            # solution does not already exist.
            train_id = df_problems['Id'][i]
            train_board = df_problems['board'][i]
            game_board = train_board.split(';')[0:-1]
            solvable = False
            agent_location = extract_agent_location(game_board=game_board)
            pet_locations = get_street_pet_locations(game_board=game_board)
            # Define our initial state to be: (location, pets_in_car, pets_on_street)
            init_state = {'agent_loc': agent_location, 'pets_in_car': [], 'pets_in_street': pet_locations}
            # Define our goal state to be: (location, pets_in_car=[], pets_on_street={})
            desired_goal_state = {'pets_in_car': [], 'pets_in_street': {}}
            # Define our problem to be:
            pet_detective_problem = PetDetectiveProblem(initial_state=init_state,
                                                        goal_state=desired_goal_state, game_board=game_board)
            pet_detective_problem.print_world(game_state=init_state)
            start_time_uniform_cost = time.time()
            solution, num_nodes_expanded = uniform_cost_search(problem_subclass=pet_detective_problem)
            solution_time = (time.time() - start_time_uniform_cost)
            if solution is not None:
                # Solution found!
                solvable = True
                print("----------------Solution Time %s Seconds for Valid Solution of Puzzle %d-------------"
                      % (i, solution_time))
                print("Solution String: %s" % solution)
            else:
                print("----------------Solution Time %s Seconds for NO Solution of Puzzle %d-------------"
                      % (i, solution_time))
                print("Solution: NONE")
            # Write solution to csv of form:
            #problem_set,puzzle_id,solvable,solution,num_nodes_expanded,solution_time
            csv_line = '%s,%d,%s,%s,%d,%f\n' \
                       % (training_file, train_id, solvable, solution, num_nodes_expanded, solution_time)
            with open(solution_file, 'a') as fp:
                fp.write(csv_line)
        break
            # TODO: Modify program to not re-solve puzzles and to restart timeclock.


if __name__ == '__main__':
    # Enumerate training files:
    training_data = ['../prog2/train/lumosity_breadth_first_search_train.csv']
    # Grab solution file:
    solution_file = '../prog2/solutions/solutions.csv'
    # Perform experiment for desired training file:
    main(training_file=training_data[0], solution_file=solution_file)
