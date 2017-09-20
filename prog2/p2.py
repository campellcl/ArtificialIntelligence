"""
p2.py
Implementation of Artificial Intelligence's second programming assignment.
"""
import pandas as pd
from queue import PriorityQueue
from prog2.PetDetectiveProblem import PetDetectiveProblem

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

def print_world(game_board):
        """
        print_world: Prints a human-readable version of the game board to the console.
        :param game_board: The provided string during initialization that represents the game board in array form.
        :return None: Upon completion; a human-readable string representation of the game state is printed to stdout.
        """
        world_string = ''
        for row in game_board:
            world_string = world_string + row + "\n"
        print(world_string)


def get_solution_from_state(state):
    print("Supposed to get the solution now...woops!")
    pass


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
    # Define a node of the form: (path_cost, state)
    node = (0, {'state': problem_subclass.initial})
    frontier = PriorityQueue()
    # Add the initial node to the frontier:
    frontier.put_nowait(node)
    # Initialize the explored set:
    explored = set()
    while True:
        if frontier.empty():
            # Failure, no solution.
            return None
        node = frontier.get_nowait()
        if problem_subclass.goal_test(node[1]['state']):
            return get_solution_from_state(node[1]['state'])
        # Modify the node's state to be hashable for a set:
        hashable_state = get_hashable_state_representation(node[1]['state'])
        explored.add(hashable_state)
        for action in problem_subclass.actions(node[1]['state']):
            resultant_state = problem_subclass.result(state=node[1]['state'], action=action)
            path_cost = problem_subclass.path_cost(c=1,state1=node[1]['state'],
                                                              action=action,state2=resultant_state)
            child_node = (path_cost + node[0], {'state':resultant_state, 'problem': problem_subclass,
                                                'node': node, 'action': action})
            if get_hashable_state_representation(child_node[1]['state']) not in explored \
                    or get_hashable_state_representation(child_node[1]['state']) not in frontier:
                frontier.put_nowait(child_node)
            elif child_node[1]['state'] in frontier:
                # check to see if frontier child_node has higher path cost:
                for node in frontier.queue:
                    if node[1]['state'] == child_node[1]['state']:
                        if node[0] > child_node[0]:
                            frontier.queue.remove(node)
                            frontier.put_nowait(child_node)


def main(training_file):
    # Read in required information:
    df = None
    with open(training_file, 'r') as fp:
        df = pd.read_csv(fp)
    for i, row in enumerate(df.iterrows()):
        train_id = df['Id'][i]
        train_board = df['board'][i]
        game_board = train_board.split(';')[0:-1]
        agent_location = extract_agent_location(game_board=game_board)
        pet_locations = get_street_pet_locations(game_board=game_board)
        # Define our initial state to be: (location, pets_in_car, pets_on_street)
        init_state = {'agent_loc': agent_location, 'pets_in_car': [], 'pets_in_street': pet_locations}
        # Define our goal state to be: (location, pets_in_car=[], pets_on_street={})
        desired_goal_state = {'pets_in_car': [], 'pets_in_street': {}}
        # Define our problem to be:
        pet_detective_problem = PetDetectiveProblem(initial_state=init_state,
                                                    goal_state=desired_goal_state, game_board=game_board)
        print_world(game_board=game_board)
        solution = uniform_cost_search(problem_subclass=pet_detective_problem)
        if solution is not None:
            # Solution found!
            print("Found solution: %s" % solution)


if __name__ == '__main__':
    # Enumerate training files:
    training_data = ['../prog2/train/lumosity_breadth_first_search_train.csv']
    # Perform experiment for desired training file:
    main(training_file=training_data[0])
