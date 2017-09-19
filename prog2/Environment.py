"""
Environment.py
Represents an environment (world state) for the Pet Detective game.
"""

from prog2.UniformCostAgent import UniformCostAgent

__author__ = "Chris Campell"
__version__ = "9/19/2017"


class Environment:
    puzzle_id = None
    game_board = None
    # agent_location measured from top left (matrix notation)
    agent_location = (None, None)
    pet_locations = [(None, None)]

    def __init__(self, puzzle_id, game_board):
        self.puzzle_id = puzzle_id
        self.game_board = game_board.split(';')[0:-1]
        self.agent_location = Environment.extract_agent_location(game_board)

    @staticmethod
    def extract_agent_location(game_board):
        """
        extract_agent_location: Helper class method during initialization. Returns the agent location when indexed from
            the upper left of the board toward the bottom right.
        :param game_board: The provided string during initialization that represents the game board.
        :return (i, j): A tuple where the ith index represents the row on the board, and the jth index represents the
            column on the board where the agent is located.
        """
        game_rows = game_board.split(';')
        player_position = (None, None)
        for i, row in enumerate(game_rows):
            for j, element in enumerate(row):
                if game_rows[i][j] == '^':
                    return i, j
        if player_position[0] is None or player_position[1] is None:
            return None

    def get_num_pets_remaining(self):
        """
        get_num_pets_remaining: Returns the number of pets that are remaining on the game board.
        :return num_pets: The number of pets on the provided game board.
        """
        num_pets = 0
        for i,row in enumerate(self.game_board):
            for j, ele in enumerate(row):
                if ele.isalpha():
                    if ele.islower():
                        num_pets += 1
        return num_pets

    def get_street_pet_locations(self):
        """
        get_street_pet_locations: Returns a dictionary of the pets (lowercase char __repr__) that are still on the game-
            board, and their associated locations.
        :return pet_locations: A dictionary of pet locations of the form...
            :key pet: The lowercase letter representing the pet.
            :var location: The location of the pet in cartesian coordinate form (x, y).
        """
        pet_locations = {}
        for i, row in enumerate(self.game_board):
            for j, ele in enumerate(row):
                if ele.isalpha():
                    if ele.islower():
                        pet_locations[ele] = (i, j)
        return pet_locations

    def print_world(self):
        """
        print_world: Prints a human-readable version of the game board to the console.
        :return None: Upon completion; a human-readable string representation of the game state is printed to stdout.
        """
        deliminated_strings = self.game_board.split(';')
        world_string = ''
        for row in deliminated_strings:
            world_string = world_string + row + "\n"
        print(world_string)

    def is_road(self, desired_location):
        """
        is_road: Returns if the desired location is a road ('-', '+', '|', {a-z,A-Z}).
        :param desired_location: The location to check for existence of a road in the form (x,y).
        :return boolean: True if the provided location is a road (may contain pet), false otherwise.
        """
        if self.game_board[desired_location[0]][desired_location[1]] != '.':
            return True
        else:
            return False

    def is_valid_action(self, action):
        """
        is_valid_action: Determines if the specified action is executable based on the presence of the roadways and the
            map edge.
        :param action: The action that the agent wishes to perform in the context of the environment.
        :return boolean: True if the action is executable according to game rules, false otherwise.
        """
        if action == UniformCostAgent.Actions.MOVE_UP:
            return self.is_road(desired_location=(self.agent_location[0] - 1,self.agent_location[1]))
        elif action == UniformCostAgent.Actions.MOVE_RIGHT:
            return self.is_road(desired_location=(self.agent_location[0], self.agent_location[1] + 1))
        elif action == UniformCostAgent.Actions.MOVE_DOWN:
            return self.is_road(desired_location=(self.agent_location[0] + 1,self.agent_location[1]))
        elif action == UniformCostAgent.Actions.MOVE_LEFT:
            return self.is_road(desired_location=(self.agent_location[0], self.agent_location[1] - 1))
        else:
            print("is_valid_action: Error, unknown agent action.")

    def goal_test(self, num_pets_in_car, num_pets_in_street):
        if num_pets_in_street == 0:
            # No more pets in the street.
            if num_pets_in_car == 0:
                # goal_test success:
                return True
        return False
