"""
ProgramTwo.py
Implementation for programming assignment two of Artificial Intelligence.
"""

import numpy as np
import pandas as pd
from prog2.Environment import Environment
from prog2.UniformCostAgent import UniformCostAgent
__author__ = "Chris Campell"
__version__ = "9/18/2017"


def main(training_file):
    df = None
    with open(training_file, 'r') as fp:
        df = pd.read_csv(fp)
    for i, row in enumerate(df.iterrows()):
        train_id = df['Id'][i]
        train_board = df['board'][i]
        pet_world = Environment(puzzle_id=train_id, game_board=train_board)
        uniform_pet_agent = UniformCostAgent(location=pet_world.agent_location, pets_in_car=[],
                                     pets_on_street=pet_world.get_street_pet_locations(), environment_instance=pet_world)
        solution = uniform_pet_agent.uniform_cost_search()
        failure = False
        while not failure:
            if len(pet_agent.frontier) == 0:
                failure = True
                break
            # select lowest cost node in frontier:


if __name__ == '__main__':
    training_data = ['../prog2/train/lumosity_breadth_first_search_train.csv']
    main(training_file=training_data[0])
