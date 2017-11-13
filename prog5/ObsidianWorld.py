"""
ObsidianWorld.py
"""

from collections import OrderedDict
import numpy as np
import sys
from pathlib import Path

__author__ = "Chris Campell"
__version__ = '11/6/2017'


class ObsidianWorldMDP:
    initial_state = None
    terminal_states = None
    states = None
    edges = None
    movement_cpts = None
    gamma = None
    # The number of states:
    n = None
    # The number of edges:
    m = None
    # The number of terminal states:
    n_t = None

    def __init__(self, initial_state, terminal_states, states, edges, movement_cpts, gamma):
        """
        __init__: Constructor for objects of type ObsidianWorldMDP. Creates an obsidian world instance with data parsed
            from the text file.
        :param initial_state: The starting state.
        :param terminal_states: The terminal states.
        :param states: A list of all the states and their associated rewards.
        :param edges: A list of all the directed edge connections.
        :param movement_cpts: The conditional probability tables associated with movement in a stochastic environment.
        :param gamma: The discount factor to penalize long action sequences.
        """
        for state, reward in states:
            if state == initial_state:
                self.initial_state = (state, reward)
        self.terminal_states = []
        # Keep track of the reward associated with each terminal state:
        for term_state in terminal_states:
            for state, reward in states:
                if state == term_state:
                    self.terminal_states.append((term_state, reward))
        self.states = states
        # Maintain a list of non-terminal states:
        self.nonterm_states = states.copy()
        for term_state, reward in self.terminal_states:
            self.nonterm_states.remove((term_state, reward))
        self.edges = edges
        self.movement_cpts = movement_cpts
        self.gamma = gamma
        # Set the number of states 'n':
        self.n = len(states)
        # Set the number of edges 'm':
        self.m = len(edges)
        # Set the number of terminal states 'n_t':
        self.n_t = len(terminal_states)


def parse_information(input_file):
    world_info = {'initial_state': None, 'states': [], 'terminal_states': None, 'movement_cpts': {}, 'gamma': None, 'edges': {}}
    with open(input_file, 'r') as fp:
        input_file = fp.read()
    input_text = input_file.split('\n')
    loop_locations = {}
    for line_num, line in enumerate(input_text):
        if line.isnumeric():
            loop_start = line_num + 1
            loop_end = line_num + int(line) + 1
            loop_locations[len(loop_locations)] = [loop_start, loop_end]
    for loop_id, loop_info in loop_locations.items():
        if loop_id == 0:
            # States loop:
            for i in range(loop_info[0], loop_info[1]):
                split_state_text = input_text[i].split(' ')
                state = (split_state_text[0], int(split_state_text[1]))
                world_info['states'].append(state)
            # Get terminal states:
            terminal_states_text = input_text[loop_info[1]].split(' ')
            world_info['terminal_states'] = terminal_states_text
        elif loop_id == 1:
            # Movement cpts loop:
            for i in range(loop_info[0], loop_info[1]):
                split_cpt_dir = input_text[i].split(' ')
                # The desired move is north (key one)
                source_dir = split_cpt_dir[0]
                # The value is the probability in ending up in key '{N,E,S,W}' given the source_dir:
                world_info['movement_cpts'][source_dir] = {
                    'N': float(split_cpt_dir[1]),
                    'E': float(split_cpt_dir[2]),
                    'S': float(split_cpt_dir[3]),
                    'W': float(split_cpt_dir[4])
                }
        elif loop_id == 2:
            # State edges loop:
            for i in range(loop_info[0], loop_info[1]):
                state_trans_pair = input_text[i].split(' ')
                s = state_trans_pair[0]
                action = state_trans_pair[1]
                s_prime = state_trans_pair[2]
                if s in world_info['edges']:
                    world_info['edges'][s][action] = s_prime
                else:
                    world_info['edges'][s] = {}
                    world_info['edges'][s][action] = s_prime
        else:
            print("Warning: This input loop is extraneous or does not exist.")
    if input_text[-1] == '':
        input_text = input_text[:-1]
    # Get the initial state (the last line)
    world_info['initial_state'] = input_text[-1]
    # Get gamma (the second to last line)
    world_info['gamma'] = float(input_text[-2])
    return world_info


def value_iteration(mdp, epsilon):
    U = {state: 0 for (state, reward) in mdp.states}
    # Insert the utilities of the terminal states:
    for term_state, reward in mdp.terminal_states:
        U[term_state] = reward
    # Create U' to hold the batch utility updates:
    U_prime = U.copy()
    # A boolean flag to determine convergence:
    converged = False
    while not converged:
        U = U_prime.copy()
        delta = 0.0
        # Perform value iteration from the specified start state:
        for i, (state, reward) in enumerate(mdp.states):
            # If the state is a terminal state, its utility is pre-determined:
            if state not in [state for state, reward in mdp.terminal_states]:
                # If the state is not a terminal state we need to calculate it's expected utility:
                expected_utility = OrderedDict()
                for desired_action, desired_s_prime in mdp.edges[state].items():
                    # The desired action is the action the agent wishes to perform.
                    # The desired s' is the state that the agent wishes to end up in.
                    remaining_states = mdp.edges[state].copy()
                    remaining_states.pop(desired_action)
                    # Compute the expected utility of the desired action, given the stochastic probability of the
                    #   desired action actually occurring:
                    expected_utility[desired_action] = mdp.movement_cpts[desired_action][desired_action]*U[desired_s_prime]
                    # The expected utility of the desired action also depends on the stochastic probabilities of the
                    #   agent ending up in any other reachable state:
                    for stochastic_action, stochastic_s_prime in remaining_states.items():
                        expected_utility[desired_action] += mdp.movement_cpts[desired_action][stochastic_action]*U[stochastic_s_prime]
                # Perform the expected utility update:
                U_prime[state] = reward + (mdp.gamma * np.max(list(expected_utility.values())))
            # Calculate the change in utility for the current state:
            delta_update = np.abs(U_prime[state] - U[state])
            # Update the maximum change in utility (delta):
            if delta_update > delta:
                delta = delta_update
        # Perform a check for convergence using epsilon as a tolerance:
        if delta <= epsilon*((1 - mdp.gamma)/mdp.gamma):
            converged = True
    # Return the true (converged) utilities for each state:
    return U


def is_terminal_state(s, mdp):
    """
    is_terminal_state: Returns true if the provided state is a terminal state.
    :param s: The state to check for finiteness.
    :param mdp: The markov decision process instance.
    :return boolean: True if the provided state is terminal, false otherwise.
    """
    return s in [state for state, reward in mdp.terminal_states]


def expected_utilities(neighbors, utilities, trans_prob_tables):
    """
    expected_utilities: Computes the expected utility (the average utility value of the outcomes of the specified action
        executed in the provided state, weighted by the probability that the action actually occurs) for every possible
        action performed given the adjacent states and the stochastic transition probabilities.
    :param neighbors: The adjacent states which factor into the expected utility calculation.
    :param utilities: The actual utilities of the states in the state-space.
    :param trans_prob_tables: The probability transition tables that define the probability associated with each action.
    :return expected_utility: The expected utility for the provided action when executed at the provided state.
    """
    expected_utils = OrderedDict()
    for desired_action, desired_s_prime in neighbors.items():
        remaining_states = neighbors.copy()
        remaining_states.pop(desired_action)
        # Compute the expected utility of the desired action, given the stochastic probability of the
        #   desired action actually occurring:
        expected_utils[desired_action] = trans_prob_tables[desired_action][desired_action]*utilities[desired_s_prime]
        # The expected utility of the desired action also depends on the stochastic probabilities of the
        #   agent ending up in any other reachable state:
        for stochastic_action, stochastic_s_prime in remaining_states.items():
            expected_utils[desired_action] += \
                trans_prob_tables[desired_action][stochastic_action]*utilities[stochastic_s_prime]
    return expected_utils


def compute_policy(utilities, mdp):
    """
    compute_policy: Given a converged list of states and utilities and a Markov Decision Process (MDP) instance,
        calculates the optimal policy based on the expected_utility of every state.
    :param utilities: The list of state, utility pairs.
    :param mdp: An instance of a Markov Decision Process (MDP)
    :return policy: The optimal policy given the calculated expected utilities of each state.
    """
    policy = {}
    for state, util in utilities.items():
        # Only compute the policy for non-terminal states:
        if not is_terminal_state(state, mdp):
            # Since this state is a non-terminal state, its policy is its expected utility:
            # Compute the expected utilities of all available actions:
            expected_utils = expected_utilities(neighbors=mdp.edges[state],
                                               utilities=utilities, trans_prob_tables=mdp.movement_cpts)
            max_expected_util = -np.inf
            best_action = None
            for action, expected_util in expected_utils.items():
                if expected_util >= max_expected_util:
                    max_expected_util = expected_util
                    best_action = action
            policy[state] = best_action
    return policy


def main(input_file):
    metadata = parse_information(input_file)
    mdp = ObsidianWorldMDP(
        initial_state=metadata['initial_state'],
        terminal_states=metadata['terminal_states'],
        states=metadata['states'],
        edges=metadata['edges'],
        movement_cpts=metadata['movement_cpts'],
        gamma=metadata['gamma']
    )
    utilities = value_iteration(mdp=mdp, epsilon=10**-5)
    policies = compute_policy(utilities, mdp)
    write_dest = 'solutions/' + input_file.split('/')[1].replace('.txt', '.csv')
    write_path = Path(write_dest)
    if not write_path.is_file():
        with open(write_dest, 'w') as fp:
            fp.write('State,Utility,Policy\n')
            for state, reward in sorted(mdp.states):
                utility = utilities[state]
                if state not in [state for state, reward in mdp.terminal_states]:
                    policy = policies[state]
                    fp.write('%s,%.3f,%s\n' % (state, utility, policy))
                else:
                    fp.write('%s,%.3f,\n' % (state, utility))


if __name__ == '__main__':
    input_files = [
        'fig_17_3.txt', 'loser.txt',
        'simple_g09_r0.txt', 'simple_g10_r1.txt',
        'simple_g10_r3.txt', 'tunnel_a2_g10_r1.txt',
        'tunnel_a2_g10_r1.txt', 'tunnel_a4_g10_r1.txt',
        'random_0.txt', 'random_1.txt',
        'random_2.txt', 'random_3.txt',
        'random_4.txt'
    ]
    for input_file in input_files:
        main(input_file='value_iteration/' + input_file)
