"""
BayesNetwork.py
An implementation of a Bayesian Network for Programming Assignment Four.
"""

__author__ = "Chris Campell"
__version__ = "10/28/2017"

import pandas as pd

def enumeration_ask(X,e,bn):
    """
    enumeration_ask: Returns a probability distribution over X.
    :param X: The query variable for which probabilities are to be inferred.
    :param e: The observed values for variables E.
    :param bn: A Bayesian Network with variables {X} union E union Y (hidden variables).
    :return norm_dist_x: A normalized probability distribution over the query variable X.
    """
    return NotImplementedError


def enumerate_all(vars, e):
    """
    enumerate_all: Helper method for enumeration_ask, computes the joint probability distribution of the provided
        variables, given a set of observations.
    :param vars: A list of input variables for which to construct the joint from.
    :param e: A list of observations of the input variables which influence the joint distribution.
    :return joint_dist: The joint distribution of the provided 'vars' with the supplied observations 'e'.
    """
    return NotImplementedError


def construct_probability_table(node, observations):
    """
    construct_probability_table: Builds a probability table for the provided node given the provided observations-
        (the number of Trues and False's).
    :param node: A node in the Bayesian Network.
    :param observations: The observations of the evidence variables in the network.
    :return:
    """
    return NotImplementedError


def main(bayes_net, observations):
    pass

if __name__ == '__main__':
    bn_one_path = 'bn1.json'
    observations_one_path = 'data1.csv'
    bn_two_path = 'bn2.json'
    observations_two_path = 'data2.csv'
    with open(bn_one_path, 'r') as fp:
        bayes_net_one = pd.read_csv(fp)
    with open(bn_two_path, 'r') as fp:
        bayes_net_two = pd.read_csv(fp)
    with open(observations_one_path, 'r') as fp:
        observations_one = pd.read_csv(fp)
    with open(observations_two_path, 'r') as fp:
        observations_two = pd.read_csv(fp)
    main(bayes_net=bayes_net_one, observations=observations_one)
