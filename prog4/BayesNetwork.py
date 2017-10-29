"""
BayesNetwork.py
An implementation of a Bayesian Network for Programming Assignment Four.
"""

import pandas as pd
import json
import itertools

__author__ = "Chris Campell"
__version__ = "10/28/2017"


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


def is_independent(node, bayes_net):
    """
    is_independent: Returns True if the node is an independent node in the Bayesian Network.
    :param node: The node for which to determine dependency.
    :param bayes_net: The topology of the Bayesian Network.
    :return boolean: True if the node is independent as specified in the topology of the Bayesian Network;
        False otherwise.
    """
    for parent, child_list in bayes_net.items():
        if node in child_list:
            # The node is a child of another node, it is dependent.
            return False
    # The node is not a child of another node, it is independent.
    return True


def get_dependencies(node, bayes_net):
    """
    get_dependencies: Returns the nodes that the provided node is conditionally dependent upon.
    :param node: The node to calculate dependencies for.
    :param bayes_net: The topology of the Bayesian Network.
    :return dependencies: A list of nodes that the provided node is dependent upon. Returns None if the provided node
        is independent.
    """
    if is_independent(node=node, bayes_net=bayes_net):
        # If the node is independent, it has no dependencies:
        return None
    dependencies = []
    for parent, child_list in bayes_net.items():
        if node in child_list:
            # The node is a child of another node, it is dependent.
            dependencies.append(parent)
    return dependencies


def build_probability_table(node, bayes_net, observations, probability_tables=None, dependencies=None):
    if dependencies is None or dependencies is False:
        # Calculate the probability of the independent variable given the observations:
        num_true = observations[node].value_counts()[True]
        total_num_obs = len(observations[node])
        key = 'P(%s=True)' % node
        probability_true = (num_true / total_num_obs)
        probability_tables[key] = probability_true
        key = 'P(%s=False)' % node
        probability_tables[key] = 1 - probability_true
    else:
        # The variable is conditionally dependent, construct the CPT:
        query = 'P(' + node + "|"
        for dependency in dependencies:
            query = query + dependency + ","
        query = query[0:-1]
        query += ')'
        # Construct the CPT for the query:
        observation_subset_cols = []
        observation_subset_cols.append(node)
        for dependency in dependencies:
            observation_subset_cols.append(dependency)
        # Subset the observations by the dependent variables.
        observation_subset = observations[observation_subset_cols]
        # Create a truth table:
        truth_table = list(itertools.product([False, True], repeat=len(observation_subset_cols)))
        for tuple in truth_table:
            df_query_with_node = ''
            df_query_without_node = ''
            human_readable_df_query = ''
            for i in range(len(tuple)):
                if i == 0:
                    human_readable_df_query = human_readable_df_query + 'P(%s=%s|' %(observation_subset_cols[i], tuple[i])
                else:
                    human_readable_df_query = human_readable_df_query + '%s=%s,' %(observation_subset_cols[i], tuple[i])
                    df_query_without_node = df_query_without_node + "%s == %s & " % (observation_subset_cols[i], tuple[i])
                df_query_with_node = df_query_with_node + "%s == %s & " %(observation_subset_cols[i], tuple[i])
            human_readable_df_query = human_readable_df_query[0:-1] + ')'
            df_query_with_node = df_query_with_node[0:-3]
            df_query_without_node = df_query_without_node[0:-3]
            # Query the observation_subset:
            num_observed = observation_subset.query(df_query_with_node).count()[0]
            num_total_subset = observation_subset.query(df_query_without_node).count()[0]
            probability_tables[human_readable_df_query] = (num_observed / num_total_subset)
    return probability_tables


def main(bayes_net, observations):
    """ build the conditional probability tables """
    prob_tables = {}
    for node in bayes_net:
        if is_independent(node, bayes_net):
            # The node has no parent, it is independent.
            if node not in prob_tables:
                # The node is not already in the probability tables.
                prob_tables = build_probability_table(node=node, bayes_net=bayes_net, observations=observations , probability_tables=prob_tables)
        else:
            # The node is the child of another node, it is dependent upon its parent.
            if node not in prob_tables:
                # The node is not already in the probability tables.
                # Get the nodes that the current node is conditionally dependent upon:
                dependencies = get_dependencies(node, bayes_net)
                # Build the probability table for this node:
                prob_tables = build_probability_table(node=node, bayes_net=bayes_net, observations=observations,
                                                      dependencies=dependencies, probability_tables=prob_tables)
    pass

if __name__ == '__main__':
    bn_one_path = 'bn1.json'
    observations_one_path = 'data1.csv'
    bn_two_path = 'bn2.json'
    observations_two_path = 'data2.csv'
    with open(bn_one_path, 'r') as fp:
        bayes_net_one_with_spaces = json.load(fp=fp)
    with open(bn_two_path, 'r') as fp:
        bayes_net_two_with_spaces = json.load(fp=fp)
    with open(observations_one_path, 'r') as fp:
        observations_one = pd.read_csv(fp)
    with open(observations_two_path, 'r') as fp:
        observations_two = pd.read_csv(fp)
    # Strip the spaces from the observations column headers:
    observations_one.columns = [x.replace(' ', '') for x in observations_one.columns]
    observations_two.columns = [x.replace(' ', '') for x in observations_two.columns]
    # Strip spaces from bayes network topology for consistency:
    bayes_net_one = {}
    for node, dependencies in bayes_net_one_with_spaces.items():
        bayes_net_one[node.replace(' ', '')] = [dependent.replace(' ', '') for dependent in dependencies]
    bayes_net_two = {}
    for node, dependencies in bayes_net_two_with_spaces.items():
        bayes_net_two[node.replace(' ', '')] = [dependent.replace(' ', '') for dependent in dependencies]
    main(bayes_net=bayes_net_one, observations=observations_one)
