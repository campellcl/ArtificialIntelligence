"""
BayesNetwork.py
An implementation of a Bayesian Network for Programming Assignment Four.
"""

import pandas as pd
import json
import itertools

__author__ = "Chris Campell"
__version__ = "10/28/2017"

prob_tables = None


class BayesNetwork:
    bayes_network = None
    prob_tables = None

    def __init__(self, bayes_network):
        self.bayes_network = bayes_network

    def get_parents(self, node):
        parents = []
        for parent, child_list in self.bayes_network.items():
            if node in child_list:
                parents.append(parent)
        return parents

    def is_independent(self, node):
        """
        is_independent: Returns True if the node is an independent node in the Bayesian Network.
        :param node: The node for which to determine dependency.
        :param bayes_net: The topology of the Bayesian Network.
        :return boolean: True if the node is independent as specified in the topology of the Bayesian Network;
            False otherwise.
        """
        for parent, child_list in self.bayes_network.items():
            if node in child_list:
                # The node is a child of another node, it is dependent.
                return False
        # The node is not a child of another node, it is independent.
        return True

    def get_dependencies(self, node):
        """
        get_dependencies: Returns the nodes that the provided node is conditionally dependent upon.
        :param node: The node to calculate dependencies for.
        :param bayes_net: The topology of the Bayesian Network.
        :return dependencies: A list of nodes that the provided node is dependent upon. Returns None if the provided node
            is independent.
        """
        if self.is_independent(node=node):
            # If the node is independent, it has no dependencies:
            return None
        dependencies = []
        for parent, child_list in self.bayes_network.items():
            if node in child_list:
                # The node is a child of another node, it is dependent.
                dependencies.append(parent)
        return dependencies

    def build_probability_table(self, node, observations, probability_tables=None, dependencies=None):
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

    def enumeration_ask(self,X,e,cpts):
        """
        enumeration_ask: Returns a probability distribution over X.
        :param X: The query variable for which probabilities are to be inferred. For example HighCarValue.
        :param e: The observed values for variables E. For example e:{GoodEngine:True, WorkingAC=False}
        :param bn: A Bayesian Network with variables {X} union E union Y (hidden variables).
        :return norm_dist_x: A normalized probability distribution over the query variable X.
        """
        # Sort vars in topological order:
        edge_list = []
        for parent, child_list in self.bayes_network.items():
            for child in child_list:
                edge_list.append([parent, child])
        vars = sort_direct_acyclic_graph(edge_list=edge_list)
        # x_i can only be True or False, no need for a loop:
        # Build keys based on evidence variable and query:
        query_true_key = 'P(%s=True|' % X
        query_false_key = 'P(%s=False|' % X
        for evidence, assignment in e.items():
            query_true_key = query_true_key + '%s=%s,' % (evidence, assignment)
            query_false_key = query_false_key + '%s=%s,' % (evidence, assignment)
        query_true_key = query_true_key[0:-1] + ')'
        query_false_key = query_false_key[0:-1] + ')'
        Q = {query_true_key: None, query_false_key: None}
        e_x_i = e.copy()
        e_x_i[X] = True
        joint_prob_query = 'P(%s=%s|' % (X,True)
        for evidence, observation in e.items():
            joint_prob_query = joint_prob_query + '%s=%s,' % (evidence, observation)
        joint_prob_query = joint_prob_query[0:-1] + ')'
        Q[joint_prob_query] = self.enumerate_all(vars=vars, e=e_x_i, cpts=cpts)
        e_x_i[X] = False
        joint_prob_query = 'P(%s=%s|' % (X,False)
        for evidence, observation in e.items():
            joint_prob_query = joint_prob_query + '%s=%s,' % (evidence, observation)
        joint_prob_query = joint_prob_query[0:-1] + ')'
        Q[joint_prob_query] = self.enumerate_all(vars=vars, e=e_x_i, cpts=cpts)
        # Return the normalization of Q (take each value of Q and divide it by the sum of all the values).
        q_norm = Q.copy()
        for query, probabilty in Q.items():
            q_norm[query] = probabilty / sum(Q.values())
        return q_norm

    def enumerate_all(self, vars, e, cpts):
        """
        enumerate_all: Helper method for enumeration_ask, computes the joint probability distribution of the provided
            variables, given a set of observations.
        :param vars: A list of input variables for which to construct the joint from.
        :param e: A list of observations of the input variables which influence the joint distribution.
        :param cpts: The conditional probability tables for the Bayesian Network.
        :return joint_prob: The joint distribution of the provided 'vars' with the supplied observations 'e'.
        """
        joint_prob = None
        if not vars:
            # The provided list of evidence variables is empty, the probability is one.
            joint_prob = 1.0
            return joint_prob
        # The provided list of evidence variables is not empty:
        # Initially the query variable Y is just the first in the list of query variables.
        Y = vars[0]
        # The rest of the variables are everything but Y:
        rest = vars.copy()
        rest.remove(Y)
        # Does the query variable Y have a known value in the observed evidence variables?
        if Y in e.keys():
            # The query variable Y=y as given in the observed evidence variables:
            y = e[Y]
            # Get the parents of the query variable:
            parents = self.get_parents(Y)
            if parents:
                query = 'P(%s=%s|' % (Y,y)
                for evidence, assignment in e.items():
                    if evidence in parents:
                        query = query + '%s=%s,' % (evidence, assignment)
                query = query[0:-1] + ')'
            else:
                query = 'P(%s=%s)' % (Y,y)
            prob_Y_is_y = cpts[query]
            return prob_Y_is_y * self.enumerate_all(rest, e, cpts)
        else:
            # The query variable Y has no observed evidence (y):
            # Build a query for the CPT
            # Sum up over every possible value for Y=y given Y's parents in e:
            # If Y = High Car Value then we want sum(P(HCV|GE,AC)) for all values in the truth table.
            # For every possible value that Y can be (e.g. Y=y_i):
            conditional_probs = {}
            y_i = True
            e_y_i = e.copy()
            e_y_i[Y] = y_i
            # Get the parents of the query variable:
            parents = self.get_parents(Y)
            if parents:
                query = 'P(%s=%s|' % (Y,y_i)
                for evidence, assignment in e.items():
                    if evidence in parents:
                        query = query + '%s=%s,' % (evidence, assignment)
                query = query[0:-1] + ')'
            else:
                query = 'P(%s=%s)' % (Y,y_i)
            prob_Y_is_y = cpts[query]
            conditional_probs[query] = prob_Y_is_y * self.enumerate_all(rest,e_y_i,cpts)
            y_i = False
            if parents:
                query = 'P(%s=%s|' % (Y,y_i)
                for evidence, assignment in e.items():
                    if evidence in parents:
                        query = query + '%s=%s,' % (evidence, assignment)
                query = query[0:-1] + ')'
            else:
                query = 'P(%s=%s)' % (Y,y_i)
            e_y_i[Y] = y_i
            prob_Y_is_y = cpts[query]
            conditional_probs[query] = prob_Y_is_y * self.enumerate_all(rest,e_y_i,cpts)
            return conditional_probs


def sort_direct_acyclic_graph(edge_list):
    """
    sot_direct_acyclic_graph: Sorts the input in topological order.
    :source: http://code.activestate.com/recipes/578406-topological-sorting-again/
    :param edge_list: A list of edges [['a','b'],['b','c']] implies a->b, and b->c.
    :return node_list: The nodes sorted topographically.
    """
    # edge_set is consumed, need a copy
    edge_set = set([tuple(i) for i in edge_list])
    # node_list will contain the ordered nodes
    node_list = list()
    #  source_set is the set of nodes with no incoming edges
    node_from_list, node_to_list = zip(* edge_set)
    source_set = set(node_from_list) - set(node_to_list)
    while len(source_set) != 0 :
        # pop node_from off source_set and insert it in node_list
        node_from = source_set.pop()
        node_list.append(node_from)
        # find nodes which have a common edge with node_from
        from_selection = [e for e in edge_set if e[0] == node_from]
        for edge in from_selection :
            # remove the edge from the graph
            node_to = edge[1]
            edge_set.discard(edge)
            # if node_to don't have any remaining incoming edge :
            to_selection = [e for e in edge_set if e[1] == node_to]
            if len(to_selection) == 0 :
                # add node_to to source_set
                source_set.add(node_to)
    if len(edge_set) != 0 :
        raise IndexError # not a direct acyclic graph
    else:
        return node_list


def train_bayes_network_classifier(bayes_network_topology, observations):
    """ build the conditional probability tables """
    prob_tables = {}
    bns = BayesNetwork(bayes_network_topology)
    for node in bns.bayes_network:
        if bns.is_independent(node):
            # The node has no parent, it is independent.
            if node not in prob_tables:
                # The node is not already in the probability tables.
                bns.prob_tables = bns.build_probability_table(node=node, observations=observations,
                                                              probability_tables=prob_tables)
        else:
            # The node is the child of another node, it is dependent upon its parent.
            if node not in prob_tables:
                # The node is not already in the probability tables.
                # Get the nodes that the current node is conditionally dependent upon:
                dependencies = bns.get_dependencies(node)
                # Build the probability table for this node:
                bns.prob_tables = bns.build_probability_table(node=node, observations=observations,
                                                              dependencies=dependencies, probability_tables=prob_tables)
    return bns
    prob_tables = bns.prob_tables.copy()
    # return enumeration_ask(X=X, e=e, bn=bns, cpts=prob_tables, bns=bns)
    X = 'HighCarValue'
    e = {'WorkingAirConditioner': True, 'GoodEngine': True}
    e = {'GoodEngine': True, 'WorkingAirConditioner': False}
    e = {'GoodEngine': False, 'WorkingAirConditioner': True}
    e = {'GoodEngine': False, 'WorkingAirConditioner': False}



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
    bayes_net_topology_one = {}
    for node, dependencies in bayes_net_one_with_spaces.items():
        bayes_net_topology_one[node.replace(' ', '')] = [dependent.replace(' ', '') for dependent in dependencies]
    bayes_net_topology_two = {}
    for node, dependencies in bayes_net_two_with_spaces.items():
        bayes_net_topology_two[node.replace(' ', '')] = [dependent.replace(' ', '') for dependent in dependencies]
    bayes_net_classifier = train_bayes_network_classifier(bayes_network_topology=bayes_net_topology_one,
                                                          observations=observations_one)
    # Sort vars in topological order:
    edge_list = []
    for parent, child_list in bayes_net_classifier.bayes_network.items():
        for child in child_list:
            edge_list.append([parent, child])
    # Assign topological ordering to Bayes Network Instance:
    vars = sort_direct_acyclic_graph(edge_list=edge_list)
    bayes_net_classifier.vars = vars
    # Prompt user for input and answer any queries:
    keyboard_interrupt = False
    while not keyboard_interrupt:
        user_query_verbatim = input("Enter a Query for the Network of the form P(Query={True,False}|{Evidence}):")
        # user_query_var = user_query[user_query.index('(')+1:user_query.index('=')]
        user_query = user_query_verbatim[2:-1]
        query_type = None
        # Determine the type of query:
        if '|' in user_query_verbatim:
            query_type = 'conditional'
        else:
            if user_query_verbatim.count('=') == 1:
                query_type = 'singular'
            else:
                query_type = 'joint'
        # Extract the variables from the query:
        user_query_vars = {}
        user_evidence_vars = {}
        if query_type == 'conditional':
            query_vars = user_query[0:user_query.find('|')]
            user_evidence = user_query[user_query.find('|')+1:]
            user_query_list= query_vars.split(',')
            user_query_list = [query.split('=') for query in user_query_list]
            for var in user_query_list:
                user_query_vars[var[0]] = var[1] == 'True'
            user_evidence_list = user_evidence.split(',')
            user_evidence_list = [obs.split('=') for obs in user_evidence_list]
            for var in user_evidence_list:
                user_evidence_vars[var[0]] = var[1] == 'True'
            print("Enumeration-Ask P(Query|Evidence): %s"
                  % bayes_net_classifier.enumeration_ask(X=user_query_vars,
                                                         e=user_evidence_vars, cpts=bayes_net_classifier.prob_tables))
        elif query_type == 'joint':
            # There is no query variable:
            user_query_vars = None
            user_evidence_list = user_query.split(',')
            user_evidence_list = [obs.split('=') for obs in user_evidence_list]
            for var in user_evidence_list:
                user_evidence_vars[var[0]] = var[1] == 'True'
            print("Enumerate-All P({Evidence={True,False}}): %s"
                  % bayes_net_classifier.enumerate_all(vars=bayes_net_classifier.vars, e=user_evidence_vars,
                                                       cpts=bayes_net_classifier.prob_tables))
        elif query_type == 'singular':
            # There is no evidence variable:
            user_evidence_vars = None
            user_query_list = user_query.split('=')
            user_query_vars[user_query_list[0]] = user_query_list[1] == 'True'
            print("Enumerate-All P(Evidence={True,False}): %s"
                  % bayes_net_classifier.enumerate_all(vars=bayes_net_classifier.vars,
                                                       e={}, cpts=bayes_net_classifier.prob_tables))
        else:
            print("The input query %s is malformed. Expected a query of type {joint,conditional,singular}"
                  % user_query_verbatim)
            exit(-1)


