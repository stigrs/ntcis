# Copyright (c) 2021 Stig Rune Sellevag
#
# This file is distributed under the MIT License. See the accompanying file
# LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
# and conditions.

"""Provides a network topology model for critical infrastructure systems."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import operator


class NetworkTopology:
    """Class for representing network topologies for critical infrastructure
    systems."""

    def __init__(self, filename):
        self.graph = None
        self.load(filename)

    def load(self, filename):
        """Load network topology from CSV file with interdependency matrix."""
        df = pd.read_csv(filename, index_col=0)
        # need to make sure dependency is interpreted as j --> i
        self.graph = nx.from_pandas_adjacency(df.transpose(), create_using=nx.DiGraph)

    def node_degree_centrality(self, descending=True):
        """Compute degree centrality for the nodes."""
        degree = list(nx.degree_centrality(self.graph).values())
        degree = {n: d for n, d in zip(self.graph.nodes, degree)}
        degree = dict(sorted(degree.items(), key=operator.itemgetter(1), reverse=descending))
        return degree

    def node_in_degree_centrality(self, descending=True):
        """Compute in-degree centrality for the nodes."""
        degree = list(nx.in_degree_centrality(self.graph).values())
        degree = {n: d for n, d in zip(self.graph.nodes, degree)}
        degree = dict(sorted(degree.items(), key=operator.itemgetter(1), reverse=descending))
        return degree

    def node_out_degree_centrality(self, descending=True):
        """Compute out-degree centrality for the nodes."""
        degree = list(nx.out_degree_centrality(self.graph).values())
        degree = {n: d for n, d in zip(self.graph.nodes, degree)}
        degree = dict(sorted(degree.items(), key=operator.itemgetter(1), reverse=descending))
        return degree

    def betweenness_centrality(self, descending=True):
        """Compute betweenness centrality for the nodes."""
        betweenness = list(nx.betweenness_centrality(self.graph).values())
        betweenness = {n: d for n, d in zip(self.graph.nodes, betweenness)}
        betweenness = dict(sorted(betweenness.items(), key=operator.itemgetter(1), reverse=descending))
        return betweenness

    def closeness_centrality(self, outward=False, descending=True):
        """Compute closeness centrality for the nodes."""
        graph = self.graph
        if outward:
            graph = self.graph.reverse()
        closeness = list(nx.closeness_centrality(graph).values())
        closeness = {n: d for n, d in zip(self.graph.nodes, closeness)}
        closeness = dict(sorted(closeness.items(), key=operator.itemgetter(1), reverse=descending))
        return closeness

    def eigenvector_centrality(self, outward=False, descending=True):
        """Compute eigenvector centrality for the nodes."""
        graph = self.graph
        # compute out-edges eigenvector centrality (in-edges is default)
        if outward:
            graph = self.graph.reverse()
        centrality = list(nx.eigenvector_centrality(graph).values())
        centrality = {n: d for n, d in zip(self.graph.nodes, centrality)}
        centrality = dict(sorted(centrality.items(), key=operator.itemgetter(1), reverse=descending))
        return centrality

    def articulation_points(self):
        """Find the articulation points of the topology."""
        return list(nx.articulation_points(self.graph.to_undirected()))

    def plot(self, filename=None, figsize=(12, 12), node_size=5000, dpi=300, seed=None):
        """Plot network topology."""
        pos = nx.spring_layout(self.graph, seed=seed)

        el = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d["weight"] >= 0.05]
        es = [(u, v) for (u, v, d) in self.graph.edges(data=True) if d["weight"] < 0.05]

        fig, _ = plt.subplots(figsize=figsize)

        # Nodes:
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size)

        # Edges:
        nx.draw_networkx_edges(self.graph, pos, edgelist=el, width=4, node_size=node_size)
        nx.draw_networkx_edges(self.graph, pos, edgelist=es, width=1, node_size=node_size)

        # Labels:
        nx.draw_networkx_labels(self.graph, pos)

        fig.tight_layout()
        plt.axis("off")

        if filename:
            plt.savefig(filename, dpi=dpi)
