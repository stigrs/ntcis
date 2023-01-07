# Copyright (c) 2023 Stig Rune Sellevag
#
# This file is distributed under the MIT License. See the accompanying file
# LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
# and conditions.

"""Provides methods for analysing network topology of powergrids."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import momepy
import operator
import contextily as ctx


class Powergrid:
    """Class for representing network topologies of powergrids."""

    def __init__(self, filename, multigraph=True, explode=False, epsg=None):
        self.graph = None
        self.grid = None
        self.load(filename, multigraph, explode)
        if epsg:
            self.grid.to_crs(epsg)

    def load(self, filename, multigraph=True, explode=False):
        """Load powergrid from GEOJSON file."""
        self.grid = gpd.read_file(filename)
        if explode:
            self.grid = self.grid.explode()
        self.graph = momepy.gdf_to_nx(
            self.grid, multigraph=multigraph, directed=False)

    def node_degree_centrality(self, descending=True):
        """Compute degree centrality for the nodes."""
        degree = nx.degree_centrality(self.graph)
        sorted_degree = sorted(
            degree.items(), key=operator.itemgetter(1), reverse=descending)
        return sorted_degree

    def print_node_degree_centrality(self, descending=True):
        """Print node degree centrality."""
        print("Node Degree Centrality (top ten):")
        print("-" * 50)
        print("{0:<35}\t{1}".format("Node", "Value"))
        print("-" * 50)
        i = 1
        for v, c in self.node_degree_centrality(descending):
            print("({0[0]:.6f}, {0[1]:.6f})\t\t{1:.8f}".format(v, c))
            if i >= 10:
                break
            i += 1
        print("-" * 50)

    def node_betweenness_centrality(self, descending=True):
        """Compute betweenness centrality for the nodes."""
        betweenness = nx.betweenness_centrality(self.graph)
        sorted_betweenness = sorted(
            betweenness.items(), key=operator.itemgetter(1), reverse=descending)
        return sorted_betweenness

    def print_node_betweenness_centrality(self, descending=True):
        """Print node betweenness centrality."""
        print("Node Betweenness Centrality (top ten):")
        print("-" * 50)
        print("{0:<35}\t{1}".format("Node", "Value"))
        print("-" * 50)
        i = 1
        for v, c in self.node_betweenness_centrality(descending):
            print("({0[0]:.6f}, {0[1]:.6f})\t\t{1:.8f}".format(v, c))
            if i >= 10:
                break
            i += 1
        print("-" * 50)

    def edge_betweenness_centrality(self, descending=True):
        """Compute betweenness centrality for the edges."""
        betweenness = nx.edge_betweenness_centrality(self.graph)
        sorted_betweenness = sorted(
            betweenness.items(), key=operator.itemgetter(1), reverse=descending)
        return sorted_betweenness

    def print_edge_betweenness_centrality(self, descending=True):
        """Print edge betweenness centrality."""
        print("Edge Betweenness Centrality (top ten):")
        print("-" * 90)
        print("{0:<35}\t{1:<35}\t{2}".format("Node A", "Node B", "Value"))
        print("-" * 90)
        i = 1
        for v, c in self.edge_betweenness_centrality(descending):
            print("({0[0]:.6f}, {0[1]:.6f})\t\t({1[0]:.6f}, {1[1]:.6f})\t\t{2:.8f}".format(
                v[0], v[1], c))
            if i >= 10:
                break
            i += 1
        print("-" * 90)

    def articulation_points(self):
        """Find the articulation points of the topology."""
        return list(nx.articulation_points(self.graph.to_undirected()))

    def articulation_point_targeted_attack(self, nattacks=1, filename=None, figsize=(12, 12), node_size=5, dpi=300):
        """Carry out articulation point-targeted attack."""
        graph_attacked = self.graph
        ap = list(nx.articulation_points(graph_attacked.to_undirected()))

        if nattacks < 1:
            nattacks = 1
        if nattacks > len(ap):
            nattacks = len(ap)

        for i in range(nattacks):
            graph_attacked.remove_node(ap[i])

        pos = {n: [n[0], n[1]] for n in list(graph_attacked.nodes)}
        _, _ = plt.subplots(figsize=figsize)
        nx.draw(graph_attacked, pos, node_size=5)
        if filename:
            plt.savefig(filename, dpi=dpi)

        return graph_attacked

    def print_articulation_points(self):
        """Print articulation points of the topology."""
        print("Articulation Points (top ten):")
        print("-" * 31)
        i = 1
        for point in self.articulation_points():
            print("({0[0]:.6f}, {0[1]:.6f})".format(point))
            if i >= 10:
                break
            i += 1
        print("-" * 31)

    def plot(self, filename=None, figsize=(12, 12), dpi=300, add_basemap=False):
        """Plot original grid."""
        _, ax = plt.subplots(figsize=figsize)
        self.grid.plot(ax=ax)
        if add_basemap:
            ctx.add_basemap(ax, crs=self.grid.crs)
        plt.tight_layout()
        plt.ylabel("North")
        plt.xlabel("East")
        if filename:
            plt.savefig(filename, dpi=dpi)

    def plot_topology(self, filename=None, figsize=(12, 12), node_size=5, dpi=300):
        """Plot powergrid topology."""
        pos = {n: [n[0], n[1]] for n in list(self.graph.nodes)}
        _, _ = plt.subplots(figsize=figsize)
        nx.draw(self.graph, pos, node_size=node_size)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=dpi)
