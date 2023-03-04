# Copyright (c) 2023 Stig Rune Sellevag
#
# This file is distributed under the MIT License. See the accompanying file
# LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
# and conditions.

"""Provides methods for analysing network topology of infrastructure grids."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx
import random as rd
import momepy
import operator
import copy
import pathlib
from shapely import wkt


def largest_connected_component(graph):
    """Return largest connected component of the infrastructure grid."""
    return len(max(nx.connected_components(graph), key=len))


def largest_connected_component_subgraph(graph):
    """Return largest connected component as a subgraph."""
    lcc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(lcc).copy()

def efficiency(graph, weight=None):
    """Return global efficiency of the infrastructure grid.

    Reference:
        - Latora, V., and Marchiori, M. (2001). Efficient behavior of
          small-world networks. Physical Review Letters 87.
        - Latora, V., and Marchiori, M. (2003). Economic small-world behavior
          in weighted networks. Eur Phys J B 32, 249-263.
        - Bellingeri, M., Bevacqua, D., Scotognella, F. et al. A comparative 
          analysis of link removal strategies in real complex weighted networks. 
          Sci Rep 10, 3911 (2020). https://doi.org/10.1038/s41598-020-60298-7
    """
    n = graph.number_of_nodes()
    if n < 2:
        eff = 0
    else:
        inv_d = []
        for node in graph:
            if weight is None:
                dij = nx.single_source_shortest_path_length(graph, node)
            else:
                dij = nx.single_source_dijkstra_path_length(graph, node, weight=weight)
            inv_dij = [1/d for d in dij.values() if d != 0]
            inv_d.extend(inv_dij)
        eff = sum(inv_d) / (n * (n - 1))
    return eff


def degree_centrality(graph, weight=None):
    """Compute degree centrality for the nodes."""
    # weight is not used, but included to have the same API for centrality methods.
    degree = nx.degree_centrality(graph) 
    sorted_degree = sorted(degree.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_degree


def betweenness_centrality(graph, weight=None):
    """Compute betweenness centrality for the nodes."""
    betweenness = nx.betweenness_centrality(graph, weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_betweenness


def edge_betweenness_centrality(graph, weight=None):
    """Compute betweenness centrality for the edges."""
    betweenness = nx.edge_betweenness_centrality(graph, weight=weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_betweenness

def current_flow_betweenness_centrality(graph, weight=None):
    """Compute current flow betweenness centrality for the nodes.
    
    If graph is not connected, compute current flow betweenness centrality for the nodes
    of the largest connected component."""
    if nx.is_connected(graph):
        betweenness = nx.current_flow_betweenness_centrality(graph, weight=weight)
    else:
        lcc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(lcc).copy()
        betweenness = nx.current_flow_betweenness_centrality(subgraph, weight=weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_betweenness


def edge_current_flow_betweenness_centrality(graph, weight=None):
    """Compute current flow betweenness centrality for the edges.
    
    If graph is not connected, compute current flow betweenness centrality for the nodes
    of the largest connected component."""
    if nx.is_connected(graph):
        betweenness = nx.edge_current_flow_betweenness_centrality(graph, weight=weight)
    else:
        lcc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(lcc).copy()
        betweenness = nx.edge_current_flow_betweenness_centrality(subgraph, weight=weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_betweenness


def articulation_points(graph):
    """Find the articulation points of the topology."""
    return list(nx.articulation_points(graph.to_undirected()))


def plot_topology(graph, filename=None, figsize=(12, 12), node_size=5, dpi=300):
    """Plot infrastructure topology."""
    pos = {n: [n[0], n[1]] for n in list(graph.nodes)}
    _, _ = plt.subplots(figsize=figsize)
    nx.draw(graph, pos, node_size=node_size)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi)


class Infrastructure:
    """Class for representing network topology of infrastructure grids."""

    def __init__(self, filename, multigraph=False, explode=False, capacity=None, epsg=None):
        """Initialise infrastructure.

        Arguments:
            filename: name of file with geodata for the infrastructure grid
            multigraph: true if multigraph should be created
            explode: true if multilinestrings should be expanded
            epsg: transform to EPSG code
            capacity: string specifying the attribute with infrastructure capacities for the edges
        """
        self.graph = None
        self.grid = None
        self.__multigraph = multigraph
        self.load(filename, multigraph, explode, capacity, epsg)

    def load(self, filename, multigraph=False, explode=False, capacity=None, epsg=None):
        """Load geodata for infrastructure grid from file (e.g. GEOJSON format)."""
        if pathlib.Path(filename).suffix == ".csv":
            df = pd.read_csv(filename)
            df["geometry"] = df["geometry"].apply(wkt.loads)
            self.grid = gpd.GeoDataFrame(df, crs=epsg)
        else: # assume it is a format geopandas can read directly
            self.grid = gpd.read_file(filename)
            if epsg:
                self.grid.to_crs(epsg)
        self.__multigraph = multigraph
        if explode:
            self.grid = self.grid.explode()
        if capacity:
            # avoid division by zero, NaN or Inf
            self.grid[capacity] = self.grid[capacity].replace(to_replace=np.inf, value=np.finfo(float).max)
            self.grid[capacity] = self.grid[capacity].replace(to_replace=np.nan, value=np.finfo(float).eps)
            self.grid[capacity] = self.grid[capacity].replace(to_replace=0.0, value=np.finfo(float).eps)
            # shortest path weights are calculated as the resiprocal of the capacity
            self.grid["weight"] = 1.0 / self.grid[capacity] 
        self.graph = momepy.gdf_to_nx(self.grid, multigraph=multigraph, directed=False)

    def get_graph(self):
        """Return a deep copy of the graph."""
        return copy.deepcopy(self.graph) 

    def remove_false_nodes(self):
        """Clean topology of existing LineString geometry by removal of nodes of degree 2."""
        self.grid = momepy.remove_false_nodes(self.grid)
        self.graph = momepy.gdf_to_nx(self.grid, multigraph=self.__multigraph, directed=False)

    def close_gaps(self, tolerance):
        """Close gaps in LineString geometry where it should be contiguous.
        Snaps both lines to a centroid of a gap in between."""
        self.grid.geometry = momepy.close_gaps(self.grid, tolerance)
        self.graph = momepy.gdf_to_nx(self.grid, multigraph=self.__multigraph, directed=False)

    def extend_lines(self, tolerance):
        """Extends unjoined ends of LineString segments to join with other segments within a set tolerance."""
        self.grid = momepy.extend_lines(self.grid, tolerance)
        self.graph = momepy.gdf_to_nx(self.grid, multigraph=self.__multigraph, directed=False)

    def largest_connected_component(self):
        """Return largest connected component."""
        return largest_connected_component(self.graph)

    def largest_connected_component_subgraph(self):
        """Return largest connected component as subgraph."""
        return largest_connected_component_subgraph(self.graph)

    def efficiency(self, weight=None):
        """Return the efficiency of the network."""
        return efficiency(self.graph, weight=weight)

    def degree_centrality(self):
        """Compute degree centrality for the nodes."""
        return degree_centrality(self.graph)

    def betweenness_centrality(self, weight=None):
        """Compute betweenness centrality for the nodes."""
        return betweenness_centrality(self.graph, weight)

    def edge_betweenness_centrality(self, weight=None):
        """Compute betweenness centrality for the edges."""
        return edge_betweenness_centrality(self.graph, weight)

    def current_flow_betweenness_centrality(self, weight=None):
        """Compute current flow betweenness centrality for the nodes."""
        return current_flow_betweenness_centrality(self.graph, weight=None) 

    def edge_current_flow_betweenness_centrality(self, weight=None):
        """Compute current flow betweenness centrality for the edges."""
        return edge_current_flow_betweenness_centrality(self.graph, weight=None) 

    def articulation_points(self):
        """Find the articulation points of the topology."""
        return articulation_points(self.graph)

    def node_iterative_centrality_attack(self, nattacks=1, weight=None, centrality_method=betweenness_centrality):
        """Carry out iterative targeted attack on nodes. 

        Arguments:
            nattacks: Number of attacks to be carried out
            weight: If weight is not none, use weighted centrality and efficiency measures
            centrality_method: Measure used for assessing the centrality of the nodes

        Reference:
            Petter Holme, Beom Jun Kim, Chang No Yoon, and Seung Kee Han
            Phys. Rev. E 65, 056109. https://arxiv.org/abs/cond-mat/0202410v1
        """
        if nattacks < 1:
            nattacks = 1
        if nattacks > len(self.graph.edges):
            nattacks = len(self.graph.edges)

        graph_attacked = self.get_graph() # work on a local copy of the topology
        nodes_attacked = [0]  # list with coordinates of attacked nodes

        lcc = [largest_connected_component(graph_attacked)]
        eff = [efficiency(graph_attacked, weight)]

        weight_centrality = weight
        if centrality_method == betweenness_centrality:
            weight_centrality = None # use unweighted

        for _ in range(nattacks):
            bc = centrality_method(graph_attacked, weight_centrality) 
            graph_attacked.remove_node(bc[0][0])
            nodes_attacked.append(bc[0][0])
            lcc.append(largest_connected_component(graph_attacked))
            eff.append(efficiency(graph_attacked, weight))

        return graph_attacked, nodes_attacked, lcc, eff

    def edge_iterative_centrality_attack(self, nattacks=1, weight=None, centrality_method=edge_betweenness_centrality):
        """Carry out iterative targeted attack on edges.

        Arguments:
            nattacks: Number of attacks to be carried out
            weight: If weight is not none, use weighted centrality and efficiency measures
            centrality_method: Measure used for assessing the centrality of the nodes

        Reference:
            Bellingeri, M., Bevacqua, D., Scotognella, F. et al. A comparative analysis of 
            link removal strategies in real complex weighted networks. 
            Sci Rep 10, 3911 (2020). https://doi.org/10.1038/s41598-020-60298-7
        """
        if nattacks < 1:
            nattacks = 1
        if nattacks > len(self.graph.edges):
            nattacks = len(self.graph.edges)

        graph_attacked = self.get_graph() # work on a local copy of the topology
        edges_attacked = [0]  # list with coordinates of attacked edges

        lcc = [largest_connected_component(graph_attacked)]
        eff = [efficiency(graph_attacked, weight)]

        for _ in range(nattacks):
            bc = centrality_method(graph_attacked, weight)
            graph_attacked.remove_edge(bc[0][0][0], bc[0][0][1])
            edges_attacked.append(bc[0][0])
            lcc.append(largest_connected_component(graph_attacked))
            eff.append(efficiency(graph_attacked, weight))

        return graph_attacked, edges_attacked, lcc, eff

    def articulation_point_targeted_attack(self, nattacks=1, weight=None):
        """Carry out brute-force articulation point-targeted attack.

        Arguments:
            nattacks: Number of attacks to be carried out

        Reference:
            Tian, L., Bashan, A., Shi, DN. et al. Articulation points in complex networks.
            Nat Commun 8, 14223 (2017). https://doi.org/10.1038/ncomms14223
        """
        graph_attacked = copy.deepcopy(self.graph)
        ap = articulation_points(graph_attacked)

        if nattacks < 1:
            nattacks = 1
        if nattacks > len(graph_attacked.nodes):
            nattacks = len(graph_attacked.nodes, weight)

        lcc = [largest_connected_component(graph_attacked)]
        eff = [efficiency(graph_attacked)]
        nodes_attacked = []  # list with coordinates of attacked nodes

        for i in range(nattacks):
            graph_attacked.remove_node(ap[i])
            nodes_attacked.append(ap[i])
            lcc.append(largest_connected_component(graph_attacked))
            eff.append(efficiency(graph_attacked, weight))

        return graph_attacked, nodes_attacked, lcc, eff

    def random_attack(self, nattacks=1, weight=None):
        """Carry out random attack on nodes.

        Arguments:
            nattacks: Number of attacks to be carried out
            weighted: If weighted is not none, use weighted efficiency measure
        """
        if nattacks < 1:
            nattacks = 1
        if nattacks > len(self.graph.edges):
            nattacks = len(self.graph.edges)

        graph_attacked = copy.deepcopy(self.graph) # work on a local copy of the topology
        nodes_attacked = [0]  # list with coordinates of attacked nodes

        lcc = [largest_connected_component(graph_attacked)]
        eff = [efficiency(graph_attacked, weight)]

        for _ in range(nattacks):
            node = rd.sample(list(graph_attacked.nodes), 1)
            graph_attacked.remove_node(node[0])
            nodes_attacked.append(node[0])
            lcc.append(largest_connected_component(graph_attacked))
            eff.append(efficiency(graph_attacked, weight))

        return graph_attacked, nodes_attacked, lcc, eff

    def edge_random_attack(self, nattacks=1, weight=None):
        """Carry out random attack on edges.

        Arguments:
            nattacks: Number of attacks to be carried out
            weighted: If weighted is not none, use weighted efficiency measure
        """
        if nattacks < 1:
            nattacks = 1
        if nattacks > len(self.graph.edges):
            nattacks = len(self.graph.edges)

        graph_attacked = copy.deepcopy(self.graph) # work on a local copy of the topology
        edges_attacked = [0]  # list with coordinates of attacked edges

        lcc = [largest_connected_component(graph_attacked)]
        eff = [efficiency(graph_attacked, weight)]

        for _ in range(nattacks):
            edge = rd.sample(list(graph_attacked.edges), 1)
            graph_attacked.remove_edge(edge[0][0], edge[0][1])
            edges_attacked.append(edge[0])
            lcc.append(largest_connected_component(graph_attacked))
            eff.append(efficiency(graph_attacked, weight))

        return graph_attacked, edges_attacked, lcc, eff

    def print_degree_centrality(self):
        """Print node degree centrality."""
        print("Node Degree Centrality (top ten):")
        print("-" * 50)
        print("{0:<35}\t{1}".format("Node", "Value"))
        print("-" * 50)
        i = 1
        for v, c in self.degree_centrality():
            print("({0[0]:.6f}, {0[1]:.6f})\t\t{1:.8f}".format(v, c))
            if i >= 10:
                break
            i += 1
        print("-" * 50)

    def print_betweenness_centrality(self, weight=None):
        """Print node betweenness centrality."""
        print("Node Betweenness Centrality (top ten):")
        print("-" * 50)
        print("{0:<35}\t{1}".format("Node", "Value"))
        print("-" * 50)
        i = 1
        for v, c in self.betweenness_centrality(weight):
            print("({0[0]:.6f}, {0[1]:.6f})\t\t{1:.8f}".format(v, c))
            if i >= 10:
                break
            i += 1
        print("-" * 50)

    def print_edge_betweenness_centrality(self, weight):
        """Print edge betweenness centrality."""
        print("Edge Betweenness Centrality (top ten):")
        print("-" * 90)
        print("{0:<35}\t{1:<35}\t{2}".format("Node A", "Node B", "Value"))
        print("-" * 90)
        i = 1
        for v, c in self.edge_betweenness_centrality(weight):
            print("({0[0]:.6f}, {0[1]:.6f})\t\t({1[0]:.6f}, {1[1]:.6f})\t\t{2:.8f}".format(v[0], v[1], c))
            if i >= 10:
                break
            i += 1
        print("-" * 90)

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

    def plot(self, filename=None, figsize=(12, 12), dpi=300, xlabel="East", ylabel="North", add_basemap=False,
             provider=ctx.providers.OpenStreetMap.Mapnik, **kwargs):
        """Plot original infrastructure grid."""
        _, ax = plt.subplots(figsize=figsize)
        self.grid.plot(ax=ax, **kwargs)
        if add_basemap:
            ctx.add_basemap(ax, crs=self.grid.crs, source=provider)
        plt.tight_layout()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if filename:
            plt.savefig(filename, dpi=dpi)
        return ax

    def plot_topology(self, filename=None, figsize=(12, 12), node_size=5, dpi=300):
        """Plot graph of infrastructure topology."""
        plot_topology(self.graph, filename, figsize, node_size, dpi)
