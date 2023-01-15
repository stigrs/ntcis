# Copyright (c) 2023 Stig Rune Sellevag
#
# This file is distributed under the MIT License. See the accompanying file
# LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
# and conditions.

"""Provides methods for analysing network topology of infrastructure grids."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import momepy
import operator
import pygeos
import copy


def largest_connected_component(graph):
    """Return largest connected component of the infrastructure grid."""
    return len(max(nx.connected_components(graph), key=len))


def efficiency(graph, weight=None):
    """Return efficiency of the infrastructure grid.

    Reference:
        Eq. (3) in Bellingeri, M., Bevacqua, D., Scotognella, F. et al. A comparative 
        analysis of link removal strategies in real complex weighted networks. 
        Sci Rep 10, 3911 (2020). https://doi.org/10.1038/s41598-020-60298-7
    """
    eff = []
    for subgraph in (graph.subgraph(cc).copy() for cc in nx.connected_components(graph)):
        n = subgraph.number_of_nodes()
        sum_dij = 0.0
        for i in subgraph.nodes:
            for j in subgraph.nodes:
                if i is not j:
                    dij = 1.0 / nx.shortest_path_length(subgraph, source=i, target=j, weight="weight")
                    sum_dij = sum_dij + dij
        eff.append(sum_dij / (n * (n - 1)))
    return sum(eff)


def degree_centrality(graph):
    """Compute degree centrality for the nodes."""
    degree = nx.degree_centrality(graph)
    sorted_degree = sorted(degree.items(), key=operator.itemgetter(1), reverse=False)
    return sorted_degree


def betweenness_centrality(graph, weight=None):
    """Compute betweenness centrality for the nodes."""
    betweenness = nx.betweenness_centrality(graph, weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=False)
    return sorted_betweenness


def current_flow_betweenness_centrality(graph, weight=None):
    """Compute current flow betweenness centrality for the nodes.
    
    If graph is not connected, compute current flow betweenness centrality for the nodes
    of the largest connected component."""
    if graph.is_connected():
        betweenness = nx.current_flow_betweenness_centrality(graph, weight=weight)
    else:
        lcc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(lcc).copy()
        betweenness = nx.current_flow_betweenness_centrality(subgraph, weight=weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=False)
    return sorted_betweenness


def edge_betweenness_centrality(graph, weight=None):
    """Compute betweenness centrality for the edges."""
    betweenness = nx.edge_betweenness_centrality(graph, weight=weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=False)
    return sorted_betweenness


def edge_current_flow_betweenness_centrality(graph, weight=None):
    """Compute current flow betweenness centrality for the edges.
    
    If graph is not connected, compute current flow betweenness centrality for the edges
    of the largest connected component."""
    if graph.is_connected():
        betweenness = nx.edge_current_flow_betweenness_centrality(graph, weight=weight)
    else:
        lcc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(lcc).copy()
        betweenness = nx.edge_current_flow_betweenness_centrality(subgraph, weight=weight)
    sorted_betweenness = sorted(betweenness.items(), key=operator.itemgetter(1), reverse=False)
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

    def __init__(self, filename, multigraph=False, explode=False, epsg=None, capacity=None):
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
        self.load(filename, multigraph, explode, capacity)
        if epsg:
            self.grid.to_crs(epsg)

    def load(self, filename, multigraph=False, explode=False, capacity=None):
        """Load geodata for infrastructure grid from file (e.g. GEOJSON format)."""
        self.grid = gpd.read_file(filename)
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

    def close_gaps(self, tolerance):
        """Close gaps in LineString geometry where it should be contiguous.
        Snaps both lines to a centroid of a gap in between."""

        # BSD 3-Clause License
        #
        # Copyright (c) 2020, Urban Grammar
        # All rights reserved.
        #
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are met:
        #
        # 1. Redistributions of source code must retain the above copyright notice, this
        #    list of conditions and the following disclaimer.
        #
        # 2. Redistributions in binary form must reproduce the above copyright notice,
        #    this list of conditions and the following disclaimer in the documentation
        #    and/or other materials provided with the distribution.
        #
        # 3. Neither the name of the copyright holder nor the names of its
        #    contributors may be used to endorse or promote products derived from
        #    this software without specific prior written permission.
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        geom = self.grid.geometry.values.data
        coords = pygeos.get_coordinates(geom)
        indices = pygeos.get_num_coordinates(geom)

        # Generate a list of start and end coordinates and create point geometries
        edges = [0]
        i = 0
        for ind in indices:
            ix = i + ind
            edges.append(ix - 1)
            edges.append(ix)
            i = ix
        edges = edges[:-1]
        points = pygeos.points(np.unique(coords[edges], axis=0))
        buffered = pygeos.buffer(points, tolerance)
        dissolved = pygeos.union_all(buffered)

        exploded = [
            pygeos.get_geometry(dissolved, i)
            for i in range(pygeos.get_num_geometries(dissolved))
        ]

        centroids = pygeos.centroid(exploded)
        self.grid.geometry.values.data = pygeos.snap(geom, pygeos.union_all(centroids), tolerance)
        self.graph = momepy.gdf_to_nx(self.grid, multigraph=self.__multigraph, directed=False)

    def largest_connected_component(self):
        """Return largest connected component."""
        return largest_connected_component(self.graph)

    def efficiency(self, weight=None):
        """Return the efficiency of the network."""
        return efficiency(self.graph, weight=weight)

    def degree_centrality(self):
        """Compute degree centrality for the nodes."""
        return degree_centrality(self.graph)

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

    def betweenness_centrality(self, weight=None):
        """Compute betweenness centrality for the nodes."""
        return betweenness_centrality(self.graph, weight)

    def print_betweenness_centrality(self, weight=None):
        """Print node betweenness centrality."""
        print("Node Betweenness Centrality (top ten):")
        print("-" * 50)
        print("{0:<35}\t{1}".format("Node", "Value"))
        print("-" * 50)
        i = 1
        for v, c in self.node_betweenness_centrality(weight):
            print("({0[0]:.6f}, {0[1]:.6f})\t\t{1:.8f}".format(v, c))
            if i >= 10:
                break
            i += 1
        print("-" * 50)

    def edge_betweenness_centrality(self, weight=None):
        """Compute betweenness centrality for the edges."""
        return edge_betweenness_centrality(self.graph, weight)

    def print_edge_betweenness_centrality(self, weight):
        """Print edge betweenness centrality."""
        print("Edge Betweenness Centrality (top ten):")
        print("-" * 90)
        print("{0:<35}\t{1:<35}\t{2}".format("Node A", "Node B", "Value"))
        print("-" * 90)
        i = 1
        for v, c in self.edge_betweenness_centrality(weight)
            print("({0[0]:.6f}, {0[1]:.6f})\t\t({1[0]:.6f}, {1[1]:.6f})\t\t{2:.8f}".format(v[0], v[1], c))
            if i >= 10:
                break
            i += 1
        print("-" * 90)

    def articulation_points(self):
        """Find the articulation points of the topology."""
        return articulation_points(self.graph)

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

    def articulation_point_targeted_attack(self, nattacks=1):
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
            nattacks = len(graph_attacked.nodes)

        lcc = [largest_connected_component(graph_attacked)]
        eff = [efficiency(graph_attacked)]
        nodes_attacked = []  # list with coordinates of attacked nodes

        for i in range(nattacks):
            graph_attacked.remove_node(ap[i])
            nodes_attacked.append(ap[i])
            lcc.append(largest_connected_component(graph_attacked))
            eff.append(efficiency(graph_attacked))

        return graph_attacked, nodes_attacked, lcc, eff

    def betweenness_centrality_attack(self, nattacks=1, weighted=True, centrality_method="shortest_path"):
        """Carry out iterative betweenness centrality targeted attack on nodes. 

        Arguments:
            nattacks: Number of attacks to be carried out
            weighted: If weighted is true, use weighted betweenness centrality.
            centrality_method: Use shortest_path or current_flow

        Reference:
            Petter Holme, Beom Jun Kim, Chang No Yoon, and Seung Kee Han
            Phys. Rev. E 65, 056109. https://arxiv.org/abs/cond-mat/0202410v1
        """
        if nattacks < 1:
            nattacks = 1
        if nattacks > len(self.graph.edges):
            nattacks = len(self.graph.edges)

        # work on a local copy of the topology
        graph_attacked = copy.deepcopy(self.graph)
        nodes_attacked = [0]  # list with coordinates of attacked nodes

        if weighted:
            # network functioning measure
            network_measure = [efficiency(graph_attacked)]
            weight = "weight"
        else:
            network_measure = [largest_connected_component(graph_attacked)]
            weight = None

        for _ in range(nattacks):
            if centrality_method == "current_flow":
                bc = current_flow_betweenness_centrality(graph_attacked, weight)
            else: # shortest path
                bc = betweenness_centrality(graph_attacked, weight)
            graph_attacked.remove_node(bc[0][0])
            nodes_attacked.append(bc[0][0])
            if weighted:
                network_measure.append(efficiency(graph_attacked))
            else:
                network_measure.append(largest_connected_component(graph_attacked))

        return graph_attacked, nodes_attacked, network_measure

    def edge_betweenness_centrality_attack(self, nattacks=1, weighted=True, centrality_method="shortest_path"):
        """Carry out iterative betweenness centrality targeted attack on edges.

        Arguments:
            nattacks: Number of attacks to be carried out
            weighted: If weighted is true, use weighted betweenness centrality.
            centrality_method: Use shortest_path or current_flow

        Reference:
            Bellingeri, M., Bevacqua, D., Scotognella, F. et al. A comparative analysis of 
            link removal strategies in real complex weighted networks. 
            Sci Rep 10, 3911 (2020). https://doi.org/10.1038/s41598-020-60298-7
        """
        if nattacks < 1:
            nattacks = 1
        if nattacks > len(self.graph.edges):
            nattacks = len(self.graph.edges)

        # work on a local copy of the topology
        graph_attacked = copy.deepcopy(self.graph)
        edges_attacked = [0]  # list with coordinates of attacked edges

        if weighted:
            # network functioning measure
            network_measure = [efficiency(graph_attacked)]
            weight = "weight"
        else:
            network_measure = [largest_connected_component(graph_attacked)]
            weight = None

        for _ in range(nattacks):
            if centrality_method == "current_flow":
                bc = edge_current_flow_betweenness_centrality(graph_attacked, weight)
            else:
                bc = edge_betweenness_centrality(graph_attacked, weight)
            graph_attacked.remove_edge(bc[0][0][0], bc[0][0][1])
            edges_attacked.append(bc[0][0])
            if weighted:
                network_measure.append(efficiency(graph_attacked))
            else:
                network_measure.append(largest_connected_component(graph_attacked))

        return graph_attacked, edges_attacked, network_measure

    def plot(self, filename=None, figsize=(12, 12), dpi=300, add_basemap=False):
        """Plot original infrastructure grid."""
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
        """Plot graph of infrastructure topology."""
        plot_topology(self.graph, filename, figsize, node_size, dpi)
