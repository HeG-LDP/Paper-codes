#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author:
Created Time:
'''

import numpy as np
import networkx as nx
import copy



def file_read(filepath):
    f = open(filepath, 'r')
    edges = []
    for line in f.readlines():
        line = line.strip().split()
        edges.append((int(line[0]), int(line[1])))
    return edges


class DggGraph(object):
    def __init__(self, filepath, epsilon):
        self.filepath = filepath
        data = file_read(filepath)
        G = nx.Graph()
        G.add_edges_from(data)
        self.G = G
        self.size = len(self.G.nodes())
        self.epsilon = epsilon

    def pert(self):
        """
        :return:
        """
        # degs = np.array(list(dict(nx.degree(self.G)).values()))
        degs = np.array(list(dict(self.G.degree()).values()))
        perturbedDegs = degs + np.random.laplace(loc=0, scale=(2/self.epsilon),
                                                 size=degs.shape)
        perturbedDegs= np.round(perturbedDegs)
        for i in range(len(perturbedDegs)):
            if perturbedDegs[i] < 0:
                perturbedDegs[i] = 0

        return perturbedDegs

    def dgg_gen(self, Degree):
        Size = self.size
        """
        Generating synthetic maps using degree sequences and first-order zero models ------------- upper limit is very low
        :param Degree:
        :return:
        """
        Graph_empty = np.zeros([Size, Size])
        deg = copy.deepcopy(Degree)

        for i in range(Size):
            total = np.sum(deg[i+1:])
            if not total:
                break
            remain_nodes = list(range(i+1, Size))
            prob = deg[i+1:]/np.sum(deg[i+1:])
            # Need to make sure deg[i] > number of non-zero elements in prob
            remain_num = Size-i-prob.tolist().count(0)
            if deg[i] > remain_num:
                deg[i] = remain_num-1
            try:
                connected_edges = np.random.choice(remain_nodes, size=(deg[i]), replace=False, p=prob)
            except ValueError:
                print(prob.tolist().count(0))
                print(remain_num)
                print(deg[i])
                print('-------------------')
            # connected_edges = np.random.choice(remain_nodes, size=(deg[i]), replace=False)
            for j in connected_edges:
                deg[j] -= 1
                Graph_empty[i, j] = 1
                
        for i in range(Size):
            for j in range(i+1, Size):
                Graph_empty[j, i] = Graph_empty[i, j]
        # dggGraph = nx.from_numpy_matrix(Graph_empty)
        return Graph_empty

























