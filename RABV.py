#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author:
Created Time:
'''
import numpy as np
import networkx as nx



def file_read(filepath):
    f = open(filepath, 'r')
    edges = []
    for line in f.readlines():
        line = line.strip().split()
        edges.append((int(line[0]), int(line[1])))
    return edges


class RrGraph(object):
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
        # construct the edge list for each node
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        matt = np.zeros((self.size, self.size))

        edges = list(self.G.edges())
        for i in edges:
            matt[i[0], i[1]] = 1
            matt[i[1], i[0]] = 1

        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(matt-inverse_matrix).astype(int)
        return pertMatt

    # def deg_distri_gen(self, epsilon_high, epsilon_mid, epsilon_low, newOrder):
    #     pLow = np.exp(epsilon_high) / (1 + np.exp(epsilon_high))
    #     pMid = np.exp(epsilon_mid) / (1 + np.exp(epsilon_mid))
    #     pHigh = np.exp(epsilon_low) / (1 + np.exp(epsilon_low))
    #     nodesLow = newOrder[int((self.perHigh+self.perMid) * self.size):]
    #     nodesMid = newOrder[int(self.perHigh * self.size):int((self.perHigh + self.perMid) * self.size)]
    #     nodesHigh = newOrder[:int(self.perHigh * self.size)]
    #     degList = list()
    #     for i in nodesHigh:
    #         deg = self.G.degree(i)
    #         noisyDeg = int(deg*pHigh+(self.size-deg)*(1-pHigh))
    #         degList.append(noisyDeg)
    #     for i in nodesMid:
    #         deg = self.G.degree(i)
    #         noisyDeg = int(deg*pMid+(self.size-deg)*(1-pMid))
    #         degList.append(noisyDeg)
    #     for i in nodesLow:
    #         deg = self.G.degree(i)
    #         noisyDeg = int(deg*pLow+(self.size-deg)*(1-pLow))
    #         degList.append(noisyDeg)
    #     maxDeg = max(degList)
    #     degDistri = np.zeros(maxDeg + 1)
    #     for i in range(len(degDistri)):
    #         degDistri[i] = degList.count(i)
    #     return degDistri