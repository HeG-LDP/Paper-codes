import numpy as np
import networkx as nx
import community as community_louvain
from lib.hrg import Dendrogram
from lib.matrics import Matric
from sklearn import metrics
import pandas as pd
import copy
from collections import Counter
import math
import random
import matplotlib.pyplot as plt
from scipy.special import comb, perm


file_facebook = 'dataset/facebook_combined.txt'
sizeFacebook = 4039
file_enron = 'dataset/Email-Enron.txt'
sizeEnron = 36692
file_astro = 'dataset/CA-AstroPh-transform.txt'
sizeAstro = 18772
file_santa = 'dataset/santa.txt'
sizeSanta = 16216


def result_read(readPath):
    resu = list()
    f = open(readPath, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        resu.append(int(float(line)))
    return resu


def file_read(filepath):
    f = open(filepath, 'r')
    edges = []
    for line in f.readlines():
        line = line.strip().split()
        edges.append((int(line[0]), int(line[1])))
    return edges


class Hegg(object):
    def __init__(self, filepath):
        self.filepath = filepath
        data = file_read(filepath)
        G = nx.Graph()
        G.add_edges_from(data)
        self.G = G
        self.size = len(self.G.nodes())
        self.Degs = list()
        for i in range(self.size):
            self.Degs.append(self.G.degree(i))

    def public_node_select(self, highPercent):
        # degreeList = list(dict(self.G.degree()).values())  
        ids = np.array(list(dict(self.G.degree()).keys()))
        degrees = np.array(list(dict(self.G.degree()).values()))
        ind = np.argsort(ids)
        degreesFixed = degrees[ind]
        sortedInd = np.argsort(degreesFixed)[::-1]
        nodeInd = np.array(list(range(self.size)))
        highDegNodes = nodeInd[sortedInd]
        highNodesInd = highDegNodes[:int(self.size*highPercent)]
        # pubNodeList = np.random.choice(highNodesInd, size=int(self.size*pubPercent), replace=False)

        return highNodesInd


    def diffusion(self, pubNodeList, epsilon):
        # indList = list(np.range(self.size))
        # indListCop = copy.deepcopy(indList)
        count = self.size-len(pubNodeList)

        releasedDegs = [0 for i in range(self.size)]  #  Degree value of the node that has been released (perturbed degree value)
        for i in pubNodeList:           # Write degree and id of pub nodes to releasedDegs
            releasedDegs[i] = self.Degs[i]

        neiList = [[] for i in range(self.size)]        # neiList stores information about the neighboring nodes of all nodes
        for node in list(self.G.nodes()):
            neiList[node] = list(self.G.neighbors(node))

        releasedNodes = list()
        for i in pubNodeList:
            releasedNodes.append(i)

        neighborList = list()
        for i in releasedNodes:
            neighborList.extend(neiList[i])  # 1-round nodes

        # --------The graph is expressed using matrix and should form the initial pub graph.---------- 
        graphMatrix = np.zeros((self.size, self.size))
        for i in releasedNodes:
            for j in list(neiList[i]):
                graphMatrix[i, j] = 1
                graphMatrix[j, i] = 1

        while count:
            dkList = [{} for i in range(self.size)]

            # Removes duplicates from neighboring nodes of different RELEASED nodes, ensuring that each node appears only once, and the neighborList also represents all nodes of the current hop
            neighborList = list(set(neighborList))
            # Removing a node that has already been released can also be accomplished by subtracting it from the set.
            neighborList = [x for x in neighborList if x not in releasedNodes]

            preReleaseNodes = copy.deepcopy(releasedNodes)
            releasedNodes.extend(neighborList)      # Counts the current hop's nodes as releasedNodes as well
            # Count the dk information of the current node and the previous release node, and update the noise level value of the current node at the same time.
            for node in neighborList:   # Here's a loop that processes 1 hop of data
                neighbors = neiList[node]
                neighDegList = list()

                for i in neighbors:
                    if i in preReleaseNodes:
                        neighDegList.append(releasedDegs[i])

                # Count the frequency of elements in neighDegList and calculate the DK value
                Cou = dict(Counter(neighDegList))   
                dk = list(Cou.values())            
                dk_degs = list(Cou.keys())         
                UnLinkDeg = self.Degs[node]-sum(dk)

                pertDk = np.array(dk)+np.random.laplace(loc=0, scale=(1/epsilon), size=np.array(dk).shape)
                pertDk = np.round(pertDk).astype(int)

                # This code block is used to correct the dk values after perturbation to avoid too many redundant edges
                for j in range(len(pertDk)):
                    if pertDk[j] <= 0:
                        pertDk[j] = 0

                dkDict = dict(zip(list(dk_degs), list(pertDk)))
                dkList[node] = dkDict

                pertUnLink = UnLinkDeg+np.round(np.random.laplace(loc=0, scale=(1/epsilon)))
                if pertUnLink < 0:
                    pertUnLink = 0
                releasedDegs[node] = sum(pertDk)+pertUnLink     


            # Count the dk values of the current node and the current hop node
            for node in neighborList:  
                neighbors = neiList[node]
                neighDegList = list()

                for i in neighbors:
                    if i in neighborList:
                        neighDegList.append(releasedDegs[i])

                # Count the frequency of elements in neighDegList and calculate the DK value
                Cou = dict(Counter(neighDegList))  
                dk = list(Cou.values())  # e.g., dk=[2， 1， 1， 3...]
                dk_degs = list(Cou.keys())  

                pertDk = np.array(dk) + np.random.laplace(loc=0, scale=(1 / epsilon), size=np.array(dk).shape)
                pertDk = np.round(pertDk).astype(int)

                # for j in range(len(pertDk)):
                #     # if pertDk[j] <= 0:
                #     if pertDk[j] <= 4:
                #         pertDk[j] = 0

                dkDict = dict(zip(list(dk_degs), list(pertDk)))
                dkList[node].update(dkDict)

            # --------------------Current hop's graph generation, code start--------------------
            for node in neighborList:       
                dk = list(dkList[node].values())
                dk_degs = list(dkList[node].keys())

                for j in range(len(dk)):   
                    noNum = dk[j]  

                    noDeg = dk_degs[j] 
                    # The current version of the code does not have a situation where noDeg does not exist in the releasedDegs
                    if noDeg not in releasedDegs:
                        continue
                    candidateNodes = np.where(np.array(releasedDegs) == noDeg)[0]

                    if noNum <= 5:
                        continue

                    if len(candidateNodes) <= noNum:
                        for k in candidateNodes:
                            graphMatrix[node, k] = 1
                            graphMatrix[k, node] = 1
                    else:
                        selected = np.random.choice(candidateNodes, noNum, replace=False)
                        # update matrix
                        for k in selected:
                            graphMatrix[node, k] = 1
                            graphMatrix[k, node] = 1
            # ------------------code end--------------------

            count -= len(neighborList)      

            # After the node operation for the current hop, update the node list for the next hop
            neighborList = list()
            for i in releasedNodes:
                neighborList.extend(neiList[i])  # add next-round nodes
                # The neighborList should have the releasedNodes removed.
            neighborList = list(set(neighborList)-set(releasedNodes))

            if not neighborList:
                # print(count)
                break

            print('finish')

        return graphMatrix, releasedDegs


def med_aver_compute(graph):

    medianDeg = list(dict(nx.degree(graph)).values())
    medianDeg.sort()
    medianDeg = medianDeg[int(len(graph.nodes())//2)]
    averageDeg = np.sum(list(dict(nx.degree(graph)).values()))/len(graph.nodes())
    return medianDeg, averageDeg


def med_aver_fromDegs(degs):

    medianDeg = degs
    medianDeg.sort()
    medianDeg = medianDeg[int(len(degs)//2)]
    averageDeg = np.sum(degs)/len(degs)
    return medianDeg, averageDeg


def clustering_coefficient(G):
    # average CC computation
    cc = nx.clustering(G)
    total = 0
    for i in range(len(cc)):
        total += cc[i]
    return total / len(cc)


def louvain_clustering(G, size):
    nodeList = list(G.nodes())
    for i in range(size):
        if i not in nodeList:
            G.add_node(i)
    partition = community_louvain.best_partition(G)
    resu = []
    for i in range(len(G.nodes())):
        resu.append([])
    for i in range(len(G.nodes())):
        resu[partition[i]].append(i)
    resu = [i for i in resu if i != []]
    return resu


def label_gen(re, n):
    label = np.zeros(n)
    for c in range(len(re)):
        for i in re[c]:
            label[i] = c
    return label


def aver_length_compute(g):
    size = len(g.nodes())
    pathLenMatrix = np.zeros((size, size))
    path_lengths = list(nx.shortest_path_length(g))    
    for i in range(size):
        ind = path_lengths[i][0]        
        endIndList = list(path_lengths[i][1].keys())
        lenList = np.array(list(path_lengths[i][1].values()))
        dataList = (1/lenList).round(3)
        for j in range(len(endIndList)):
            pathLenMatrix[ind][endIndList[j]] = dataList[j]
    averLen = 0
    for i in range(size):
        for j in range(i+1, size):
            averLen += pathLenMatrix[i, j]
    return 2*averLen/(size*(size-1))


def shortest_path_length(G):

    return nx.average_shortest_path_length(G)


def modularity_compute(G, resu):
    """

    :param G:
    :param resu: result of clustering
    :return:
    """
    # RE: |Q-Q'|/Q
    return nx.algorithms.community.modularity(G, resu)


def ari_compute(label1, label2):
    # label1 should be the ground truth
    return metrics.adjusted_rand_score(label1, label2)


def ami_compute(label1, label2):

    return metrics.adjusted_mutual_info_score(label1, label2)


def gcc_compute(G):

    return nx.transitivity(G)/3


def Deg_distr_gen(G):
    degreeList = list(dict(nx.degree(G)).values())
    maxDeg = max(degreeList)
    degDistri = np.zeros(maxDeg + 1)
    for i in range(len(degDistri)):
        degDistri[i] = degreeList.count(i)
    return degDistri


def Deg_distr_fromDegs(degreeList):
    maxDeg = max(degreeList)
    degDistri = np.zeros(maxDeg + 1)
    for i in range(len(degDistri)):
        degDistri[i] = degreeList.count(i)
    return degDistri


def matrix2degs(matrix):
    return list(np.sum(matrix, axis=1).astype(int))



if __name__ =='__main__':
    from RABV import *
    from degreeBased import *


    for eps in [0.5, 1, 3, 4, 5, 8]:

        print(eps)
        for ijs in [5, 4, 3, 2, 1]:
            pubPercent = 0.1*ijs
            epsilon = eps
            filepath = file_facebook
            size = sizeFacebook

            Heg = Hegg(filepath=filepath)
            # Rrg = RrGraph(filepath=filepath, epsilon=epsilon)

            realMid, realAver = med_aver_compute(Heg.G)

            rrMatrix = Rrg.pert()
            rrDegs = matrix2degs(rrMatrix)
            rrGraph = nx.from_numpy_array(rrMatrix)
            midRr, averRr = med_aver_fromDegs(rrDegs)
            print('RR mid error: %.3f' % ((realMid - midRr) / realMid))
            print('RR aver error: %.3f' % ((realAver - averRr) / realAver))
            lcc_ori = clustering_coefficient(Heg.G)
            gcc_ori = gcc_compute(Heg.G)
            gcc_rr = gcc_compute(rrGraph)
            lcc_rr = clustering_coefficient(rrGraph)
            print('RR gen lcc error: %.3f' % (abs(lcc_ori - lcc_rr) / lcc_ori))
            print('RR gen gcc error: %.3f' % (abs(gcc_ori - gcc_rr) / gcc_ori))
            realResu = louvain_clustering(Heg.G, size)
            reaLabel = label_gen(realResu, size)
            rrResu = louvain_clustering(rrGraph, size)
            rrLabel = label_gen(rrResu, size)
            realModu = modularity_compute(Heg.G, realResu)
            rrModu = modularity_compute(rrGraph, rrResu)
            print('RR modularity RE error: %.3f' % (abs(realModu - rrModu) / realModu))
            rrAri = ari_compute(reaLabel, rrLabel)
            rrAmi = ami_compute(reaLabel, rrLabel)
            print('RR ari and ami are:', rrAri, rrAmi)
            realShortest = aver_length_compute(Heg.G)
            rrShortest = aver_length_compute(rrGraph)
            print('RR shortest path error: %.3f' % (abs((realShortest - rrShortest) / realShortest)))

            Dgg = DggGraph(filepath=filepath, epsilon=epsilon)
            pubNodeList = Heg.public_node_select(highPercent=pubPercent)

            graphMatrix, genDegs = Heg.diffusion(pubNodeList=pubNodeList, epsilon=epsilon)

            pertDegs = Dgg.pert().astype(int)
            dggMatrix = Dgg.dgg_gen(Degree=pertDegs)
            rrMatrix = Rrg.pert()

            oriDegdis = Deg_distr_gen(Heg.G)[1:]
            degDis = Deg_distr_fromDegs(genDegs)[1:]
            rrDegs = matrix2degs(rrMatrix)
            dggDegs = matrix2degs(dggMatrix)
            rrDegDis = Deg_distr_fromDegs(rrDegs)[1:]
            dggDegDis = Deg_distr_fromDegs(dggDegs)[1:]

            genGraph = nx.from_numpy_array(graphMatrix)
            # rrGraph = nx.from_numpy_array(rrMatrix)
            # dggGraph = nx.from_numpy_array(dggMatrix)

            # realMid, realAver = med_aver_compute(Heg.G)
            mid, aver = med_aver_fromDegs(genDegs)
            # midRr, averRr = med_aver_fromDegs(rrDegs)
            # midDgg, averDgg = med_aver_fromDegs(dggDegs)



            lcc_ori = clustering_coefficient(Heg.G)
            gcc_ori = gcc_compute(Heg.G)
            lcc_pe = clustering_coefficient(genGraph)
            gcc_pe = gcc_compute(genGraph)
            lcc_dgg = clustering_coefficient(dggGraph)
            gcc_dgg = gcc_compute(dggGraph)
            gcc_rr = gcc_compute(rrGraph)
            lcc_rr = clustering_coefficient(rrGraph)

            realResu = louvain_clustering(Heg.G, size)
            genResu = louvain_clustering(genGraph, size)
            rrResu = louvain_clustering(rrGraph, size)
            dggResu = louvain_clustering(dggGraph, size)
            reaLabel = label_gen(realResu, size)
            genLabel = label_gen(genResu, size)
            rrLabel = label_gen(rrResu, size)
            dggLabel = label_gen(dggResu, size)

            realModu = modularity_compute(Heg.G, realResu)
            genModu = modularity_compute(genGraph, genResu)
            rrModu = modularity_compute(rrGraph, rrResu)
            dggModu = modularity_compute(dggGraph, dggResu)


            Ari = ari_compute(reaLabel, genLabel)
            Ami = ami_compute(reaLabel, genLabel)
            rrAri = ari_compute(reaLabel, rrLabel)
            rrAmi = ami_compute(reaLabel, rrLabel)
            dggAri = ari_compute(reaLabel, dggLabel)
            dggAmi = ami_compute(reaLabel, dggLabel)

            realShortest = aver_length_compute(Heg.G)
            Shortest = aver_length_compute(genGraph)
            rrShortest = aver_length_compute(rrGraph)
            dggShortest = aver_length_compute(dggGraph)










