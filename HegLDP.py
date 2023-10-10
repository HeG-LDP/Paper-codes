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
        # degreeList = list(dict(self.G.degree()).values())  # 这里需要确定index是否从0开始，是否按照顺序
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

    # def diffusion(self, pubNodeList, epsilon):
    #     # indList = list(np.range(self.size))
    #     # indListCop = copy.deepcopy(indList)
    #     count = self.size-len(pubNodeList)
    #
    #     releasedDegs = list()  # 已经release的node的度值（已扰动度值）
    #     ind4Degs = list()  # 配合releasedDegs使用，（度值对应的节点id）
    #     for i in pubNodeList:
    #         ind4Degs.append(i)
    #         releasedDegs.append(self.Degs[i])
    #
    #     neiList = [[] for i in range(self.size)]        # neiList存储所有节点的邻接节点信息
    #     # degList = list()        # degList存储所有节点的度值信息
    #     for node in list(self.G.nodes()):
    #         neiList[node] = list(self.G.neighbors(node))
    #         # neiList.append(list(self.G.neighbors(node)))
    #
    #     reN = list()
    #     for i in pubNodeList:
    #         reN.append(i)
    #     releasedNodes = list(set(reN))      # 当前还仅包括pub nodes
    #     neighborList = list()
    #     for i in releasedNodes:
    #         neighborList.extend(neiList[i])  # 1-round nodes
    #
    #     # --------利用matrix表达graph，且应该形成初始的pub graph---------- 这里假设graph的节点从0开始
    #     graphMatrix = np.zeros((self.size, self.size))
    #     for i in pubNodeList:
    #         for j in list(neiList[i]):
    #             graphMatrix[i, j] = 1
    #             graphMatrix[j, i] = 1
    #
    #     while count:
    #         pertDkList = list()
    #         pertDkInd = list()
    #         endNodeList = list()
    #         pertDegList = list()
    #
    #         neighborList = list(set(neighborList))
    #         # 去除已经released 节点， 也可以用 set相减实现
    #         neighborList = [x for x in neighborList if x not in releasedNodes]
    #
    #         # relNodes = copy.deepcopy(releasedNodes) # 先前发布的节点
    #         releasedNodes.extend(neighborList)      # 将当前hop的节点也计入releasedNodes
    #
    #         for node in neighborList:   # 这里的一个循环处理1个hop的数据，循环最后需要更新neighborList
    #             neighbors = neiList[node]
    #             neighDegList = list()
    #
    #             relNeiDegList = list()
    #             curNeiDegList = list()
    #
    #             for i in neighbors:
    #                 if i in releasedNodes:        # 如果i在非当前hop的releasedNodes中，则属于当前可以发布的信息
    #                     if i not in neighborList:   # -------待修----这里不能简单的利用Degs[i],应该考虑度值扰动----
    #                         neighDegList.append(self.Degs[i])   # 存储的是邻接先前released节点的度值，计算dk仍需统计
    #                         relNeiDegList.append(self.Degs[i])
    #                     elif i < node:              # 如果i属于当前hop，则为了避免重复提交，仅让排序靠后的节点统计一次dk值
    #                         neighDegList.append(self.Degs[i])
    #                         curNeiDegList.append(self.Degs[i])
    #
    #             # 统计neighDegList中元素频率，计算DK值
    #             Cou = dict(Counter(neighDegList))   # Cou是存储当前节点和各种度值节点的DK值，字典形式存储，未排序
    #             dk = list(Cou.values())             # 如 dk=[2， 1， 1， 3...]
    #             dk_degs = list(Cou.keys())          # 存储的是dk值对应的相连节点的度值, d=[xxxx, ....]
    #             UnLinkDeg = self.Degs[node]-sum(dk)
    #             neighDegList = list(set(neighDegList))
    #
    #             # 去除扰动
    #             pertDk = np.array(dk)+np.random.laplace(loc=0, scale=(1/epsilon), size=np.array(dk).shape)
    #             pertDk = np.round(pertDk).astype(int)
    #
    #             # 此代码块用于修正扰动后的dk值，避免出现过多冗余边的情况
    #             for j in range(len(pertDk)):
    #                 if pertDk[j] <= 4:
    #                     pertDk[j] = 0
    #             pertDkList.append(pertDk)
    #             endNodeList.append(dk_degs)         # 存储dk值对应的相连节点的度值
    #             pertUnLink = UnLinkDeg+np.round(np.random.laplace(loc=0, scale=(1/epsilon)))
    #             # pertUnLink = UnLinkDeg
    #             if pertUnLink < 0:
    #                 pertUnLink = 0
    #             pertDegList.append(sum(pertDk)+pertUnLink)
    #             pertDkInd.append(node)      # 记录当前节点的index
    #
    #         # --------------每一个hop应该生成一次当前的graph----------------
    #         releasedDegs.extend(pertDegList)    # 将当前hop的所有节点的度值纳入releasedDegs
    #         ind4Degs.extend(pertDkInd)          # 同步纳入当前hop所有节点的id，以和degs对应
    #
    #         # # 此代码块目的在于减少下面循环中的计算
    #         # relDeg = list(set(releasedDegs))
    #         # indsList = list()
    #         # for k in relDeg:
    #         #     indexes = np.where(np.array(releasedDegs) == k)[0]
    #         #     indsList.append(indexes)
    #
    #         for i in range(len(pertDkList)):
    #             curNode = neighborList[i]       # 确定当前节点
    #             dk = pertDkList[i]              # dk的取值
    #             dk_degs = endNodeList[i]        # dk对应的末端节点的度值
    #
    #             for j in range(len(dk)):
    #                 noNum = dk[j]           # 这里的dk代表的是已经扰动过的当前节点和已released节点间的dk值，noNum代表当前dk值
    #                 noDeg = dk_degs[j]      # 这里的dk_degs代表dk值对应的末端节点的度值，
    #
    #                 if noDeg not in releasedDegs:
    #                     print('a ha?')
    #                     continue
    #
    #                 # 从released nodes 中寻找度值为noDeg的所有节点，然后将当前节点随机连接到noNum个节点上
    #                 inds = np.where(np.array(releasedDegs) == noDeg)[0]
    #                 candidateNodes = np.array(ind4Degs)[inds]          # ind4Degs 是当前hop所有节点的id，似乎不应该这么用
    #
    #                 if len(candidateNodes) <= noNum:
    #                     for k in candidateNodes:
    #                         graphMatrix[curNode, k] = 1
    #                         graphMatrix[k, curNode] = 1
    #                 else:
    #                     selected = np.random.choice(candidateNodes, noNum, replace=False)
    #                     # 更新matrix
    #                     for k in selected:
    #                         graphMatrix[curNode, k] = 1
    #                         graphMatrix[k, curNode] = 1
    #         # ------------------当前hop的graph生成，code end--------------------
    #
    #         count -= len(neighborList)      # 更新循环条件
    #         # 当前hop的节点操作之后，更新下一hop的节点列表
    #         neighborList = list()
    #         for i in releasedNodes:
    #             neighborList.extend(neiList[i])  # add next-round nodes
    #         # neighborList = list(set(neighborList))
    #         # # 去除已经released 节点， 也可以用 set相减实现
    #         # neighborList = [x for x in neighborList if x not in releasedNodes]
    #         if not neighborList:
    #             print(count)
    #             break
    #         print('finish')
    #     return graphMatrix


    def diffusion(self, pubNodeList, epsilon):
        # indList = list(np.range(self.size))
        # indListCop = copy.deepcopy(indList)
        count = self.size-len(pubNodeList)

        releasedDegs = [0 for i in range(self.size)]  # 已经release的node的度值（已扰动度值）
        for i in pubNodeList:           # 将pub nodes的度值和id写入releasedDegs
            releasedDegs[i] = self.Degs[i]

        neiList = [[] for i in range(self.size)]        # neiList存储所有节点的邻接节点信息
        for node in list(self.G.nodes()):
            neiList[node] = list(self.G.neighbors(node))

        releasedNodes = list()
        for i in pubNodeList:
            releasedNodes.append(i)

        neighborList = list()
        for i in releasedNodes:
            neighborList.extend(neiList[i])  # 1-round nodes

        # --------利用matrix表达graph，且应该形成初始的pub graph---------- 这里假设graph的节点从0开始
        graphMatrix = np.zeros((self.size, self.size))
        for i in releasedNodes:
            for j in list(neiList[i]):
                graphMatrix[i, j] = 1
                graphMatrix[j, i] = 1

        while count:
            dkList = [{} for i in range(self.size)]

            # 去除不同已released node的相邻结点中重复的部分，确保每个节点只出现一次，neighborList也代表当前hop的所有节点
            neighborList = list(set(neighborList))
            # 去除已经released 节点， 也可以用 set相减实现
            neighborList = [x for x in neighborList if x not in releasedNodes]

            preReleaseNodes = copy.deepcopy(releasedNodes)
            releasedNodes.extend(neighborList)      # 将当前hop的节点也计入releasedNodes

            # 统计当前节点和先前release节点的dk信息，同时更新当前节点的含噪度值
            for node in neighborList:   # 这里的一个循环处理1个hop的数据，循环最后需要更新neighborList
                neighbors = neiList[node]
                neighDegList = list()

                for i in neighbors:
                    if i in preReleaseNodes:
                        neighDegList.append(releasedDegs[i])

                # 统计neighDegList中元素频率，计算DK值
                Cou = dict(Counter(neighDegList))   # Cou是存储当前节点和各种度值节点的DK值，字典形式存储，未排序
                dk = list(Cou.values())             # 如 dk=[2， 1， 1， 3...]
                dk_degs = list(Cou.keys())          # 存储的是dk值对应的相连节点的度值, d=[xxxx, ....]
                UnLinkDeg = self.Degs[node]-sum(dk)

                pertDk = np.array(dk)+np.random.laplace(loc=0, scale=(1/epsilon), size=np.array(dk).shape)
                pertDk = np.round(pertDk).astype(int)

                # 此代码块用于修正扰动后的dk值，避免出现过多冗余边的情况,
                for j in range(len(pertDk)):
                    if pertDk[j] <= 0:
                        pertDk[j] = 0

                dkDict = dict(zip(list(dk_degs), list(pertDk)))
                dkList[node] = dkDict

                pertUnLink = UnLinkDeg+np.round(np.random.laplace(loc=0, scale=(1/epsilon)))
                if pertUnLink < 0:
                    pertUnLink = 0
                releasedDegs[node] = sum(pertDk)+pertUnLink         # 更新当前节点的扰动度值

            # 统计当前节点和当前hop节点的dk值
            for node in neighborList:  # 这里的一个循环处理1个hop的数据，循环最后需要更新neighborLi
                neighbors = neiList[node]
                neighDegList = list()

                for i in neighbors:
                    if i in neighborList:
                        neighDegList.append(releasedDegs[i])

                # 统计neighDegList中元素频率，计算DK值
                Cou = dict(Counter(neighDegList))  # Cou是存储当前节点和各种度值节点的DK值，字典形式存储，未排序
                dk = list(Cou.values())  # 如 dk=[2， 1， 1， 3...]
                dk_degs = list(Cou.keys())  # 存储的是dk值对应的相连节点的度值, d=[xxxx, ....]

                pertDk = np.array(dk) + np.random.laplace(loc=0, scale=(1 / epsilon), size=np.array(dk).shape)
                pertDk = np.round(pertDk).astype(int)

                # 此代码块用于修正扰动后的dk值，避免出现过多冗余边的情况
                # for j in range(len(pertDk)):
                #     # if pertDk[j] <= 0:
                #     if pertDk[j] <= 4:
                #         pertDk[j] = 0

                dkDict = dict(zip(list(dk_degs), list(pertDk)))
                dkList[node].update(dkDict)

            # --------------------当前hop的graph生成，code start--------------------
            for node in neighborList:       # 对每一个节点操作
                dk = list(dkList[node].values())
                dk_degs = list(dkList[node].keys())

                for j in range(len(dk)):    # 对当前节点的每一个dk值操作
                    noNum = dk[j]  # 这里的dk代表的是已经扰动过的当前节点和已released节点间的dk值，noNum代表当前dk值

                    noDeg = dk_degs[j]  # 这里的dk_degs代表dk值对应的末端节点的度值，
                    # 当前代码版本中不会出现noDeg不存在于releasedDegs中的情况
                    if noDeg not in releasedDegs:
                        continue
                    candidateNodes = np.where(np.array(releasedDegs) == noDeg)[0]

                    # 此代码块用于修正扰动后的dk值，避免出现过多冗余边的情况
                    if noNum <= 5:
                        continue

                    if len(candidateNodes) <= noNum:
                        for k in candidateNodes:
                            graphMatrix[node, k] = 1
                            graphMatrix[k, node] = 1
                    else:
                        selected = np.random.choice(candidateNodes, noNum, replace=False)
                        # 更新matrix
                        for k in selected:
                            graphMatrix[node, k] = 1
                            graphMatrix[k, node] = 1
            # ------------------当前hop的graph生成，code end--------------------

            count -= len(neighborList)      # 更新循环条件

            # 当前hop的节点操作之后，更新下一hop的节点列表
            neighborList = list()
            for i in releasedNodes:
                neighborList.extend(neiList[i])  # add next-round nodes
                # neighborList中应该去除已经 releasedNodes
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
    path_lengths = list(nx.shortest_path_length(g))     # 需要确定图的ind是按照顺序从小到大的
    for i in range(size):
        ind = path_lengths[i][0]        # 当前节点的索引
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


def degree_dis_draw(oriDegdis, degDis):
    ind = np.arange(len(degDis))
    ind_ori = np.arange(len(oriDegdis))
    plt.plot(ind_ori, oriDegdis, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='RABV-only')
    plt.plot(ind, degDis, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='LDPGen')
    # plt.plot(ind, LFGDPR, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    # plt.plot(ind, RE, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='Wdt-SCAN')
    # plt.ylim(0, 1.0)
    # plt.xlim(1, 8)
    plt.show()
    return None


def matrix2degs(matrix):
    return list(np.sum(matrix, axis=1).astype(int))


def line_3d(origin_degrees, heg_degrees, rr_degrees, dgg_degrees):
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('degree (in log2)')
    ax.set_zlabel('count')

    ax.plot(xs=np.log2(np.array(range(len(heg_degrees)))), ys=np.ones(len(heg_degrees)) * 4, zs=heg_degrees,
            c="blue", label='HeG-LDP', linewidth=2.0)
    ax.plot(xs=np.log2(np.array(range(len(dgg_degrees)))), ys=np.ones(len(dgg_degrees)) * 3, zs=dgg_degrees,
            c="orange", label='DGG', linewidth=2.0)
    ax.plot(xs=np.log2(np.array(range(len(rr_degrees)))), ys=np.ones(len(rr_degrees)) * 2,
            zs=rr_degrees, c="red", label='RABV', linewidth=2.0)
    ax.plot(xs=np.log2(np.array(range(len(origin_degrees)))), ys=np.ones(len(origin_degrees)) * 1, zs=origin_degrees,
            c="black", label='Ground truth', linewidth=2.0)

    # ax.axes.yaxis.set_visible(False)
    ax.axes.yaxis.set_ticklabels([])

    # ax.legend(loc='lower right')
    ax.legend(bbox_to_anchor=(1.0, 0.90))
    plt.show()


if __name__ =='__main__':
    from RABV import *
    from degreeBased import *


    for eps in [0.5, 1, 3, 4, 5, 8]:

        print(eps)
        print('ssssssssssssssssssssssssssssssss ')
        for ijs in [5, 4, 3, 2, 1]:
            pubPercent = 0.1*ijs
            epsilon = eps
            filepath = file_facebook
            size = sizeFacebook

            Heg = Hegg(filepath=filepath)
            # Rrg = RrGraph(filepath=filepath, epsilon=epsilon)

            realMid, realAver = med_aver_compute(Heg.G)

            # rrMatrix = Rrg.pert()
            # rrDegs = matrix2degs(rrMatrix)
            # rrGraph = nx.from_numpy_array(rrMatrix)
            # midRr, averRr = med_aver_fromDegs(rrDegs)
            # print('RR mid error: %.3f' % ((realMid - midRr) / realMid))
            # print('RR aver error: %.3f' % ((realAver - averRr) / realAver))
            # lcc_ori = clustering_coefficient(Heg.G)
            # gcc_ori = gcc_compute(Heg.G)
            # gcc_rr = gcc_compute(rrGraph)
            # lcc_rr = clustering_coefficient(rrGraph)
            # print('RR gen lcc error: %.3f' % (abs(lcc_ori - lcc_rr) / lcc_ori))
            # print('RR gen gcc error: %.3f' % (abs(gcc_ori - gcc_rr) / gcc_ori))
            # realResu = louvain_clustering(Heg.G, size)
            # reaLabel = label_gen(realResu, size)
            # rrResu = louvain_clustering(rrGraph, size)
            # rrLabel = label_gen(rrResu, size)
            # realModu = modularity_compute(Heg.G, realResu)
            # rrModu = modularity_compute(rrGraph, rrResu)
            # print('RR modularity RE error: %.3f' % (abs(realModu - rrModu) / realModu))
            # rrAri = ari_compute(reaLabel, rrLabel)
            # rrAmi = ami_compute(reaLabel, rrLabel)
            # print('RR ari and ami are:', rrAri, rrAmi)
            # realShortest = aver_length_compute(Heg.G)
            # rrShortest = aver_length_compute(rrGraph)
            # print('RR shortest path error: %.3f' % (abs((realShortest - rrShortest) / realShortest)))

            # Dgg = DggGraph(filepath=filepath, epsilon=epsilon)
            pubNodeList = Heg.public_node_select(highPercent=pubPercent)

            graphMatrix, genDegs = Heg.diffusion(pubNodeList=pubNodeList, epsilon=epsilon)

            # pertDegs = Dgg.pert().astype(int)
            # dggMatrix = Dgg.dgg_gen(Degree=pertDegs)
            # rrMatrix = Rrg.pert()

            oriDegdis = Deg_distr_gen(Heg.G)[1:]
            # degDis = Deg_distr_fromDegs(genDegs)[1:]
            # rrDegs = matrix2degs(rrMatrix)
            # dggDegs = matrix2degs(dggMatrix)
            # rrDegDis = Deg_distr_fromDegs(rrDegs)[1:]
            # dggDegDis = Deg_distr_fromDegs(dggDegs)[1:]

            genGraph = nx.from_numpy_array(graphMatrix)
            # rrGraph = nx.from_numpy_array(rrMatrix)
            # dggGraph = nx.from_numpy_array(dggMatrix)

            # realMid, realAver = med_aver_compute(Heg.G)
            mid, aver = med_aver_fromDegs(genDegs)
            # midRr, averRr = med_aver_fromDegs(rrDegs)
            # midDgg, averDgg = med_aver_fromDegs(dggDegs)
            print('mid error: %.3f' % ((realMid - mid) / realMid))
            print('aver error: %.3f' % ((realAver - aver) / realAver))
            # print('RR mid error: %.3f' % ((realMid - midRr) / realMid))
            # print('RR aver error: %.3f' % ((realAver - averRr) / realAver))
            # print('Dgg mid error: %.3f' % ((realMid - midDgg) / realMid))
            # print('Dgg aver error: %.3f' % ((realAver - averDgg) / realAver))
            print('---------------------------------------------------')

            lcc_ori = clustering_coefficient(Heg.G)
            gcc_ori = gcc_compute(Heg.G)
            lcc_pe = clustering_coefficient(genGraph)
            gcc_pe = gcc_compute(genGraph)
            # lcc_dgg = clustering_coefficient(dggGraph)
            # gcc_dgg = gcc_compute(dggGraph)
            # gcc_rr = gcc_compute(rrGraph)
            # lcc_rr = clustering_coefficient(rrGraph)

            print('gen lcc error: %.3f' % (abs(lcc_ori - lcc_pe) / lcc_ori))
            print('gen gcc error: %.3f' % (abs(gcc_ori - gcc_pe) / gcc_ori))
            # print('RR gen lcc error: %.3f' % (abs(lcc_ori - lcc_rr) / lcc_ori))
            # print('RR gen gcc error: %.3f' % (abs(gcc_ori - gcc_rr) / gcc_ori))
            # print('DGG gen lcc error: %.3f' % (abs(lcc_ori - lcc_dgg) / lcc_ori))
            # print('DGG gen gcc error: %.3f' % (abs(gcc_ori - gcc_dgg) / gcc_ori))
            print('---------------------------------------------------')

            realResu = louvain_clustering(Heg.G, size)
            genResu = louvain_clustering(genGraph, size)
            # rrResu = louvain_clustering(rrGraph, size)
            # dggResu = louvain_clustering(dggGraph, size)
            reaLabel = label_gen(realResu, size)
            genLabel = label_gen(genResu, size)
            # rrLabel = label_gen(rrResu, size)
            # dggLabel = label_gen(dggResu, size)

            realModu = modularity_compute(Heg.G, realResu)
            genModu = modularity_compute(genGraph, genResu)
            # rrModu = modularity_compute(rrGraph, rrResu)
            # dggModu = modularity_compute(dggGraph, dggResu)
            print('modularity RE error: %.3f' % (abs(realModu - genModu) / realModu))
            # print('RR modularity RE error: %.3f' % (abs(realModu - rrModu) / realModu))
            # print('DGG modularity RE error: %.3f' % (abs(realModu - dggModu) / realModu))
            print('---------------------------------------------------')

            Ari = ari_compute(reaLabel, genLabel)
            Ami = ami_compute(reaLabel, genLabel)
            # rrAri = ari_compute(reaLabel, rrLabel)
            # rrAmi = ami_compute(reaLabel, rrLabel)
            # dggAri = ari_compute(reaLabel, dggLabel)
            # dggAmi = ami_compute(reaLabel, dggLabel)
            print('ari and ami are:', Ari, Ami)
            # print('RR ari and ami are:', rrAri, rrAmi)
            # print('DGG ari and ami are:', dggAri, dggAmi)
            print('---------------------------------------------------')

            realShortest = aver_length_compute(Heg.G)
            Shortest = aver_length_compute(genGraph)
            # rrShortest = aver_length_compute(rrGraph)
            # dggShortest = aver_length_compute(dggGraph)
            print('shortest path error: %.3f' % (abs((realShortest - Shortest) / realShortest)))
            # print('RR shortest path error: %.3f' % (abs((realShortest - rrShortest) / realShortest)))
            # print('DGG shortest path error: %.3f' % (abs((realShortest - dggShortest) / realShortest)))
            print('---------------------------------------------------')
            # rrShortest = aver_length_compute(rrGraph)
            # print('RR shortest path error: %.3f' % (abs((realShortest - rrShortest) / realShortest)))

    # line_3d(origin_degrees=oriDegdis, heg_degrees=degDis, rr_degrees=rrDegDis, dgg_degrees=dggDegDis)

    # ---------测试不同比例高度值节点的edge 占有量------------
    # Heg = Heg(filepath=file_facebook)
    # pubNodeList = Heg.public_node_select(highPercent=0.5, pubPercent=0.5)
    # count = 0
    # for i in pubNodeList:
    #     count += Heg.G.degree(i)
    #
    # print(0.5*count/88234)









