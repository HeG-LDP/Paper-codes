#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time: 
'''
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn import metrics
import matplotlib.pyplot as plt


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


def line_3d(origin_degrees, eps1_degrees):
    # 线
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('degree (in log2)')
    ax.set_zlabel('count')

    ax.plot(xs=np.log2(np.array(range(len(origin_degrees)))), ys=np.ones(len(origin_degrees)) * 1, zs=origin_degrees,
            c="black", label='Ground truth',  linewidth=2.0)

    ax.plot(xs=np.log2(np.array(range(len(eps1_degrees)))), ys=np.ones(len(eps1_degrees)) * 2, zs=eps1_degrees,
            c="blue", label='Block-HRG', linewidth=2.0)

    # ax.plot(xs=np.array(range(len(origin_degrees))), ys=np.ones(len(origin_degrees)) * 1, zs=origin_degrees,
    #         c="black", label='Ground truth', linewidth=2.0)
    #
    # ax.plot(xs=np.array(range(len(eps1_degrees))), ys=np.ones(len(eps1_degrees)) * 2, zs=eps1_degrees,
    #         c="blue", label='Block-HRG', linewidth=2.0)


    # ax.plot(xs=np.log10(np.array(range(len(santa_ldpgen_degrees)))), ys=np.ones(len(santa_ldpgen_degrees)) * 3,
    #         zs=santa_ldpgen_degrees,
    #         c="red", label='LDPGen', linewidth=2.0)
    # ax.plot(xs=np.log10(np.array(range(len(santa_lf_degrees)))), ys=np.ones(len(santa_lf_degrees)) * 4, zs=santa_lf_degrees,
    #         c="orange", label='LF-GDPR', linewidth=2.0)
    # ax.plot(xs=np.log10(np.array(range(len(santa_rr_degrees)))), ys=np.ones(len(santa_rr_degrees)) * 5,
    #         zs=santa_rr_degrees,
    #         c="green", label='RR', linewidth=2.0)

    # ax.axes.yaxis.set_visible(False)
    ax.axes.yaxis.set_ticklabels([])

    # ax.legend(loc='lower right')
    ax.legend(bbox_to_anchor=(1.0, 0.90))
    plt.show()
