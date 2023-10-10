#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author:
Created Time:
'''


from HegLDP import Heg
from RABV import *
from degreeBased import *
from functions import *


file_facebook = 'dataset/facebook_combined.txt'
sizeFacebook = 4039
file_enron = 'dataset/Email-Enron.txt'
sizeEnron = 36692
file_astro = 'dataset/CA-AstroPh-transform.txt'
sizeAstro = 18772
file_santa = 'dataset/santa.txt'
sizeSanta = 16216



if __name__ == '__main__':
    pubPercent = 0.2
    epsilon = 0.5

    Heg = Heg(filepath=file_facebook)
    pubNodeList = Heg.public_node_select(highPercent=pubPercent, pubPercent=pubPercent)
    graphMatrix, genDegs = Heg.diffusion(pubNodeList=pubNodeList, epsilon=epsilon)
    genGraph = nx.from_numpy_array(graphMatrix)


    oriDegdis = Deg_distr_gen(Heg.G)[1:]
    degDis = Deg_distr_fromDegs(genDegs)[1:]
    # degDis = Deg_distr_gen(genGraph)[1:]
    # print(oriDegdis[:20])
    # print(degDis[:20])


    realMid, realAver = med_aver_compute(Heg.G)
    # mid, aver = med_aver_compute(genGraph)
    mid, aver = med_aver_fromDegs(genDegs)
    # print('real and generated mid', realMid, mid)
    print('mid error: %.3f'%((realMid-mid)/realMid))
    # print('real and generated aver', realAver, aver)
    print('aver error: %.3f'%((realAver-aver)/realAver))


    lcc_ori = clustering_coefficient(Heg.G)
    gcc_ori = gcc_compute(Heg.G)
    # print('ground truth:', lcc_ori, gcc_ori)

    lcc_pe = clustering_coefficient(genGraph)
    gcc_pe = gcc_compute(genGraph)
    print('gen lcc error: %.3f'%(abs(lcc_ori-lcc_pe)/lcc_ori))
    print('gen gcc error: %.3f'%(abs(gcc_ori-gcc_pe)/gcc_ori))


    realResu = louvain_clustering(Heg.G, sizeFacebook)
    genResu = louvain_clustering(genGraph, sizeFacebook)
    reaLabel = label_gen(realResu, sizeFacebook)
    genLabel = label_gen(genResu, sizeFacebook)

    realModu = modularity_compute(Heg.G, realResu)
    genModu = modularity_compute(genGraph, genResu)
    print('modularity RE error: %.3f'%(abs(realModu-genModu)/realModu))


    ari = ari_compute(reaLabel, genLabel)
    ami = ami_compute(reaLabel, genLabel)
    print('ari and ami are:', ari, ami)

    line_3d(origin_degrees=oriDegdis, eps1_degrees=degDis)

    realShortest = shortest_path_length(Heg.G)
    shortest = shortest_path_length(genGraph)
    print('shortest path error: %.3f'%(abs((realShortest-shortest)/realShortest)))
