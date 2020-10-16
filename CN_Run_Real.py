#!/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import networkx as nx
import random
import copy
import sys
import os
import pickle
from CN_Basic import SFN
from operator import itemgetter, attrgetter
import CN_Run_MOEA_SFCN as moea

quick_find_msg = 'real network, C(i) is 0 1 0, this may be NSGA-2!!!'


DATASET = "USAir" # "AS" , "USAir"
# RUN LOG:
HOST_SIZE_FRAC = 0.15
P_MUTATION = 0.1
P_CROSSOVER = 0.3
PHASE1_NFFE = 0  # typical time:   5*10^4:1.09h
PHASE2_NFFE = 0  # fitness function evaluation:f1 + f2 # typical time:1.5*10^4:20.9h
POP_SIZE = 40
ALPHA_R = 0.1
PHASE2_MAINTAIN_MAX = 30
# PHASE 2 mutation method #"random" "degree" "percent"
PHASE2_MUTATION_METHOD = "random"
PHASE2_MUTATION_PERCENT = 0.7  # select the last ___  percent

saveStdOut = None
file_log = None


# return a graph #返回读取到的图
def pre_process(path):
    # EdgeList.values: np.array
    EdgeList = pd.read_csv(path, sep="\t", header=None).values
    # ascending sort, remove duplicate
    NodeList = sorted(list(set(EdgeList.flatten().tolist())))
    # print(type(NodeList), NodeList)
    EdgeList = EdgeList.tolist()  # [[from, to], [from, to],...]
    #print(type(EdgeList), len(EdgeList), EdgeList[:10])
    EdgeList = [[NodeList.index(l[0]), NodeList.index(l[1])]
                for l in EdgeList if (l[0] != l[1])]  # ignore the self loop
    # print(type(EdgeList), len(EdgeList), EdgeList[:10], ' index 88:', NodeList[88])
    total = 0
    for i in range(len(EdgeList)):
        for j in range(i+1, len(EdgeList)):
            if(EdgeList[i][0] == EdgeList[j][1] and EdgeList[i][1] == EdgeList[j][0]):
                total += 1
    print("[Repetition Edge] total", total)
    G = nx.Graph()
    # Graph  , create_using=nx.Graph
    G = nx.from_edgelist(EdgeList, create_using=G)
    # G = nx.convert_to_undirected(G) # No such function???
    G.name = "As level"
    print(nx.info(G))
    return G


def USAirReadEdgenNodeList(path):
    EdgeList = list()
    NodeList = set()
    RawData = open(path)
    for line in RawData.readlines():
        l_Line = line.strip().split()
        EdgeList.append([int(l_Line[0]), int(l_Line[1])])
        NodeList.add(int(l_Line[0]))
        NodeList.add(int(l_Line[1]))
    return EdgeList, sorted(list(NodeList))



# return a graph #返回读取到的图
def pre_process_USAirLines(path):
    # EdgeList.values: np.array
    EdgeList, NodeList = USAirReadEdgenNodeList(path) # [[from, to], [from, to],...]
    # print("[USAir] NodeList", type(NodeList), len(NodeList), NodeList)
    # print("[USAir] EdgeList", type(EdgeList), len(EdgeList), EdgeList[:10])
    EdgeList = [[NodeList.index(l[0]), NodeList.index(l[1])]
                for l in EdgeList if (l[0] != l[1])]  # ignore the self loop
    # print(type(EdgeList), len(EdgeList), EdgeList[:10], ' index 88:', NodeList[88])
    total = 0
    for i in range(len(EdgeList)):
        for j in range(i+1, len(EdgeList)):
            if(EdgeList[i][0] == EdgeList[j][1] and EdgeList[i][1] == EdgeList[j][0]):
                total += 1
    print("[Repetition Edge] total", total)
    print("EdgeList\n", EdgeList, '\nEdgeList END')
    G = nx.Graph()
    # Graph  , create_using=nx.Graph
    G = nx.from_edgelist(EdgeList, create_using=G)
    # G = nx.convert_to_undirected(G) # No such function???
    G.name = "US Air Lines"
    print(nx.info(G))
    return G


def log_to_file(time_id, G):
    global saveStdOut
    global file_log
    saveStdOut = sys.stdout
    HOST_SIZE = int(G.number_of_nodes() * HOST_SIZE_FRAC)
    path = "Real_"+str(G.number_of_nodes())+"_"+str(HOST_SIZE)+"_pop"+str(
        POP_SIZE)+"_"+PHASE2_MUTATION_METHOD+"_"+str(PHASE1_NFFE)+"_"+str(PHASE2_NFFE)  # 文件夹目录
    if(PHASE2_MUTATION_METHOD == "percent"):
        path = path + "_" + str(int(PHASE2_MUTATION_PERCENT*100))
    path = os.path.join(os.path.abspath('.'), path)  # 绝对路径 + 相对路径
    folder = os.path.exists(path)  # 判断是否存在
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder ", path, " has been created  ---\n")
    else:
        print("---  folder ", path, " exits  ---\n")

    file_log = open((os.path.join(path, str(time_id)+".log")), 'w')  # 绝对路径 + time id为文件名
    sys.stdout = file_log
    file_G = open((os.path.join(path,str(time_id)+"G.data")), 'wb+')
    pickle.dump(G, file_G)
    file_G.close()


def cancel_log():
    sys.stdout = saveStdOut
    file_log.close()


def main():
    start_time = time.time()
    start_process_t = time.process_time()

    random.seed(int(start_time) % 100001651)
    np.random.seed(int(start_time) % 2154937333)

    ### Preparation Work ###
    path = os.path.join(os.path.abspath("."), "data")
    if(DATASET == "AS"):
        path = os.path.join(path, "as19981229.txt")  # path
        G = pre_process(path)
    elif(DATASET == "USAir"):
        path = os.path.join(path, "USAirLines.txt")  # path
        G = pre_process_USAirLines(path)
    else:
        print("[ERROR] DATASET value ERROR")
    print("path:", path)

    HOST_SIZE = int(G.number_of_nodes()*HOST_SIZE_FRAC)
    print('[Logging to file] time id: ', int(start_time), 'start time: ',
          time.asctime(time.localtime(start_time)), '\n', quick_find_msg)
    print('PHASE1_NFFE:', PHASE1_NFFE, ' PHASE2_NFFE:', PHASE2_NFFE,
          ' HOST_SIZE:', HOST_SIZE, "PHASE2_MUTATION_METHOD:", PHASE2_MUTATION_METHOD, "PERCENT:", PHASE2_MUTATION_PERCENT, "POP_SIZE:", POP_SIZE)

    #################### MOEA start here ################################
    log_to_file(int(start_time), G)
    print(nx.info(G))

    ### definition of MOEA ########################################################
    MOEA = moea.MOEA_SFCN(G, HOST_SIZE=HOST_SIZE,
                          ALPHA_R=ALPHA_R, ALPHA_D=1, RATIO=1, POP_SIZE=POP_SIZE, PHASE2_MAINTAIN_MAX=PHASE2_MAINTAIN_MAX, in_PHASE2_MUTATION_METHOD=PHASE2_MUTATION_METHOD, in_PHASE2_MUTATION_PERCENT=PHASE2_MUTATION_PERCENT)
    ### PHASE1 process single object EA ###############################################
    MOEA.single_object_process(PHASE1_NFFE)
    ### PHASE2 process NSGA-2: PHASE2_NFFE: cal counts of lambda_c & robustness##########
    front, f1_fit, f2_fit = MOEA.NSGA_2_process(PHASE2_NFFE)
    ### Algorithm END MOEA ########################################################

    ### networks info ####################################################
    print('\n\n\n', nx.info(MOEA.pop[-1].G), '\n\n\n')
    print('POP_SIZE:', MOEA.POP_SIZE, ' HOST_SIZE:', MOEA.HOST_SIZE,
          ' MOEA.in_NETWORK_SIZE:', MOEA.in_NETWORK_SIZE, 'P_MUTATION:', P_MUTATION, "P_CROSSOVER:", P_CROSSOVER, ' ALPHA_R:', MOEA.pop[
              0].ALPHA_R,
          ' ALPHA_D:', MOEA.pop[0].ALPHA_D, ' RATIO:', MOEA.pop[
              0].RATIO, ' MOEA.in_PHASE2_MUTATION_METHOD:', MOEA.in_PHASE2_MUTATION_METHOD,
          "MOEA.in_PHASE2_MUTATION_PERCENT:", MOEA.in_PHASE2_MUTATION_PERCENT)

    print('phase_1_time:', MOEA.phase_1_time, ' PHASE1_NFFE:', PHASE1_NFFE)
    print('phase_2_time:', MOEA.phase_2_time, ' PHASE2_NFFE:', PHASE2_NFFE,
          ' total time(now):', (time.process_time() - start_process_t)/3600, 'hours')

    ### ndmn_front ####################################################################
    ndmn_front = [MOEA.pop[n] for n in front[0]]
    ndmn_front = sorted(ndmn_front, key=attrgetter('lambda_c'))
    f1_fit = [moea.cal_fit1(i) for i in ndmn_front]
    f2_fit = [moea.cal_fit2(i) for i in ndmn_front]
    print('result: f1_fit', f1_fit, ' best lambda_c:', np.max(f1_fit))
    print('result: f2_fit', f2_fit, ' best Robustness:', np.max(f2_fit))
    print('len(ndmn_front):', len(ndmn_front))
    for i in range(len(ndmn_front)):
        print('index:', i, ' lambda_c:', f1_fit[i], ' Robustness:', f2_fit[i])
        ndmn_front[i].host_l = sorted(ndmn_front[i].host_l)
        print(ndmn_front[i].host_l)

    ### Hyper volume ################################################################
    print("HV:", moea.cal_hyper_volume(f1_fit, f2_fit))
    ### property ####################################################################
    moea.print_property(ndmn_front)
    #################################################################################

    ### cancel log to file ######################################################
    cancel_log()
    print('[End] Total time: ', (time.time() - start_time)/3600, 'hours, process time', (time.process_time() - start_process_t)/3600, '#MOEA_SFCN', "HV:", moea.cal_hyper_volume(f1_fit, f2_fit), "\n\n")


if __name__ == "__main__":
    main_loop_time = 1
    print("main loop time = ", main_loop_time)
    for i in range(main_loop_time):
        main()#python CN_Run_Real.py
