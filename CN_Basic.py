#!/bin/env python
# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
from networkx.algorithms import approximation


# TimeLog = {'recal_total':[], 'cc':[], 'R_total':[], 'recal_bet':[], 'cf':[], 'remove':[]}


class SFN:
    # self.RATIO \in [0,1], when self.RATIO=1, set capacity according to degree;
    # RATIO=0, according to betweeness(NOT including senders receivers).
    # ALPHA_D:  degree part of capacity setting: according to degree**ALPHA_D / total(degree**ALPHA_D)
    # cannot use before init: lambda_c b_sum L0
    def __init__(self, G: nx.graph, host_count: int,
                 ALPHA_R: float = 0.1, ALPHA_D: float = 1, RATIO: float = 1):
        if (RATIO > 1):
            print("__init__  RATIO>1 error!!!")
            RATIO = 1
        if (RATIO < 0):
            print("__init__  RATIO<0 error!!!")
            RATIO = 1
        self.G = G  # Graph
        # count of host
        self.host_count = host_count
        # list of host
        self.host_l = None  # set in the process #mutation, crossover, etc
        self.generate_host_randomly()  # set: self.host_l

        # redundancy parameter; 冗余参数
        self.ALPHA_R = ALPHA_R
        # degree power parameter #according to degree**ALPHA_D / total(degree**ALPHA_D)
        self.ALPHA_D = ALPHA_D  # 1 for proportional
        # capacity RATIO, 1:according to degree, 0: betweeness
        self.RATIO = RATIO  # [0,1]

        # cal after:
        # critical generating rate
        self.lambda_c = 9999999
        # betweeness sum
        self.b_sum = -1
        # initial total load
        self.L0 = -1
        # flag for Robustness calculate
        self.RFlag = False  # record the R can be used directly or not
        self.R = 0  # Robustness recorded

    def __hash__(self):
        return hash(self.lambda_c)

    def __eq__(self, other):
        if isinstance(other, SFN):
            return (set(self.host_l) == set(other.host_l))
        else:
            return False

    # set self.host_count: randomly generate host #self.host_count: number of host
    def generate_host_randomly(self):
        N = nx.number_of_nodes(self.G)
        if(self.host_count >= N):
            print('[error]!!! self.host_count >= network size')
            return False
        self.host_l = np.random.choice(
            N, self.host_count, replace=False).tolist()
        return True

    # don't use as possible as you can #do it at crossover is better
    # remove duplicate host and add host randomly
    # True: No duplicate; False: some duplicate but have been removed
    def remove_duplicate(self):
        self.host_l = list(set(self.host_l))
        if(len(self.host_l) == self.host_count):
            return True
        if(len(self.host_l) == 0):
            print('ini of sfn host list, host size: ori: 0; now: ', end='')
            self.generate_host_randomly()
        while(len(self.host_l) < self.host_count):
            self.host_l.append(random.randint(0, nx.number_of_nodes(self.G)-1))
            self.host_l = list(set(self.host_l))
        print(len(self.host_l))
        return False

    # draw graph #needed: topology: G; host: host_l

    def draw_graph(self):
        nlist = list()
        degree_exits = list(set(sorted([d for n, d in self.G.degree()])))
        # print('degree_exits: ', degree_exits)
        degree_exits = sorted(degree_exits)
        # print('degree_exits: ', degree_exits)
        for d_level in degree_exits:
            sub_list = [n for n, d in self.G.degree() if d == d_level]
            nlist.insert(0, sub_list)
        i = 1
        # print('before:',nlist)
        plt.figure(figsize=(4, 4))
        while(i < len(nlist)-4):  # 忽略最中心一层与最外三层
            if(len(nlist[i]) <= 1):
                nlist[i+1].extend(nlist[i])
                nlist.pop(i)
            elif(len(nlist[i]) <= 2**(i)
                 and len(nlist[i])+len(nlist[i+1]) > len(nlist[i+2])
                 and (len(nlist[i+1])+len(nlist[i+2])) < 2**(i)):
                nlist[i+2].extend(nlist[i+1])
                nlist.pop(i+1)
            else:
                i += 1
        # print('after:',nlist)
        n_color = list()
        for i in self.G.nodes:
            if i in self.host_l:
                n_color.append("#ea0000")
            else:
                n_color.append("#ffd306")
        nx.draw_shell(self.G, nlist=nlist, with_labels=False, node_size=5,
                      font_size=7, width=0.2, edge_color='#272727', node_color=n_color)
        plt.show()
        return None

    # must be call as early as possible!!! #need: Graph: G, host: host_l
    # attribute add: usep, betweenness #self set: self.b_sum #usep: b/self.b_sum
    # return: sum of betweenness

    def set_betweenness_and_usep(self):
        bc = nx.betweenness_centrality_subset(
            self.G, self.host_l, self.host_l, normalized=False)  # re: dict
        # bc.update({n: 2*b for n,b in bc.items()})#double
        # M = len(self.host_l)
        # for i in self.host_l:#cal betweenness of sender and receiver ###
        #     bc[i] += M-1
        # set attribute: betweenness
        nx.set_node_attributes(self.G, bc, 'betweenness')
        # sum of betweenness
        self.b_sum = sum([b for b in bc.values()])
        # cal: use probability
        bc.update({n: b/self.b_sum for n, b in bc.items()})
        # set attribute: use probability
        nx.set_node_attributes(self.G, bc, 'usep')
        return self.b_sum

    # call after attributeuse "usep" are set.
    # attribute add: capacity; according to degree and betweeness; self.RATIO \in [0,1]
    # when self.RATIO=1, according to degree; RATIO=0, according to betweeness
    # whem set degree part: according to degree**ALPHA_D / total(degree**ALPHA_D)

    def set_ori_capacity_by_degree2betweeness(self):
        R_total = nx.number_of_nodes(self.G)
        dv = nx.degree(self.G)
        d_sum = sum([d**self.ALPHA_D for n, d in dv])
        dv = {n: d**self.ALPHA_D/d_sum * self.RATIO for n,
              d in dv}  # uniformize: sum(dv)=1
        b_RATIO = 1.0 - self.RATIO  # [0, 1]
        b_cap = {n: self.G.nodes[n]['usep']*b_RATIO for n in self.G.nodes}
        cap_dict = {n: (dv[n]+b_cap[n])*R_total for n in self.G.nodes}
        nx.set_node_attributes(self.G, cap_dict, 'ori_capacity')
        return None

    # return: node id of degree max(if more than one, random select one)

    def find_degree_max(self):
        dv = nx.degree(self.G)
        dmax = max([v for k, v in dv])  # degree max
        # if more than one, randomly select one
        nid = random.choice([i for i, j in dv if j == dmax])
        return nid

    # use_p max
    # return: node id of load max #if there are some nodes with the same load, choose the min(id)
    def find_load_max(self):
        nid = 0
        max_usep = -1
        for i in self.G.nodes:
            if(self.G.nodes[i]['usep'] > max_usep):
                nid, max_usep = i, self.G.nodes[i]['usep']
        return nid

    #cal: lambda_c

    def cal_lambda_c(self):
        self.lambda_c = 9999999
        M = len(self.host_l)
        D = self.cal_average_D_formula()
        for i in self.G.nodes:
            if(self.G.nodes[i]['usep'] > 1e-100):
                temp = self.G.nodes[i]['ori_capacity'] /    \
                    (D * self.G.nodes[i]['usep'] * M)
                # select the smalleer one
                self.lambda_c = temp if temp < self.lambda_c else self.lambda_c
        return self.lambda_c

    # cal: initial total load(L0)# need attribute: usep, ori_capacity
    # slow, just for verify #attribute needed: ori_capacity

    def cal_L0(self):
        RU_min = 9999999  # record capacity[i]/usp[i] #R[i] = capacity[i]
        for i in self.G.nodes:
            if(self.G.nodes[i]['usep'] > 1e-100):
                temp = self.G.nodes[i]['ori_capacity']/self.G.nodes[i]['usep']
                RU_min = temp if temp < RU_min else RU_min
        self.L0 = RU_min / (1 + self.ALPHA_R)
        return self.L0

    # cal: initial total load(L0)
    # need: lambda_c, b_sum, M, ALPHA(redundancy parameter)

    def cal_L0_fast(self):
        M = len(self.host_l)
        D = self.cal_average_D_formula()
        self.L0 = self.lambda_c * D * M / (1+self.ALPHA_R)
        return self.L0

    # betweenness usep cap lambda_c L0
    # typically 0.1 second  0.9~0.11

    def update(self):
        self.RFlag = False  # current Value R can NOT be directly used
        if(self.remove_duplicate() == False):  # remove host duplication
            print('[Error] Duplication of hosts occurred!')
        self.set_betweenness_and_usep()  # set b_sum
        self.set_ori_capacity_by_degree2betweeness()
        self.cal_lambda_c()
        self.cal_L0_fast()
        return None

    # precondition: attribute: ori_capacity
    # cal: R(Robustness), formula:R = 1/N sum(s(Q)) Q \in [1, N]

    def cal_R_2(self):
        # global TimeLog
        # t = time.process_time()
        if(self.RFlag == True):  # current Value R can be directly used
            return self.R
        # R2_start_time = time.process_time()
        N = nx.number_of_nodes(self.G)
        sub_R_l = []  # record R in each recursion
        for Q in range(1, N):  # Q \in [1, N)
            sfn_c = copy.deepcopy(self)  # copy a new graph to attack

            # t1 = time.process_time()
            # remove Q nodes # avg time:1.76/cal_R, takes 22.2%
            for _ in range(Q):
                load_max_nid = sfn_c.find_load_max()
                sfn_c.G.remove_node(load_max_nid)
            # TimeLog['remove'].append(time.process_time()-t1)

            # t2 = time.process_time()
            # avg time:5.219/cal_R
            sub_R = self.casading_failure_process(sfn_c.G)
            # TimeLog['cf'].append(time.process_time()-t2)

            if(sub_R <= 1e-100):
                break
            sub_R_l.append(sub_R)
        # print(sub_R_l, len(sub_R_l))
        self.R = sum(sub_R_l) / N
        self.RFlag = True
        # TimeLog['R_total'].append(time.process_time()-t)
        # print('R2 time used: ', time.process_time() - R2_start_time, self.R)
        return self.R

    # casading failure process
    # return: Robustness
    def casading_failure_process(self, G):  # avg time:7.67/cal_R
        # global TimeLog
        while(True):  # casading failure process

            # t = time.process_time()
            # betweenness and usep #avg time:5.04/cal_R
            new_b_sum, new_L0 = self.recal_loop(G)
            # TimeLog['recal_total'].append(time.process_time()-t)

            if(new_b_sum == 0):
                return 0
            to_delete = list()
            for n in G.nodes:
                if(new_L0*G.nodes[n]['usep'] > G.nodes[n]['ori_capacity']):
                    to_delete.append(n)
            if(len(to_delete) == 0):  # no node was deleted in this recursion
                break  # no casading failure
            for n in to_delete:
                G.remove_node(n)

        # t2 = time.process_time()
        # connected_components generator #avg time:0.00156/cal_R
        cc_g = nx.connected_components(G)
        # TimeLog['cc'].append(time.process_time()-t2)
        #host in connected_components
        h_cc = list()
        for cc in cc_g:
            h_cc.append(len([n for n in self.host_l if n in cc]))
        # print('len(h_cc):',len(h_cc), h_cc, 'max:', max(h_cc), 'sum:', sum(h_cc))
        return max(h_cc) / len(self.host_l)

    # re-cal: new_b_sum, L0, betweenness, usep #loop for cal robustness
    # parameter: new Graph G
    # return: new_b_sum, new_L0
    def recal_loop(self, G: nx.graph):  # G: the Graph after removing some nodes
        # global TimeLog
        new_host = [n for n in self.host_l if n in G]

        # t = time.process_time()
        # avg time:4.9/cal_R, takes 64.6%
        bc = nx.betweenness_centrality_subset(
            G, new_host, new_host, normalized=False)
        # TimeLog['recal_bet'].append(time.process_time()-t)

        nx.set_node_attributes(G, bc, 'betweenness')
        new_b_sum = sum([b for b in bc.values()])
        if(new_b_sum <= 1e-100):
            # print("[warning]new_b_sum==0! new_host:", new_host, "new_b_sum:", new_b_sum)
            return 0, 0
        # sometimes new_b_sum==0
        bc.update({n: b/new_b_sum for n, b in bc.items()})
        nx.set_node_attributes(G, bc, 'usep')
        new_L0 = new_b_sum/self.b_sum * self.L0
        return new_b_sum, new_L0

    # cal: average distance(host-host) of Graph #host only
    def cal_average_D(self):
        d_sum = 0
        M = len(self.host_l)
        count = 0
        for i in range(M):
            for j in range(M - i - 1):  # only host to host
                d_sum += nx.shortest_path_length(self.G,
                                                 source=self.host_l[i], target=self.host_l[i+j+1])
                count += 1
        return d_sum / (M*(M-1)/2)

    # cal: average distance(host-host) of Graph
    def cal_average_D_formula(self):
        M = len(self.host_l)
        return (2*self.b_sum / (M*(M-1))) + 1

    # Network average clustering coefficient 网络平均集聚系数
    def cal_average_clustering(self):  # for complete graph, it's 1
        return approximation.average_clustering(self.G)

    # return: (node_id, usep) that the node has the max usep
    def usep_max(self):
        usep_dict = {k: self.G.nodes[k]['usep'] for k in self.G.nodes}
        # node id  of usep max
        return max(usep_dict.items(), key=lambda x: x[1])

    # average closeness_centrality of host
    # 接近中心性，到其余所有节点的平均距离的倒数 #计算的是所有节点
    def cal_avg_closeness_centrality(self):
        # dict: {node_id: closeness_centrality}
        d = nx.closeness_centrality(self.G)
        # only care about the host
        d = {k: v for k, v in d.items() if k in self.host_l}
        d = [v for k, v in d.items()]  # select the closeness_centrality values
        return np.mean(d)

    # 全局通信效率，用于探究受到攻击下全局通信效率的降低过程
    # 计算一次大约需要35-40s #复杂度过高

    def cal_GlobalComEffi(self):
        N = nx.number_of_nodes(self.G)
        arrGlobalComEffi = np.zeros(N)
        for Q in range(0, N):
            sfn_c = copy.deepcopy(self)  # copy a new graph to attack
            for _ in range(Q):
                load_max_nid = sfn_c.find_load_max()
                sfn_c.G.remove_node(load_max_nid)
            while(True):  # casading failure process
                new_b_sum, new_L0 = self.recal_loop(sfn_c.G)
                if(new_b_sum == 0):
                    break
                to_delete = list()
                for n in sfn_c.G.nodes:
                    if(new_L0*sfn_c.G.nodes[n]['usep'] > sfn_c.G.nodes[n]['ori_capacity']):
                        to_delete.append(n)
                if(len(to_delete) == 0):  # no node was deleted in this recursion
                    break  # no casading failure
                for n in to_delete:
                    sfn_c.G.remove_node(n)
            arrGlobalComEffi[Q] = self.sub_GlobalComEffi(sfn_c.G, N)
        # print(arrGlobalComEffi, "<- END", type(arrGlobalComEffi))
        return arrGlobalComEffi

    #
    def sub_GlobalComEffi(self, in_G, N):
        s = 0.0
        # dPairPath = dict(nx.all_pairs_shortest_path_length(in_G))
        # for i in range(len(self.host_l)):
        #     Nidx_i = self.host_l[i]
        #     if(Nidx_i not in dPairPath.keys()):
        #         continue
        #     PathLength = dPairPath[Nidx_i]
        #     for j in range(i+1, len(self.host_l)):
        #         Nidx_j = self.host_l[j]
        #         if(Nidx_j in PathLength.keys()):
        #             s += 1 / PathLength[Nidx_j]
        for i in range(len(self.host_l)):
            Nidx_i = self.host_l[i]
            if(Nidx_i not in in_G.nodes):
                continue
            for j in range(i+1, len(self.host_l)):
                Nidx_j = self.host_l[j]
                try:
                    PathLen = nx.shortest_path_length(in_G, Nidx_i, Nidx_j)
                except nx.NetworkXNoPath:
                    continue
                except nx.NodeNotFound:
                    continue
                s += 1/PathLen  # 注意加上的应该为距离的倒数，不连通的则加0
        return s / (N*(N-1))

    # Assortative Coefficient #度的同配性

    def cal_assortativity_coefficient(self):
        return nx.degree_assortativity_coefficient(self.G, nodes=self.host_l)


# Just for test
if __name__ == "__main__":
    # barabasi_albert_graph(n, m, seed=None)
    G = nx.random_graphs.barabasi_albert_graph(300, 2, 1)
    # print(nx.info(G))
    sfn = SFN(G, 50)
    sfn.host_l = [11, 22, 33, 42, 44, 47, 52, 56, 57, 58, 69, 77, 84, 85, 88, 104, 106, 118, 127, 129, 146, 162, 175, 184, 189,
                  196, 199, 210, 211, 215, 216, 225, 227, 228, 238, 239, 241, 243, 250, 254, 257, 260, 263, 272, 274, 277, 285, 290, 293, 294]
    start_time = time.process_time()
    sfn.update()
    print("[CN_Bsaic Test] update time used:",
          time.process_time() - start_time)

    start_time = time.process_time()
    arrGlobalComEffi = sfn.cal_GlobalComEffi()
    print("[CN_Bsaic cal_GlobalCommunicactionEfficiency] time used:",
          time.process_time() - start_time)
    print("arrGlobalComEffi", arrGlobalComEffi, arrGlobalComEffi.shape,
          type(arrGlobalComEffi))  # , arrGlobalComEffi.shape
