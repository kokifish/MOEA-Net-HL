#!/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
import networkx as nx
import random
import copy
import sys
import pickle
import os
from CN_Basic import SFN
from operator import itemgetter, attrgetter

# import numba
# from numba import jit

quick_find_msg = 'time seed=int(start_time), C(i): 0 1 0, m=2, auto classify, msg changed on Apr. 1'
# Attention! phase2_input.pop(1)#保留初始种群中lambda_c最差的,亦即Robustness通常最好的一个
BA_M = 2
NETWORK_SIZE = 500
HOST_SIZE = 25
P_MUTATION = 0.1
P_CROSSOVER = 0.3
POP_SIZE = 40
PHASE2_MAINTAIN_MAX = 30  # 前沿解最大维持代数，达到该代数后认为已收敛
ALPHA_R = 0.1  # 节点冗余量 ## Attention! This should be analyzed ########################
# PHASE2 mutation method #"random" "degree" "percent"
PHASE2_MUTATION_METHOD = "percent"
PHASE2_MUTATION_PERCENT = 0.7  # select the last ___  percent

# 80000 20000
PHASE1_NFFE = 80000  # typical time:   5*10^4:1.09h ~1.5h
PHASE2_NFFE = 20000  # fitness function evaluation:f1 + f2# typical time: 1.5*10^4:20.9h


saveStdOut = None
file_log = None


def cal_fit1(sfn: SFN):  # maximize
    return sfn.lambda_c


def cal_fit2(sfn: SFN):
    return sfn.cal_R_2()  # maximize


def fast_ndmn_sort(fit):  # write for Maximum question
    # fit: list(list() * M) for M object, ith fit for fitness of ith object
    # return [set, set, ...] for index of first front, second front...
    N = len(fit[0])  # population size
    S = [set() for i in range(N)]
    n = [0 for i in range(N)]
    front = [set()]  # front[0] is the set of best front, front[1] for next...
    for p in range(N):
        for q in range(N):
            if(fit[0][p] >= fit[0][q] and fit[1][p] > fit[1][q]) \
                    or (fit[0][p] > fit[0][q] and fit[1][p] >= fit[1][q]):  # p dominates q #dominate(p, q, fit)
                S[p].add(q)  # p dominates q
            elif(fit[0][q] >= fit[0][p] and fit[1][q] > fit[1][p]) \
                    or (fit[0][q] > fit[0][p] and fit[1][q] >= fit[1][p]):  # q dominates p #dominate(q, p, fit)
                n[p] += 1  # p BE dominatED count
        if(n[p] == 0):  # p is not dominates by any q
            front[0].add(p)
    i = 0
    while(len(front[i]) != 0):
        Q = set()
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if(n[q] == 0):
                    Q.add(q)
        front.append(Q)
        i += 1
    return front[0:-1]  # skip the last empty set()


def cal_crowd_distance(front, fit):
    # fit: list(list() * M) for M object, ith fit for fitness of ith object
    # sort function for function cal_crowd_distance (ascending)
    def sort_crowd_distance(front, sub_fit):
        sorted_front = [(i, sub_fit[i]) for i in front]
        sorted_front = sorted(sorted_front, key=itemgetter(1))  # sort by value
        # return keys sorted by value (ascending)
        return list(dict(sorted_front).keys())

    l = len(front)  # len of front is l
    d = {i: 0 for i in front}
    M = len(fit)  # object count
    for m in range(M):
        front = sort_crowd_distance(front, fit[m])
        d[front[0]] = d[front[-1]] = 999999999999999999999999
        max_fit, min_fit = max(fit[m]), min(fit[m])
        for i in range(1, l-1):
            d[front[i]] += abs(fit[m][front[i-1]] - fit[m]
                               [front[i+1]]) / (max_fit - min_fit)
    return d


def cal_hyper_volume(fit_1, fit_2):
    # 计算超体积HV #非支配前沿解的fit. fit_1: lambda_c 小到大 fit_2:robustness 大到小
    HV = fit_1[0]*fit_2[0]
    for i in range(1, len(fit_1)):
        HV += (fit_1[i]-fit_1[i-1]) * fit_2[i]
    return HV


class MOEA_SFCN:
    # ALPHA_R: redundancy alpha; ALPHA_D: degree alpha; RATIO=1 onlly depend on degree
    # Multiobjective Evolution Algorithm
    def __init__(self, G: nx.Graph, HOST_SIZE: int,
                 ALPHA_R, ALPHA_D, RATIO, POP_SIZE, PHASE2_MAINTAIN_MAX,
                 in_PHASE2_MUTATION_METHOD, in_PHASE2_MUTATION_PERCENT):
        self.pop = list()  # list(SFN, SFN, ...); len=POP_SIZE
        self.HOST_SIZE = HOST_SIZE  # const int
        self.in_NETWORK_SIZE = G.number_of_nodes()
        self.POP_SIZE = POP_SIZE  # const int
        for _ in range(POP_SIZE):
            sfn = SFN(G, host_count=HOST_SIZE,
                      ALPHA_R=ALPHA_R, ALPHA_D=ALPHA_D, RATIO=RATIO)
            sfn.update()
            self.pop.append(sfn)
        self.cur_R_NFFE = 0  # current Number of Robustness Fitness Function Evaluation
        # variable for termination judge ↓ ###
        self.maintain_pop = set()  # maintain pop set, used for termination decision-making
        self.maintain_count = 0  # maintain time count
        self.PHASE2_MAINTAIN_MAX = PHASE2_MAINTAIN_MAX
        ####################
        self.P_DISTRIBUTION_DEGREE = dict(G.degree)
        self.LOW_DEGREE_NODE_LIST = sorted(
            self.P_DISTRIBUTION_DEGREE.items(), key=lambda d: d[1])
        strat_idx = int(len(self.LOW_DEGREE_NODE_LIST) *
                        (1 - in_PHASE2_MUTATION_PERCENT))
        self.LOW_DEGREE_NODE_LIST = list(
            dict(self.LOW_DEGREE_NODE_LIST[strat_idx:]).keys())
        print("LOW_DEGREE_NODE_LIST:", len(self.LOW_DEGREE_NODE_LIST), len(
            self.LOW_DEGREE_NODE_LIST)/self.in_NETWORK_SIZE, self.LOW_DEGREE_NODE_LIST)
        self.P_DISTRIBUTION_DEGREE.update(
            {n: d**-1 for n, d in self.P_DISTRIBUTION_DEGREE.items()})
        total = sum([d for n, d in self.P_DISTRIBUTION_DEGREE.items()])
        self.P_DISTRIBUTION_DEGREE.update(
            {n: d/total for n, d in self.P_DISTRIBUTION_DEGREE.items()})
        self.P_DISTRIBUTION_DEGREE = [
            d for n, d in self.P_DISTRIBUTION_DEGREE.items()]

        self.in_PHASE2_MUTATION_METHOD = in_PHASE2_MUTATION_METHOD
        self.in_PHASE2_MUTATION_PERCENT = in_PHASE2_MUTATION_PERCENT

    def select_half(self):  # select the individual to POP_SIZE # e.g. len>=40 to len=20
        fit_l = [sfn.lambda_c for sfn in self.pop]  # list(sfn.lambda_c,...)
        # fit_l = [v/sum(fit_l) for v in fit_l]
        idx_l = list()
        # print('fit_l',fit_l, len(fit_l))
        while(len(idx_l) < self.POP_SIZE):
            idx = fit_l.index(max(fit_l))
            fit_l[idx] = -1
            idx_l.append(idx)

        # the index of sfn selected to be a member of next pop
        # idx_l = np.random.choice(np.arange(len(fit_l)), size=self.POP_SIZE, replace=False, p=fit_l)
        next_pop = list()
        for i in idx_l:
            next_pop.append(self.pop[i])
        self.pop = next_pop
        return None

    # for phase 1 #return the index of the best sfn according to lambda_c
    def select_best_idx(self):
        fit_l = [sfn.lambda_c for sfn in self.pop]
        return fit_l.index(max(fit_l))

    def select_worst_idx(self):
        fit_l = [sfn.lambda_c for sfn in self.pop]
        return fit_l.index(min(fit_l))

    # bug! time_id:  ...73026, lambda_c最佳的在PHASE2支配掉了？
    def single_object_process(self, PHASE1_NFFE: int):
        start_time = time.process_time()
        phase2_input = copy.deepcopy(self.pop)
        phase2_input = sorted(
            phase2_input, key=attrgetter('lambda_c'))  # ascending
        loop_NFFE = 0
        cur_NFFE = self.POP_SIZE
        # Only for log ################
        l = [sfn.lambda_c for sfn in phase2_input]
        print('[PHASE 1] ini pop lambda_c  :', len(l), l)
        l = [sfn.cal_R_2() for sfn in phase2_input]
        print('[PHASE 1] ini pop Robustness:', len(l), l)
        # Only for log ################
        while(cur_NFFE < PHASE1_NFFE):
            child = list()  # list(SFN, SFN, ...); len=POP_SIZE
            self.remove_pop_duplicate()
            idx_l = [i for i in range(len(self.pop))]
            random.shuffle(idx_l)
            for i in range(len(self.pop)//2):  # crossover process
                c_1, c_2 = self.crossover(
                    self.pop[idx_l[2*i]], self.pop[idx_l[2*i+1]])
                child.append(c_1)
                child.append(c_2)
            for sfn in child:  # mutation process
                self.mutation(sfn, "random")
                sfn.update()

            self.pop = self.pop + child
            self.select_half()  # 筛选掉self.pop的一半
            # select one and put in phase2_input
            if(loop_NFFE >= 0.5*(PHASE1_NFFE//self.POP_SIZE)):
                loop_NFFE = 0
                idx = self.select_best_idx()
                if(phase2_input[-1].lambda_c + 1e-10 < self.pop[idx].lambda_c):
                    phase2_input.append(self.pop[idx])
                # Only for log ################
                l = [sfn.lambda_c for sfn in phase2_input]
                print('phase2_input(lambda_c): ', len(l), l)
            loop_NFFE += self.POP_SIZE
            cur_NFFE += self.POP_SIZE
        # while end! # now let phase2_input's size to POP_SIZE
        del_num = len(phase2_input) - self.POP_SIZE
        del_index = [int((len(phase2_input)/del_num)*i) +
                     1 for i in range(del_num)]
        if(len(del_index) > 1):
            if(del_index[-1] == len(phase2_input)-1):
                del_index[-1] -= 1
            i = -1
            while(del_index[i] == del_index[i-1]):
                del_index[i-1] -= 1
                i -= 1
        del_index = set(del_index)
        print(type(del_index), 'del_index:', del_index, len(del_index))
        self.pop = [phase2_input[i] for i in range(
            len(phase2_input)) if i not in del_index]  # refresh self.pop
        # Only for log ################
        l = [sfn.lambda_c for sfn in self.pop]
        print('final(lambda_c): ', len(l), l)
        l = [sfn.cal_R_2() for sfn in self.pop]
        print('final(Robustness): ', len(l), l)
        print('lambda_c best sfn host: ', self.pop[-1].host_l)
        # Only for log ################################
        self.phase_1_time = time.process_time() - start_time
        return None

    # True: continue #False: terminate
    def Phase2_continue_judge(self, front):  # front: 前沿解下标
        cur_pop = set([self.pop[n] for n in front[0]])
        print('[Log] Phase2 Judge: len(front[0])=', len(
            cur_pop), 'maintain_count=', self.maintain_count)
        if(cur_pop == self.maintain_pop and self.maintain_count >= self.PHASE2_MAINTAIN_MAX):
            print('[Terminate!] Front NOT change for ',
                  self.maintain_count, 'iterations, stoped!')
            return False  # terminate, should not continue
        if(cur_pop == self.maintain_pop):
            # random.seed(int(time.process_time()))
            # np.random.seed(int(time.process_time())%2154937331)
            # print("[Log] random.seed np.random.seed changed. self.maintain_count=", self.maintain_count)
            self.maintain_count += 1
        else:  # front change
            self.maintain_pop = cur_pop
            self.maintain_count = 0
        return True  # continue to loop, no convergence

    def NSGA_2_process(self, PHASE2_NFFE: int):
        start_time = time.process_time()
        gen_n = 0
        self.cur_R_NFFE = self.POP_SIZE
        while(self.cur_R_NFFE * 2 < PHASE2_NFFE):

            # print('PHASE2 generation No.', gen_n,'; self.cur_R_NFFE *2 :', self.cur_R_NFFE*2)
            # remove the duplicate sfn, and randomly generate some sfn so that len(self.pop)==self.POP_SIZE
            self.remove_pop_duplicate()
            self.make_new_pop()  # cur pop size will extend to 2*cur pop size(maybe less to POP_SIZE)
            f1_fit = [cal_fit1(i) for i in self.pop]
            f2_fit = [cal_fit2(i) for i in self.pop]
            self.cur_R_NFFE += self.POP_SIZE
            front = fast_ndmn_sort([f1_fit, f2_fit])  # front: 分rank的前沿解下标
            # terminate judge # False: DO NOT continue
            if(not self.Phase2_continue_judge(front)):
                break

            parent = list()  # new generation #list(SFN)
            i = 0  # front rank
            # select top fit to new generation
            while(len(parent) + len(front[i]) <= self.POP_SIZE):
                parent.extend([self.pop[n] for n in front[i]])
                i += 1

            crowd_dict = cal_crowd_distance(
                front[i], [f1_fit, f2_fit])  # call: cal_crowd_distance
            crowd_sorted = list(sorted(crowd_dict.items(), key=itemgetter(
                1), reverse=True))  # descending order
            # print('crowd_sorted:', crowd_sorted, len(crowd_sorted), type(crowd_sorted))
            # the number to fill the parent
            to_append = list(crowd_sorted)[0: self.POP_SIZE-len(parent)]
            for index, _ in to_append:
                parent.append(self.pop[index])

            self.pop = parent
            gen_n = gen_n + 1  # now len(self.pop)==POP_SIZE

        print('self.cur_R_NFFE:', self.cur_R_NFFE, ' gen_n:', gen_n)
        self.phase_2_time = time.process_time() - start_time
        f1_fit = [cal_fit1(i) for i in self.pop]
        f2_fit = [cal_fit2(i) for i in self.pop]
        front = fast_ndmn_sort([f1_fit, f2_fit])
        print("PHASE2_MUTATION_METHOD:", PHASE2_MUTATION_METHOD)
        return front, f1_fit, f2_fit

    def make_new_pop(self):
        child = list()
        index_l = [i for i in range(len(self.pop))]
        random.shuffle(index_l)  # random number list
        for i in range(len(self.pop)//2):  # crossover
            c_1, c_2 = self.crossover(
                self.pop[index_l[2*i]], self.pop[index_l[2*i+1]])
            child.append(c_1)
            child.append(c_2)
        for sfn in child:  # mutation
            self.mutation(sfn, self.in_PHASE2_MUTATION_METHOD)
            sfn.update()
            self.pop.append(sfn)
        return None

    # remove duplacate and make sure that pop size == POP_SIZE
    def remove_pop_duplicate(self):
        self.pop = list(set(self.pop))  # remove_pop_duplicate
        while(len(self.pop) < self.POP_SIZE):  # let pop size == POP_SIZE
            sfn = copy.deepcopy(self.pop[0])
            sfn.host_l = []
            sfn.update()
            self.pop.append(sfn)
        return None

    # make sure that after crossover, no same host
    def crossover(self, a: SFN, b: SFN):
        c_1 = copy.deepcopy(a)
        c_2 = copy.deepcopy(b)
        set_a = set(a.host_l)
        set_b = set(b.host_l)
        intersect = set_a & set_b  # 交集
        diff_count = len(set_a) - len(intersect)
        set_a_sub = set(random.sample(
            set_a - intersect, int(diff_count*P_CROSSOVER)))
        set_b_sub = set(random.sample(
            set_b - intersect, int(diff_count*P_CROSSOVER)))
        set1 = (set_a | set_b_sub) - set_a_sub
        set2 = (set_b | set_a_sub) - set_b_sub
        # print(len(set1),len(set2), end="")
        c_1.host_l = list(set1)
        c_2.host_l = list(set2)
        # print("[Time] crossover", time.process_time() - crossover_t)
        return c_1, c_2  # child 1 and 2 # type: SFN

    # this will not modify any object of self
    def mutation(self, sfn, method):
        if(method == "random"):
            for i in range(len(sfn.host_l)):
                if(random.random() < P_MUTATION):
                    h_id = random.randint(
                        0, self.in_NETWORK_SIZE-1)  # [a,b] 闭区间！
                    host_set = set(sfn.host_l)
                    while(h_id in host_set):
                        h_id = random.randint(0, self.in_NETWORK_SIZE-1)
                    sfn.host_l[i] = h_id
        elif(method == "degree"):
            to_choose = np.random.choice(np.arange(sfn.G.number_of_nodes()), size=len(
                sfn.host_l), replace=False, p=self.P_DISTRIBUTION_DEGREE)
            host_set = set(sfn.host_l)
            for i in range(len(sfn.host_l)):
                if(random.random() < P_MUTATION and to_choose[i] not in host_set):
                    h_id = to_choose[i]
        elif(method == "percent"):
            to_choose = np.random.choice(
                self.LOW_DEGREE_NODE_LIST, size=len(sfn.host_l), replace=False)
            host_set = set(sfn.host_l)
            for i in range(len(sfn.host_l)):
                if(random.random() < P_MUTATION and to_choose[i] not in host_set):
                    h_id = to_choose[i]
        else:
            print("[ERROR] mutation TYPE ERROR")
        return None


def log_to_file(time_id, G):
    # 存储至文件
    # 文件名：NETWORK_SIZE _ HOST_SIZE _ pop+POP_SIZE _ PHASE2_MUTATION_METHOD PHASE1_NFFE _ PHASE2_NFFE (percent) _ ALPHA_+ALPHA_R
    global saveStdOut
    global file_log
    saveStdOut = sys.stdout

    path = str(NETWORK_SIZE) + "_" + str(HOST_SIZE) + "_pop" + str(POP_SIZE) + "_" + \
        PHASE2_MUTATION_METHOD + "_" + \
        str(PHASE1_NFFE) + "_" + str(PHASE2_NFFE)  # 文件夹目录
    if(PHASE2_MUTATION_METHOD == "percent"):
        path = path + "_" + str(int(PHASE2_MUTATION_PERCENT*100))
    path = path + "_ALPHA_" + str(ALPHA_R)
    path = os.path.join(os.path.abspath('.'), path)  # 绝对路径 + 相对路径
    folder = os.path.exists(path)  # 判断是否存在
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder ", path, " has been created  ---")

    else:
        print("---  folder ", path, " exits  ---")

    # 绝对路径 + time id为文件名
    file_log = open((os.path.join(path, str(time_id) + str('.log'))), 'w')
    sys.stdout = file_log
    file_G = open((os.path.join(path, str(time_id) + str('G.data'))), 'wb+')
    pickle.dump(G, file_G)
    file_G.close()


def cancel_log():
    sys.stdout = saveStdOut
    file_log.close()


def print_property(ndmn_front):
    avg_D = [i.cal_average_D() for i in ndmn_front]
    print('cal_average_D', avg_D, ' mean:', np.mean(avg_D),
          ' max:', max(avg_D), ' min:', min(avg_D))
    avg_clustering = [i.cal_average_clustering() for i in ndmn_front]
    print('cal_average_clustering', avg_clustering, ' mean:', np.mean(avg_clustering),
          ' max:', max(avg_clustering), ' min:', min(avg_clustering))
    avg_closeness_centrality = [
        i.cal_avg_closeness_centrality() for i in ndmn_front]
    print('avg_closeness_centrality', avg_closeness_centrality, ' mean:', np.mean(avg_closeness_centrality),
          ' max:', max(avg_closeness_centrality), ' min:', min(avg_closeness_centrality))
    # assortativity_coefficient = [i.cal_assortativity_coefficient() for i in ndmn_front] #度同配性
    # print('assortativity_coefficient of host', assortativity_coefficient, ' mean:', np.mean(assortativity_coefficient), \
    #             ' max:', max(assortativity_coefficient), ' min:', min(assortativity_coefficient))


def main():
    start_time = time.time()
    start_process_t = time.process_time()

    random.seed(int(start_time) % 100001651)
    np.random.seed(int(start_time) % 2154937333)

    # (n, m, seed) #n: network size
    G = nx.random_graphs.barabasi_albert_graph(
        NETWORK_SIZE, BA_M, int(start_time))  # time.time()

    print('[Logging to file] time id: ', int(start_time), 'start time: ',
          time.asctime(time.localtime(start_time)), '\n', quick_find_msg)
    print('PHASE1_NFFE:', PHASE1_NFFE, 'PHASE2_NFFE:', PHASE2_NFFE, 'NETWORK_SIZE:',
          NETWORK_SIZE, 'HOST_SIZE:', HOST_SIZE, 'BA_M:', BA_M, "PHASE2_MUTATION_METHOD:",
          PHASE2_MUTATION_METHOD, "PERCENT:", PHASE2_MUTATION_PERCENT, "POP_SIZE:", POP_SIZE)

    # begin to log to file ##################################
    log_to_file(int(start_time), G)
    print('Attention! Here is the message for quick find: \n ',
          quick_find_msg, time.localtime(start_time), ' \n\n')
    ######################################

    # definition of MOEA ##########################################################
    MOEA = MOEA_SFCN(G, HOST_SIZE=HOST_SIZE,
                     ALPHA_R=ALPHA_R, ALPHA_D=1, RATIO=1, POP_SIZE=POP_SIZE, PHASE2_MAINTAIN_MAX=PHASE2_MAINTAIN_MAX,
                     in_PHASE2_MUTATION_METHOD=PHASE2_MUTATION_METHOD, in_PHASE2_MUTATION_PERCENT=PHASE2_MUTATION_PERCENT)
    # PHASE1 process single object EA #############################################
    MOEA.single_object_process(PHASE1_NFFE)
    # PHASE2 process NSGA-2: PHASE2_NFFE: cal counts of lambda_c & robustness####
    front, f1_fit, f2_fit = MOEA.NSGA_2_process(PHASE2_NFFE)

    # networks info ####################################################################
    print('\n\n\n', nx.info(MOEA.pop[-1].G), '\n\n\n')
    print('BA_M:', BA_M, 'POP_SIZE:', MOEA.POP_SIZE, 'HOST_SIZE:', MOEA.HOST_SIZE,
          'MOEA.in_NETWORK_SIZE:', MOEA.in_NETWORK_SIZE, 'P_MUTATION:', P_MUTATION, "P_CROSSOVER:", P_CROSSOVER, ' ALPHA_R:', MOEA.pop[
              0].ALPHA_R,
          'ALPHA_D:', MOEA.pop[0].ALPHA_D, 'RATIO:', MOEA.pop[0].RATIO, ' MOEA.in_PHASE2_MUTATION_METHOD:', MOEA.in_PHASE2_MUTATION_METHOD,
          "MOEA.in_PHASE2_MUTATION_PERCENT:", MOEA.in_PHASE2_MUTATION_PERCENT)

    print('phase_1_time:', MOEA.phase_1_time, ' PHASE1_NFFE:', PHASE1_NFFE)
    print('phase_2_time:', MOEA.phase_2_time, ' PHASE2_NFFE:', PHASE2_NFFE,
          ' total time(now):', (time.process_time() - start_process_t)/3600, 'hours')

    # ndmn_front ##################################################################
    ndmn_front = [MOEA.pop[n] for n in front[0]]
    ndmn_front = sorted(ndmn_front, key=attrgetter('lambda_c'))
    f1_fit = [cal_fit1(i) for i in ndmn_front]
    f2_fit = [cal_fit2(i) for i in ndmn_front]
    print('result: f1_fit', f1_fit, ' best lambda_c:', np.max(f1_fit))
    print('result: f2_fit', f2_fit, ' best Robustness:', np.max(f2_fit))
    print('len(ndmn_front):', len(ndmn_front))
    for i in range(len(ndmn_front)):
        print('index:', i, ' lambda_c:', f1_fit[i], ' Robustness:', f2_fit[i])
        ndmn_front[i].host_l = sorted(ndmn_front[i].host_l)
        print(ndmn_front[i].host_l)

    # Hyper volume ################################################
    print("HV:", cal_hyper_volume(f1_fit, f2_fit))
    # property ################################################
    print_property(ndmn_front)
    ################################################

    # cancel log to file ############################
    cancel_log()
    print('[End] Total time: ', (time.time() - start_time)/3600, 'hours, process time',
          (time.process_time() - start_process_t)/3600, '#MOEA_SFCN', "HV:", cal_hyper_volume(f1_fit, f2_fit), "\n\n")


if __name__ == "__main__":
    main_loop_time = 1
    print("main loop time = ", main_loop_time, "\n")
    for i in range(main_loop_time):
        print("[", i, "/", main_loop_time, "]")
        main()
