import numpy as np
import pandas as pd
import networkx as nx
import random
import math
sd = 123
random.seed(sd)
np.random.seed(sd)
#References: https://github.com/MouinulIslamNJIT/Rank-Aggregation_Proportionate_Fairness/blob/main/AlgRAPF.py

def RAPF(profile_df, profile_item_group_dict, seed):
    """
    RAPF from SIGMOD'22
    :param profile_df: Dataframe of preference profile
    :param profile_item_group_dict: Dictionary of candidates (keys) and groups (values)
    :param k_cnt: length of consensus
    :param seed: seed for reproducability
    :return: consensus ranking
    """
    #Their code wants candidates to be represented by ints
    int_candidates = list(range(len(profile_item_group_dict.keys())))
    int_to_string_cand = dict(zip(int_candidates, list(profile_item_group_dict.keys())))
    string_to_int_cand = dict(zip( list(profile_item_group_dict.keys()), int_candidates))
    np.random.seed(seed)  # for reproducibility
    random.seed(seed) # for reproducibility
    num_rankings = len(profile_df.columns)
    #pick random
    rand = random.randint(0,num_rankings-1)
    single_ranking = profile_df[profile_df.columns[rand]]  # isolate ranking
    single_ranking = np.array(
        single_ranking[~pd.isnull(single_ranking)]
    )

    group = [profile_item_group_dict[i] for i in single_ranking]
    single_ranking = np.asarray([string_to_int_cand[c] for c in single_ranking])

    numberOfItem = len(single_ranking)

    rankGrp = {}
    for i in range(0, len(single_ranking)):
        rankGrp[single_ranking[i]] = group[i]

    grpCount = {}
    for i in group:
        grpCount[i] = 0

    rankGrpPos = {}
    for i in single_ranking:
        grpCount[rankGrp[i]] = grpCount[rankGrp[i]] + 1
        rankGrpPos[i] = grpCount[rankGrp[i]]

    rankRange = {}
    for item in single_ranking:
        i = rankGrpPos[item]
        n = numberOfItem
        fp = grpCount[rankGrp[item]]
        r1 = math.floor((i - 1) * n / fp) + 1
        r2 = math.ceil(i * n / fp)
        if r2 > numberOfItem:
            r2 = numberOfItem
        rankRange[item] = (r1, r2)

    B = nx.Graph(seed = seed)
    top_nodes = []
    bottom_nodes = []

    for i in single_ranking:
        top_nodes.append(i)
        bottom_nodes.append(str(i))
    B.add_nodes_from(top_nodes, bipartite=0)
    B.add_nodes_from(bottom_nodes, bipartite=1)

    for i in single_ranking:
        r1, r2 = rankRange[i]
        # print(r1,r2)
        for j in range(1, numberOfItem + 1):
            if j >= r1 and j <= r2:
                # print(i,j)
                B.add_edge(i, str(j), weight=abs(i - j))
            else:
                B.add_edge(i, str(j), weight=100000000000)
                # print(i,j)

    my_matching = nx.algorithms.bipartite.minimum_weight_full_matching(B, top_nodes, "weight")

    vy = list(my_matching.keys())  # v @ position y, where y is zero-indexed
    v = vy[0:numberOfItem]
    y = vy[numberOfItem:numberOfItem * 2]

    ranking = np.zeros(numberOfItem, dtype=int)
    for ind_i in range(0, numberOfItem):
        ranking[int(y[ind_i]) - 1] = v[ind_i]

    ranking = [int_to_string_cand[r] for r in ranking]
    return pd.DataFrame(ranking)