##################################
## GREEDY CELIS-STRASZAK-VISHNO-ICALP-2018
# References: https://github.com/AnayMehrotra/FairRankingWithNoisyAttributes/blob/master/algorithms.py
##################################
import numpy as np
import time
def greedy_fair_ranking(W, PT, L, U, getBirkhoff=True, verbose=False):
    # Solves the noisy fair LP then uses birkhoff_von_neumann_decomposition
    m = W.shape[0]
    n = W.shape[1]
    g = PT.shape[0]

    # ensure that each item has exactly one protected attribute
    for i in range(m): assert sum(PT[:, i]) == 1

    # ensure there are no lower bounds
    assert np.sum(np.abs(L)) == 0


    st = time.time()
    # iterator for each group
    g_it = np.zeros(g, dtype=int)
    # list of id's of items in each of the groups
    g_id = [[] for l in range(g)]
    # list of utilities of items in each group
    g_W = [[] for l in range(g)]
    # indices of items in each group sorted in decreasing order of utility
    g_argsort = [[] for l in range(g)]

    for i in range(m):
        for l in range(g):
            if PT[l][i] == 1:
                g_id[l].append(i) # add id to list
                g_W[l].append(W[i][0]) # add utility to list

    # set sorted indices of the utility
    for l in range(g): g_argsort[l] = np.argsort(-np.array(g_W[l]))

    # ranking
    x = np.zeros((m,n))

    for j in range(n): # iterate over positions
        mx = -1
        mx_grp = -1 # store the best group with a slack fairness constraint

        cur_ind = np.zeros(g, dtype=int) # index of the best unranked candidate in each group

        for l in range(g):
            if g_it[l] >= U[l,j] or g_it[l] >= len(g_id[l]): continue # cannot add candidate from group l

            cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

            if g_W[l][cur_ind[l]] > mx:
                mx = g_W[l][cur_ind[l]]
                mx_grp = l # chose group l

        if mx_grp == -1:
            # allow the ranking to violate the fairness constraints
            # (by selecting the itiem with the highest utility from the remaining items)
            for l in range(g):
                if g_it[l] >= len(g_id[l]): continue # cannot add candidate from group l

                cur_ind[l] = g_argsort[l][g_it[l]] # index of the best unranked candidate in group l

                if g_W[l][cur_ind[l]] > mx:
                    mx = g_W[l][cur_ind[l]]
                    mx_grp = l # chose group l

        # ensure problem is feasible
        assert mx_grp != -1

        mx_id = g_id[mx_grp][cur_ind[mx_grp]] # id of the best unranked candidate from the chosen group
        x[mx_id, j] = 1 # place candidate at position j
        g_it[mx_grp] += 1 # record that we ranked anoter candidate from group mx_grp

    if verbose: print(f'Time taken for Greedy: {time.time() - st}')
    return x