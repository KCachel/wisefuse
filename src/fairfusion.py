import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import normalize

def fair_fusion(base_ranks_df, scores_df, item_group_dict, reg_param, fairness_f, fusion):
    """
    WISE fusion
    :param base_ranks_df: Pandas dataframe for base rankings
    :param scores_df: Pandas dataframe for score associated with items in base rankings
    :param item_group_dict: Dictionary where keys are items and values are groups
    :param reg_param: Fairness control parameter in range (0,1], higher is more fair, typically .9
    :param fairness_f: 'EQUAL' or 'PROPORTIONAL' forms of fairness
    :param fusion: 'borda' or 'COMBmnz' standard fusion
    :return: Pandas dataframe of consensus
    """
    candidate_ids = list(item_group_dict.keys())
    if fusion == "borda":
        original_scores = __borda(base_ranks_df, candidate_ids)

    if fusion == "COMBmnz":
        original_scores = __combmnz(base_ranks_df, scores_df, candidate_ids)

    fx = np.asarray([original_scores[cand] for cand in candidate_ids])  # Numpy array of original fusion scores
    per_group_fusion, smallest_grp_size = make_group_fusion(original_scores, item_group_dict)

    A, rc_indx_to_cand_id = group_matrix(item_group_dict, per_group_fusion, fairness_f)
    if fairness_f == 'PROPORTIONAL':
        A = normalize(A, axis=1, norm='l1') #make rows sum to one
    D = diagnol_matrix(A)
    alpha = reg_param
    Dhalf = fractional_matrix_power(D, -0.5)
    S = Dhalf@ A @ Dhalf
    inner = np.identity(len(candidate_ids)) - (alpha*S)
    fx_star = fractional_matrix_power(inner, -1) @ fx

    updated_scores = dict(zip(candidate_ids, fx_star))
    ids = list(updated_scores.keys())
    new_scores = [updated_scores[cand] for cand in ids]
    scores, ordered_candidate_ids = zip(*sorted(zip(new_scores, ids), reverse=True))
    return pd.DataFrame(ordered_candidate_ids)


def __combmnz(preference_df, scores_df, candidate_ids):
    """
    Calculate COMBmnz score per item.
    :param preference_df: Dataframe of base ranking items
    :param scores_df: Dataframe of scores assigned to items
    :param candidate_ids: Ids of items
    :return: Dictionary of combmnz scores
    """
    num_rankings = len(preference_df.columns)
    rel_scores = {key: 0 for key in candidate_ids}
    norm = {key: 0 for key in candidate_ids}
    for r in range(0, num_rankings):
        single_ranking = preference_df[preference_df.columns[r]]  # isolate ranking
        single_ranking = np.array(
            single_ranking[~pd.isnull(single_ranking)]
        )

        for item_pos in range(0, len(single_ranking)):
            item = single_ranking[item_pos]
            rel_scores[item] += scores_df.iloc[item_pos, r]
            norm[item] += 1
    comb_scores = dict(zip(rel_scores.keys(), map(lambda x, y: x / y if y else 0, rel_scores.values(), norm.values())))
    return comb_scores

def __borda(preference_df, candidate_ids):
    """
    Calculate borda score per item.
    :param preference_df: Data frame of base ranking items
    :param candidate_ids:Ids of items
    :return: Dictionary of borda scores
    """
    num_rankings = len(preference_df.columns)
    borda_scores = {key: 0 for key in candidate_ids}
    num_items = len(candidate_ids) # use for borda count with same scores per ranking
    for r in range(0, num_rankings):
        single_ranking = preference_df[preference_df.columns[r]]  # isolate ranking
        single_ranking = np.array(
            single_ranking[~pd.isnull(single_ranking)]
        )  # drop any NaNs
        points_at_pos = list(range(num_items - 1, -1, -1))
        for item_pos in range(0, len(single_ranking)):
            item = single_ranking[item_pos]
            borda_scores[item] +=points_at_pos[item_pos]

    return borda_scores


def make_group_fusion(fusion_scores, item_group_dict):
    """
    Make per_group_rankings. A dictionary whose keys are group_ids, and each value is a list of lists where the first
    list contains candidate ids in fusion score order, and the second list contains corresponding fusion score.
    :param fusion_scores: Dictionary of item key and fusion scores as values.
    :param item_group_dict:
    :return:
    """
    candidate_ids = list(item_group_dict.keys())
    fx = [fusion_scores[cand] for cand in candidate_ids]
    fx, candidate_ids  = zip(*sorted(zip(fx, candidate_ids), reverse= True))
    unique_grps, grp_count_items = np.unique(
        list(item_group_dict.values()), return_counts=True
    )

    per_group_rankings = {key: [[],[]] for key in unique_grps}
    for indx in range(0, len(candidate_ids)):
        item = candidate_ids[indx]
        item_score = fx[indx]
        item_group = item_group_dict[item]
        per_group_rankings[item_group][0].append(item) # add item
        per_group_rankings[item_group][1].append(item_score)  # add score

    smallest_grp_size = np.min(grp_count_items)
    return per_group_rankings, smallest_grp_size

def group_matrix(item_group_dict, per_group_fusion, fairness_f):
    """
    Determine within group similarity matrix A
    :param item_group_dict: Dictionary where keys are items and values are groups
    :param per_group_fusion: Dictionary of groups as keys and lists of ordered items
    :param fairness_f: 'EQUAL' or 'PROPORTIONAL' forms of fairness
    :return: matrix A,
    """
    candidate_ids = list(item_group_dict.keys())
    rc_indx_to_cand_id = np.asarray(candidate_ids)
    num_candidates = len(candidate_ids)
    A = np.zeros((num_candidates, num_candidates))
    for indx_i in range(0, num_candidates):
        for indx_j in range(0, num_candidates):
            if indx_i != indx_j: # ignore diagnols
                cand_i = rc_indx_to_cand_id[indx_i]
                cand_j = rc_indx_to_cand_id[indx_j]
                if item_group_dict[cand_i] != item_group_dict[cand_j]: # only compare different groups
                    aff = similiarity(cand_i, cand_j, item_group_dict, per_group_fusion, fairness_f)
                    A[indx_i, indx_j] = aff
    return A, rc_indx_to_cand_id

def similiarity(cand_i, cand_j, item_group_dict, per_group_fusion, fairness_f):
    """
    Calculate similarity (eq. 2 or 3) in paper.
    :param cand_i: 1st item
    :param cand_j: 2nd item
    :param item_group_dict: Dictionary where keys are items and values are groups
    :param per_group_fusion: Dictionary of groups as keys and lists of ordered items
    :param fairness_f:  'EQUAL' or 'PROPORTIONAL' forms of fairness
    :return: aff, a within a group similarity score
    """
    global aff
    grp_cand_i = item_group_dict[cand_i]
    grp_cand_j = item_group_dict[cand_j]
    ranking_grp_cand_i = per_group_fusion[grp_cand_i][0]
    ranking_grp_cand_j = per_group_fusion[grp_cand_j][0]

    if fairness_f == "EQUAL":
        place_cand_i = ranking_grp_cand_i.index(cand_i) + 1  # adjust for zero index
        place_cand_j = ranking_grp_cand_j.index(cand_j) + 1  # adjust for zero index
        delta = np.abs(place_cand_i - place_cand_j)
        if delta < 1: #candidates in same place in the group
            aff = 1
        else:
            aff = 0.00001
    if fairness_f == "PROPORTIONAL":
        if len(ranking_grp_cand_i) >= len(ranking_grp_cand_j):
            max_c = cand_i
            min_c = cand_j
        else:
            max_c = cand_j
            min_c = cand_i
        ratio = (len(per_group_fusion[item_group_dict[max_c]][0]))/(len(per_group_fusion[item_group_dict[min_c]][0]))
        place_max = np.ceil((per_group_fusion[item_group_dict[max_c]][0].index(max_c) + 1)/ ratio)
        place_min = per_group_fusion[item_group_dict[min_c]][0].index(min_c) + 1  # adjust for zero index
        delta = np.abs(place_max - place_min)
        if delta < 1: #candidates in same place in the group
            aff = 1
        else:
            aff = 0.00001

    return aff

def diagnol_matrix(A):
    """
    Calcualte diagnol matrix
    :param A: Matrix A
    :return: Diagnol of Matrix A
    """
    vals = np.sum(A, axis = 0)
    num_candidates = np.shape(A)[0]
    D = np.zeros((num_candidates, num_candidates))
    for indx_i in range(0, num_candidates):
        for indx_j in range(0, num_candidates):
            if indx_i == indx_j:
                D[indx_i,indx_j] = vals[indx_i]
    return D


