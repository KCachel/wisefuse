import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power

def fair_fusion(preference_df, item_group_dict, reg_param):
    """
    Function to calculate fair fusion scores.
    :param preference_df:
    :param item_group_dict:
    :param reg_param:
    :return:
    """
    candidate_ids = list(item_group_dict.keys())
    borda_scores = borda(preference_df, candidate_ids)

    fx = np.asarray([borda_scores[cand] for cand in candidate_ids]) # Numpy array of borda fusion scores
    fx_cand_id = np.asarray(candidate_ids) # Numpy array of candidate id to help index fx
    per_group_fusion, smallest_grp_size = make_group_fusion(borda_scores, item_group_dict)

    W, rc_indx_to_cand_id = group_affinity_matrix(borda_scores, item_group_dict, per_group_fusion, smallest_grp_size)
    D = diagnol_matrix(W)

    alpha = 1 /(1+reg_param)
    Dhalf = fractional_matrix_power(D, -0.5)
    S = Dhalf@ W @ Dhalf
    inner = np.identity(len(candidate_ids)) - alpha*S
    fx_star = fractional_matrix_power(inner, -0.5) @ fx
    fx_star = (1- alpha)*fx_star
    updated_scores = dict(zip(candidate_ids, fx_star))

    ids = list(updated_scores.keys())
    new_scores = [updated_scores[cand] for cand in ids]
    scores, ordered_candidate_ids = zip(*sorted(zip(new_scores, ids), reverse=True))
    return pd.DataFrame(ordered_candidate_ids)


def borda(preference_df, candidate_ids):
    """
    Calculate borda score per item.
    :param preference_df:
    :param candidate_ids:
    :return:
    """
    num_rankings = len(preference_df.columns)
    borda_scores = {key: 0 for key in candidate_ids}
    num_items = len(candidate_ids) # use for borda count with same scores per ranking
    for r in range(0, num_rankings):
        single_ranking = preference_df[preference_df.columns[r]]  # isolate ranking
        single_ranking = np.array(
            single_ranking[~pd.isnull(single_ranking)]
        )  # drop any NaNs
        # num_items = len(single_ranking) # use for borda coutn with different scores per ranking
        points_at_pos = list(range(num_items - 1, -1, -1))
        for item_pos in range(0, len(single_ranking)):
            item = single_ranking[item_pos]
            borda_scores[item] +=points_at_pos[item_pos]

    return borda_scores


def make_group_fusion(fusion_scores, item_group_dict):
    """
    Make per_group_rankings. A dictionary whose keys are group_ids, and each value is a list of lists where the first
    list contains candidate ids in fusion score order, and the second list contains corresponding fusion score.
    :param fusion_scores:
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

def group_affinity_matrix(fusion_scores, item_group_dict, per_group_fusion, smallest_grp_size):
    """
    Determine matrix W
    :param fusion_scores:
    :param item_group_dict:
    :return:
    """
    candidate_ids = list(item_group_dict.keys())
    rc_indx_to_cand_id = np.asarray(candidate_ids)
    num_candidates = len(candidate_ids)
    W = np.zeros((num_candidates, num_candidates))
    for indx_i in range(0, num_candidates):
        for indx_j in range(0, num_candidates):
            if indx_i != indx_j: # ignore diagnols
                cand_i = rc_indx_to_cand_id[indx_i]
                cand_j = rc_indx_to_cand_id[indx_j]
                if 0 == 0: #item_group_dict[cand_i] != item_group_dict[cand_j]: # only compare different groups
                    aff = cross_group_affinity(cand_i, cand_j, item_group_dict, per_group_fusion, smallest_grp_size)
                    W[indx_i, indx_j] = aff
    return W, rc_indx_to_cand_id

def cross_group_affinity(cand_i, cand_j, item_group_dict, per_group_fusion, smallest_grp_size):
    grp_cand_i = item_group_dict[cand_i]
    grp_cand_j = item_group_dict[cand_j]
    ranking_grp_cand_i = per_group_fusion[grp_cand_i][0]
    ranking_grp_cand_j = per_group_fusion[grp_cand_j][0]

    place_cand_i = ranking_grp_cand_i.index(cand_i) + 1 # adjust for zero index
    place_cand_j = ranking_grp_cand_j.index(cand_j) + 1  # adjust for zero index
    delta = np.abs(place_cand_i - place_cand_j)
    # aff = 1 / (delta + 1)
    if delta < smallest_grp_size - 2: #adjust for zero index delta min
        aff = 1 / (delta + 1)
    else:
        aff = 0.00001
    return aff

def diagnol_matrix(W):
    vals = np.sum(W, axis = 0)
    num_candidates = np.shape(W)[0]
    D = np.zeros((num_candidates, num_candidates))
    for indx_i in range(0, num_candidates):
        for indx_j in range(0, num_candidates):
            if indx_i == indx_j:
                D[indx_i,indx_j] = vals[indx_i]
    return D








#
# #preference_df = pd.read_csv('testdata.csv')
# # item_group_dict = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g'], [0, 0, 0, 0, 1, 1, 1]))
# preference_df = pd.read_csv('../testadvmin.csv')
# item_group_dict_adv_min = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g'], [0, 0, 1, 0, 1, 1, 1]))
# print(regularized_group_affinity_pa(preference_df, item_group_dict_adv_min, .000001))