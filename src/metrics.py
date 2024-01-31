# Script for metrics
# References: Geyik, S. C., Ambler, S., & Kenthapadi, K. (2019, July).
# Fairness-aware ranking in search & recommendation systems with application to linkedin talent search.
# In Proceedings of the 25th acm sigkdd international conference on knowledge discovery & data mining (pp. 2221-2231).

import numpy as np
import pandas as pd


def NDKL_onebyone(ranking_df, item_group_dict, fair_rep):
    """
    Calculate Normalized Discounted KL-Divergence Score (Geyik et al.) at all prefixes.
    :param ranking_df: Pandas dataframe of ranking(s).
    :param item_group_dict: Dictionary of items (keys) and their group membership (values).
    :return: NDKL value.
    """
    if len(ranking_df.columns) > 1:
        raise AssertionError("NDKL can only be calculated on a single ranking.")

    single_ranking = ranking_df[ranking_df.columns[0]]  # isolate ranking
    single_ranking = np.array(
        single_ranking[~pd.isnull(single_ranking)]
    )  # drop any NaNs

    group_ids = [item_group_dict[c] for c in single_ranking]
    unique_grps = np.unique(group_ids)
    group_ids = np.asarray(
        [np.argwhere(unique_grps == grp_of_item)[0, 0] for grp_of_item in group_ids]
    )
    num_groups = len(unique_grps)
    num_items = len(group_ids)

    if fair_rep == 'PROPORTIONAL':
        # dr = __distributions(group_ids, num_groups)  # Distributions per group
        dr = __distributions(group_ids, num_groups)  # Distributions per group
    if fair_rep == 'EQUAL':
        dr = np.tile((1 / (num_groups)), num_groups)  # for more equal chunks
    Z = __Z_Vector(num_items)  # Array of Z scores

    # Eq. 4 in Geyik et al.
    return (1 / np.sum(Z)) * np.sum(
        [
            Z[i]
            * __kl_divergence(__distributions(group_ids[0 : i + 1], num_groups), dr)
            for i in range(0, num_items)
        ]
    )




def NDKL_groupchunks(ranking_df, item_group_dict, fair_rep):
    """
    Calculate Normalized Discounted KL-Divergence Score (Geyik et al.) where chunks are num group increments.
    :param ranking_df: Pandas dataframe of ranking(s).
    :param item_group_dict: Dictionary of items (keys) and their group membership (values).
    :param fair_rep EQUAL or PROPORTIONAL
    :return: NDKL value.
    """
    if len(ranking_df.columns) > 1:
        raise AssertionError("NDKL can only be calculated on a single ranking.")

    single_ranking = ranking_df[ranking_df.columns[0]]  # isolate ranking
    single_ranking = np.array(
        single_ranking[~pd.isnull(single_ranking)]
    )  # drop any NaNs

    group_ids = [item_group_dict[c] for c in single_ranking]
    unique_grps = np.unique(list(item_group_dict.values()))
    group_ids = np.asarray(
        [np.argwhere(unique_grps == grp_of_item)[0, 0] for grp_of_item in group_ids]
    )
    all_groups = np.asarray(list(item_group_dict.values()))
    all_group_ids = np.asarray(
        [np.argwhere(unique_grps == grp_of_item)[0, 0] for grp_of_item in all_groups]
    )
    num_groups = len(unique_grps)
    num_items = len(single_ranking)

    if fair_rep == 'PROPORTIONAL':
        #dr = __distributions(group_ids, num_groups)  # Distributions per group
        dr = __distributions(all_group_ids, num_groups)  # Distributions per group
    if fair_rep == 'EQUAL':
        dr = np.tile((1/(num_groups)), num_groups) #for more equal chunks
      # Array of Z scores

    chunks = list(range(num_groups, num_items + num_groups,num_groups))
    Z = __Z_Vector(len(chunks))
    vals = []
    for ind in range(0, len(list(range(num_groups, num_items+ num_groups,num_groups)))):
        end_prefix = chunks[ind]
        P = __distributions(group_ids[0 : end_prefix], num_groups)
        kl = __kl_divergence(P, dr)
        vals.append(Z[ind]*kl)
    result = (1 / np.sum(Z)) * np.sum(vals)
    return result


def NDKL_allpos(ranking_df, item_group_dict, fair_rep):
    """
    Calculate Normalized Discounted KL-Divergence Score (Geyik et al.).
    :param ranking_df: Pandas dataframe of ranking(s).
    :param item_group_dict: Dictionary of items (keys) and their group membership (values).
    :return: NDKL value.
    """
    if len(ranking_df.columns) > 1:
        raise AssertionError("NDKL can only be calculated on a single ranking.")

    single_ranking = ranking_df[ranking_df.columns[0]]  # isolate ranking
    single_ranking = np.array(
        single_ranking[~pd.isnull(single_ranking)]
    )  # drop any NaNs

    group_ids = [item_group_dict[c] for c in single_ranking]
    unique_grps = np.unique(list(item_group_dict.values()))
    group_ids = np.asarray(
        [np.argwhere(unique_grps == grp_of_item)[0, 0] for grp_of_item in group_ids]
    )
    all_groups = np.asarray(list(item_group_dict.values()))
    all_group_ids = np.asarray(
        [np.argwhere(unique_grps == grp_of_item)[0, 0] for grp_of_item in all_groups]
    )
    num_groups = len(unique_grps)
    num_items = len(single_ranking)

    if fair_rep == 'PROPORTIONAL':
        #dr = __distributions(group_ids, num_groups)  # Distributions per group
        dr = __distributions(all_group_ids, num_groups)  # Distributions per group
    if fair_rep == 'EQUAL':
        dr = np.tile((1/(num_groups)), num_groups) #for more equal chunks
    Z = __Z_Vector(num_items)  # Array of Z scores

    #Eq. 4 in Geyik et al.
    return (1 / np.sum(Z)) * np.sum(
        [
            Z[i]
            * __kl_divergence(__distributions(group_ids[0 : i + 1], num_groups), dr)
            for i in range(0, num_items)
        ]
    )


def __distributions(ranking, num_groups):
    """
    Calculate the proportion of each group
    :param ranking: Numpy array of group id represented in the ranking.
    :param num_groups: Int, number of distinct groups
    :return: Numpy array of each group's proportion.
    """
    return np.array(
        [((ranking == i).sum()) / len(ranking) for i in range(0, num_groups)]
    )


def __Z_Vector(k):
    """
    Calculate Z score
    :param k: Int, position of ranking.
    :return: Numpy array of Z values.
    """
    return 1 / np.log2(np.array(range(0, k)) + 2)

def __kl_divergence(p, q):
    """
    Calculate KL-Divergence between P and Q, with epsilon to avoid divide by zero.
    :param p: Numpy array p distribution.
    :param q: Numpy array q distribution.
    :return: KL-Divergence score.
    """
    epsilon = 0.0000001  # Epsilon is used here to avoid P or Q is equal to 0. "
    p = p + epsilon
    q = q + epsilon

    return np.sum(p * np.log(p / q))