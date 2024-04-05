# Script for metrics
# References: Geyik, S. C., Ambler, S., & Kenthapadi, K. (2019, July).
# Fairness-aware ranking in search & recommendation systems with application to linkedin talent search.
# In Proceedings of the 25th acm sigkdd international conference on knowledge discovery & data mining (pp. 2221-2231).

import numpy as np
import pandas as pd
from itertools import combinations


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

def avg_rbo(profile_df, consensus_df):
    num_rankings = len(profile_df.columns)
    rbo_vals = []
    for i in range(0, num_rankings):
        single_base_ranking = profile_df[profile_df.columns[i]]
        single_base_ranking =  single_base_ranking[~pd.isnull(single_base_ranking)]  # drop any NaNs
        rbo_vals.append(RankingSimilarity(single_base_ranking.tolist(), consensus_df[0].tolist()).rbo())
    return np.mean(rbo_vals)


def avg_wg_kt(ranking_a_df, ranking_b_df, item_group_dict):
    kt_per_group = []
    for grp in np.unique(list(item_group_dict.values())):
        rank_a = [c for c in ranking_a_df[0].tolist() if item_group_dict[c] == grp]
        rank_b = [c for c in ranking_b_df[0].tolist() if item_group_dict[c] == grp]
        kt_per_group.append(ktd(rank_a, rank_b))
    return np.mean(kt_per_group)

def avg_wg_rbo(ranking_a_df, ranking_b_df, item_group_dict):
    rbo_per_group = []
    for grp in np.unique(list(item_group_dict.values())):
        rank_a = [c for c in ranking_a_df[0].tolist() if item_group_dict[c] == grp]
        rank_b = [c for c in ranking_b_df[0].tolist() if item_group_dict[c] == grp]
        rbo_per_group.append(RankingSimilarity(rank_a, rank_b).rbo())
    return np.mean(rbo_per_group)


def ktd(rank_a, rank_b):

    tau = 0
    n_candidates = len(rank_a)
    a = list(range(0, len(rank_a))) #make ints
    b = [rank_a.index(cand) for cand in rank_b]
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(a[i] - a[j]) ==
                -np.sign(b[i] - b[j]))
    return tau



# pylint: disable=C0103, R0914, R0201
"""Main module for rbo."""
from typing import List, Optional, Union
from tqdm import tqdm


class RankingSimilarity:
    """
    This class will include some similarity measures between two different
    ranked lists.
    """
    def __init__(
        self,
        S: Union[List, np.ndarray],
        T: Union[List, np.ndarray],
        verbose: bool = False,
    ) -> None:
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ["a", "b", "c", "d", "e"]
        T = ["b", "a", 1, "d"]

        Both lists reflect the ranking of the items of interest, for example,
        list S tells us that item "a" is ranked first, "b" is ranked second,
        etc.

        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element"s position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
            verbose: If True, print out intermediate results. Default to False.
        """

        assert type(S) in [list, np.ndarray]
        assert type(T) in [list, np.ndarray]

        assert len(S) == len(set(S))
        assert len(T) == len(set(T))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder

    def assert_p(self, p: float) -> None:
        """Make sure p is between (0, 1), if so, assign it to self.p.

        Args:
            p (float): The value p.
        """
        assert 0.0 < p < 1.0, "p must be between (0, 1)"
        self.p = p

    def _bound_range(self, value: float) -> float:
        """Bounds the value to [0.0, 1.0]."""

        try:
            assert (0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print("Value out of [0, 1] bound, will bound it.")
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one

    def rbo(
        self,
        k: Optional[float] = None,
        p: float = 1.0,
        ext: bool = False,
    ) -> float:
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
        RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        If p = 1, it returns to the un-bounded set-intersection overlap,
        according to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it
        essentially control the "top-weightness". Simply put, to an extreme,
        a small p value will only consider first few items, whereas a larger p
        value will consider more items. See Eq. (21) for quantitative measure.

        Args:
            k: The depth of evaluation.
            p: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext: If True, we will extrapolate the rbo, as in Eq. (23).

        Returns:
            The rbo at depth k (or extrapolated beyond).
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float("inf")
        k = min(self.N_S, self.N_T, k)

        # initialize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            self.assert_p(p)
            weights = [1.0 * (1 - p) * p**d for d in range(k)]

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        for d in tqdm(range(1, k), disable=~self.verbose):

            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases did not happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p**k)

        return self._bound_range(AO[-1])

    def rbo_ext(self, p=0.98):
        """
        This is the ultimate implementation of the rbo, namely, the
        extrapolated version. The corresponding formula is Eq. (32) in the rbo
        paper.
        """

        self.assert_p(p)

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        # since we are dealing with un-even lists, we need to figure out the
        # long (L) and short (S) list first. The name S might be confusing
        # but in this function, S refers to short list, L refers to long list
        if len(self.S) > len(self.T):
            L, S = self.S, self.T
        else:
            S, L = self.S, self.T

        s, l = len(S), len(L)  # noqa

        # initialize the overlap and rbo arrays
        # the agreement can be simply calculated from the overlap
        X, A, rbo = [0] * l, [0] * l, [0] * l

        # first item
        S_running, L_running = {S[0]}, {L[0]}  # for O(1) look up
        X[0] = 1 if S[0] == L[0] else 0
        A[0] = X[0]
        rbo[0] = 1.0 * (1 - p) * A[0]

        # start the calculation
        disjoint = 0
        ext_term = A[0] * p

        for d in tqdm(range(1, l), disable=~self.verbose):
            if d < s:  # still overlapping in length

                S_running.add(S[d])
                L_running.add(L[d])

                # again I will revoke the DP-like step
                overlap_incr = 0  # overlap increment at step d

                # if the new items are the same
                if S[d] == L[d]:
                    overlap_incr += 1
                else:
                    # if the new item from S is in L already
                    if S[d] in L_running:
                        overlap_incr += 1
                    # if the new item from L is in S already
                    if L[d] in S_running:
                        overlap_incr += 1

                X[d] = X[d - 1] + overlap_incr
                # Eq. (28) that handles the tie. len() is O(1)
                A[d] = 2.0 * X[d] / (len(S_running) + len(L_running))
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                ext_term = 1.0 * A[d] * p**(d + 1)  # the extrapolate term

            else:  # the short list has fallen off the cliff
                L_running.add(L[d])  # we still have the long list

                # now there is one case
                overlap_incr = 1 if L[d] in S_running else 0

                X[d] = X[d - 1] + overlap_incr
                A[d] = 1.0 * X[d] / (d + 1)
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                X_s = X[s - 1]  # this the last common overlap
                # second term in first parenthesis of Eq. (32)
                disjoint += 1.0 * (1 - p) * (p**d) * \
                    (X_s * (d + 1 - s) / (d + 1) / s)
                ext_term = 1.0 * ((X[d] - X_s) / (d + 1) + X[s - 1] / s) * \
                    p**(d + 1)  # last term in Eq. (32)

        return self._bound_range(rbo[-1] + disjoint + ext_term)

    def top_weightness(
            self,
            p: Optional[float] = None,
            d: Optional[int] = None):
        """
        This function will evaluate the degree of the top-weightness of the
        rbo. It is the implementation of Eq. (21) of the rbo paper.

        As a sanity check (per the rbo paper),
        top_weightness(p=0.9, d=10) should be 86%
        top_weightness(p=0.98, d=50) should be 86% too

        Args:
            p (float), default None: A value between zero and one.
            d (int), default None: Evaluation depth of the list.

        Returns:
            A float between [0, 1], that indicates the top-weightness.
        """

        # sanity check
        self.assert_p(p)

        if d is None:
            d = min(self.N_S, self.N_T)
        else:
            d = min(self.N_S, self.N_T, int(d))

        if d == 0:
            top_w = 1
        elif d == 1:
            top_w = 1 - 1 + 1.0 * (1 - p) / p * (np.log(1.0 / (1 - p)))
        else:
            sum_1 = 0
            for i in range(1, d):
                sum_1 += 1.0 * p**(i) / i
            top_w = 1 - p**(i) + 1.0 * (1 - p) / p * (i + 1) * \
                (np.log(1.0 / (1 - p)) - sum_1)  # here i == d-1

        if self.verbose:
            print("The first {} ranks have {:6.3%} of the weight of "
                  "the evaluation.".format(d, top_w))

        return self._bound_range(top_w)