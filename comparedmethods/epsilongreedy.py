import pandas as pd
import numpy as np
from comparedmethods import *
import copy

# References: Feng, Y., & Shah, C. (2022, June).
# Has CEO gender bias really been fixed? adversarial attacking and improving gender fairness in image search.
# In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 11, pp. 11882-11890).


def post_epsilongreedy(base_rankings_df, scores_df, candidate_ids, item_group_dict, epsilon, seed, fusion):
    if fusion == "borda":
        consensus, cr_scores_df = BORDA(base_rankings_df, candidate_ids)
    if fusion == "COMBmnz":
        consensus, cr_scores_df = COMBmnz(base_rankings_df, scores_df, candidate_ids)
    return _epsilon_greedy(consensus, item_group_dict, cr_scores_df, epsilon, seed)

def pre_epsilongreedy(preference_df, scores_df, candidate_ids, item_group_dict, epsilon, seed, fusion):
    pref = preference_df.copy(deep=True)
    scor = scores_df.copy(deep=True)
    num_rankings = len(pref.columns)
    np_pref = pref.to_numpy()
    np_scores = scor.to_numpy()
    for r in range(0, num_rankings):
        single_ranking = np_pref[:, r]  # isolate ranking
        single_ranking = pd.DataFrame(np.array(single_ranking[~pd.isnull(single_ranking)]))  # drop any NaNs
        single_scores = np_scores[:, r]  # isolate scores
        single_scores = pd.DataFrame(np.array(single_scores[~pd.isnull(single_scores)]))  # drop any NaNs
        updated_ranking, updated_scores = _epsilon_greedy(single_ranking, item_group_dict, single_scores, epsilon, seed)
        # update profile
        np_pref[0: len(updated_ranking), r] = updated_ranking.to_numpy().flatten()
        np_scores[0: len(updated_scores), r] = updated_scores.to_numpy().flatten()
    prefdata_df = pd.DataFrame(np_pref)
    scoresdata_df = pd.DataFrame(np_scores)
    if fusion == "borda":
        consensus, cr_scores_df = BORDA(prefdata_df, candidate_ids)
    if fusion == "COMBmnz":
        consensus, cr_scores_df = COMBmnz(prefdata_df, scoresdata_df, candidate_ids)
    return consensus, cr_scores_df

def _epsilon_greedy(current_ranking_df, item_group_dict, current_ranking_scores_df, epsilon, seed):
    """
    Epsilon-Greedy reranking algorithm.
    :param current_ranking_df: Pandas dataframe of ranking to be reranked.
    :param item_group_dict: Dictionary of items (keys) and their group membership (values).
    :param current_ranking_scores_df: Pandas dataframe of relevance scores associated with each item in the ranking.
    :param epsilon: Float epsilon value in [0,1].
    :param seed: Random seed value for reproducibility.
    :return: reranking, Pandas dataframe of items,item_group_reranked_dict, dictionary of items and group membership,  Pandas dataframe  of scores for reranking,
    """

    # Convert dataframes to numpy arrays
    current_ranking = current_ranking_df[0].to_numpy()
    current_group_ids = np.asarray([item_group_dict[i] for i in current_ranking])
    current_ranking_scores = current_ranking_scores_df[0].to_numpy()

    ranking = list(current_ranking)
    curr_ranking = copy.deepcopy(ranking)
    np.random.seed(seed)  # for reproducibility
    reranking = []
    for i in range(len(curr_ranking)):
        p = np.random.rand()
        if (
            p <= epsilon and i < len(curr_ranking) - 1
        ):  # swap items & can't swap last item
            temp = curr_ranking[i]
            j = np.random.randint(i + 1, len(curr_ranking))
            curr_ranking[i] = curr_ranking[j]
            curr_ranking[j] = temp
            reranking.append(curr_ranking[i])
        else:  # keep original ranking
            reranking.append(curr_ranking[i])

    reranking = np.asarray(reranking)
    reranking_scores = np.asarray(
        [current_ranking_scores[ranking.index(item)] for item in reranking]
    )
    reranking_ids = np.asarray(
        [current_group_ids[ranking.index(item)] for item in reranking]
    )
    item_group_reranked_dict = dict(zip(reranking, reranking_ids))
    return (
        pd.DataFrame(reranking),
        #item_group_reranked_dict,
        pd.DataFrame(reranking_scores),
    )