import numpy as np
import pandas as pd

def BORDA(preference_df, candidate_ids):
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

    ids = list(borda_scores.keys())
    new_scores = [borda_scores[cand] for cand in ids]
    scores, ordered_candidate_ids = zip(*sorted(zip(new_scores, ids), reverse=True))
    return pd.DataFrame(ordered_candidate_ids)