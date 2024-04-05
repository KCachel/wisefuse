import numpy as np
import pandas as pd

def COMBmnz(base_ranking_df, scores_df, candidate_ids):
    """
    Calculate COMBmnz score per item.
    :param base_ranking_df: Dataframe of items
    :param scores_df: Dataframe of scores assigned to items
    :param candidate_ids: Ids of items
    :return: COMBmnz consensus ranking
    """
    num_rankings = len(base_ranking_df.columns)
    rel_scores = {key: 0 for key in candidate_ids}
    norm = {key: 0 for key in candidate_ids}
    for r in range(0, num_rankings):
        single_ranking = base_ranking_df[base_ranking_df.columns[r]]  # isolate ranking
        single_ranking = np.array(
            single_ranking[~pd.isnull(single_ranking)]
        )

        for item_pos in range(0, len(single_ranking)):
            item = single_ranking[item_pos]
            rel_scores[item] += scores_df.iloc[item_pos, r]
            norm[item] += 1
    comb_scores = dict(zip(rel_scores.keys(), map(lambda x, y: x / y if y else 0, rel_scores.values(), norm.values())))
    ids = list(comb_scores.keys())
    new_scores = [comb_scores[cand] for cand in ids]
    scores, ordered_candidate_ids = zip(*sorted(zip(new_scores, ids), reverse=True))
    return pd.DataFrame(ordered_candidate_ids), pd.DataFrame(scores)