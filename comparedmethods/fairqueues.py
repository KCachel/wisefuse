from comparedmethods import *
from copy import deepcopy
import numpy as np
import pandas as pd

# References: https://github.com/ejohnson0430/fair_online_ranking/blob/master/algorithms/fair_queues.py
# https://github.com/ejohnson0430/fair_online_ranking/blob/master/algorithms/can_be_fair.py
# https://github.com/ejohnson0430/fair_online_ranking/blob/master/algorithms/metrics.py


def post_fairqueues(base_ranking_df, scores_df, candidate_ids, item_group_dict, fusion, ddp_thresh):
    if fusion == "borda":
        consensus, cr_scores_df = BORDA(base_ranking_df, candidate_ids)
    if fusion == "COMBmnz":
        consensus, cr_scores_df = COMBmnz(base_ranking_df, scores_df, candidate_ids)
    return _fairqueues(consensus, item_group_dict, cr_scores_df, ddp_thresh)

def pre_fairqueues(preference_df, scores_df, candidate_ids, item_group_dict, fusion, ddp_thresh):
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
        updated_ranking, updated_scores = _fairqueues(single_ranking, item_group_dict, single_scores, ddp_thresh)
        #update profile
        np_pref[0: len(updated_ranking), r] = updated_ranking.to_numpy().flatten()
        np_scores[0: len(updated_scores), r] = updated_scores.to_numpy().flatten()
    prefdata_df = pd.DataFrame(np_pref)
    scoresdata_df = pd.DataFrame(np_scores)
    if fusion == "borda":
        consensus, cr_scores_df = BORDA(prefdata_df, candidate_ids)
    if fusion == "COMBmnz":
        consensus, cr_scores_df = COMBmnz(prefdata_df, scoresdata_df, candidate_ids)
    return consensus, cr_scores_df

def _fairqueues(single_ranking, item_group_dict, single_scores, ddp_thresh):
    """
    Function to format to Gupta et al. code. Note that their code assumes score is a unique id.
    :param single_ranking: Dataframe ranking items
    :param item_group_dict: Dictionary where keys are items and values are groups
    :param single_scores: Dataframe of item scores
    :ddp_thresh: DDP value
    :return:
    """
    num_groups = len(np.unique(list(item_group_dict.values())))
    group2int = dict(zip(list(np.unique(list(item_group_dict.values()))), list(range(0, num_groups))))
    groups = [group2int[item_group_dict[a]] for a in single_ranking[0].tolist()]
    positions = [float(i) for i in list(range(1, len(single_ranking)+1))]
    scores = single_scores[0].tolist()
    # IF scores are the same we have to add a slight offset
    multiples_dict = {key: 0 for key in scores}
    for i in range(len(scores)):
        multiples_dict[scores[i]] += 1  # increase the occurrence count
        if multiples_dict[scores[i]] > 1:  # occurred more than once
            scores[i] = scores[i] +multiples_dict[scores[i]]/100000000  # if yes append an teeny val
    updatedscores2cand = dict(zip(scores, single_ranking[0].tolist()))
    updatedscores2scores = dict(zip(scores, single_scores[0].tolist()))
    batch = np.column_stack([positions, groups, scores]).astype(float)
    past_exposures = np.zeros([num_groups,2])
    past_exposures = append_exposures(batch, past_exposures)
    result = fair_rerank(batch, past_exposures, ddp_thresh, num_groups)
    result_ranking = pd.DataFrame([updatedscores2cand[a] for a in result[:,2]])
    result_scores = pd.DataFrame([updatedscores2scores[a] for a in result[:,2]])
    return result_ranking, result_scores


#Below code from https://github.com/ejohnson0430/fair_online_ranking

def fair_rerank(batch, past_exposures, ddp_thresh, num_groups):
    qs = np.zeros(num_groups, dtype=int)
    groups, queue_sizes = np.unique(batch[:,1].astype(int), return_counts=True)
    qs[groups] = queue_sizes
    queues = {}
    for j in groups:
        queues[j] = np.sort(batch[batch[:,1]==j,2])[::-1]
    curr_ranking = np.zeros([0,3])
    while len(queues) > 1:
        # try top member of each group ordered by relevance
        found = False
        for j in sorted(queues, key=lambda j: queues[j][0], reverse=True):
            sim_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
            sim_qs = deepcopy(qs)
            sim_qs[j] -= 1
            if can_be_fair(sim_ranking, past_exposures, sim_qs, ddp_thresh):
                found = True
                curr_ranking = sim_ranking
                qs = sim_qs
                queues[j] = queues[j][1:]
                if len(queues[j])==0: del queues[j]
                break
        # if heuristic cannot meet constraint, add from group w/ lowest exposure
        if not found:
            j = min_exposure_heuristic(curr_ranking, qs, past_exposures)
            curr_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
            qs[j] -= 1
            queues[j] = queues[j][1:]
            if len(queues[j])==0: del queues[j]
    # add candidates from remaining group
    for j in queues:
        for rel in queues[j]:
            curr_ranking = append_to_ranking(curr_ranking, j, rel)
    return curr_ranking

def group_exposures(batch, num_groups):
    exposures = np.zeros([num_groups,2])
    for j in range(num_groups):
        inds = batch[:,1]==j
        exposures[j,1] = np.sum(inds)
        exposures[j,0] = np.sum(1 / np.log2(1 + batch[inds, 0]))
    return exposures

def append_exposures(batch, past_exposures):
    return past_exposures + group_exposures(batch, past_exposures.shape[0])

def get_mean_exposures(past_exposures):
    return past_exposures[:,0] / np.maximum(past_exposures[:,1], 1)

def append_to_ranking(ranking, j, rel):
    return np.vstack((ranking, np.array([len(ranking) + 1, j, rel])))

def min_exposure_heuristic(curr_ranking, qs, past_exposures):
    h = heuristic(curr_ranking, qs)
    mean_exposures = get_mean_exposures(past_exposures + h)
    mean_exposures[qs==0] = np.inf
    return np.argmin(mean_exposures)

def can_be_fair(curr_ranking, past_exposures, _qs, ddp_thresh):
    # curr_ranking should be the ranking up to this point
    # past_exposures should be a G x 2 matrix
    # qs is the list of queue sizes
    # ddp_thresh should be obvious
    qs = deepcopy(_qs)
    sim_ranking = deepcopy(curr_ranking)
    while(np.sum(qs) > 0):
        j = min_exposure_heuristic(sim_ranking, qs, past_exposures)
        sim_ranking = append_to_ranking(sim_ranking, j, 0)
        qs[j] -= 1
    sim_exposures = append_exposures(sim_ranking, past_exposures)
    sim_ddp = DDP(sim_exposures)
    return sim_ddp <= ddp_thresh

def heuristic(curr_ranking, qs):
    h = np.zeros([qs.shape[0],2])
    Nr = curr_ranking.shape[0]
    Nu = np.sum(qs)
    remaining_ranks = np.arange(Nr, Nr+Nu) + 1
    avg_remaining_exposure = np.mean(1 / np.log2(1 + remaining_ranks))
    exposures = group_exposures(curr_ranking, len(qs))
    for group, remaining_ct in enumerate(qs):
        h[group,1] = remaining_ct + exposures[group,1]
        h[group,0] = remaining_ct * avg_remaining_exposure + exposures[group,0]
    return h

def DDP(past_exposures, min_count=5):
    # return largest difference in mean exposures
    included = past_exposures[:,1] >= min_count
    if np.sum(included) < 2: return 0
    mean_exposures = past_exposures[included,0] / past_exposures[included,1]
    return np.max(mean_exposures) - np.min(mean_exposures)

def fair_online(batches, ddp_thresh, num_groups, past_exposures=None, debug=False):
    processed_batches = [batch.copy() for batch in batches]
    if past_exposures is None: past_exposures = np.zeros([num_groups,2])
    for i, batch in enumerate(processed_batches):
        processed_batches[i] = fair_rerank(batch, past_exposures, ddp_thresh, num_groups)
        result = fair_rerank(batch, past_exposures, ddp_thresh, num_groups)
        past_exposures = append_exposures(processed_batches[i], past_exposures)
        ddp = DDP(past_exposures)
        if debug and ddp > ddp_thresh:
            print(f'FA*IR did not meet constraint on batch {i}: DDP {ddp}')
    return processed_batches

