import numpy as np
import pandas as pd
from experiment_core import *
import pycountry_convert as pc

# Dataset Preparation
#1. We have to create a preference profile
econf = pd.read_excel('data\econf\efotw-2023-master-index-data-for-researchers-iso.xlsx')

rank_store = []
score_store = []
#reporting was every 5 years
for yr in range(1975, 2000, 5):
    filter= econf.loc[econf['Year'] == yr]
    rank = filter[['Countries', 'Economic Freedom Summary Index']].sort_values(by = ['Economic Freedom Summary Index'],
                                                                                     ignore_index = True, ascending=False)
    econ = rank[['Economic Freedom Summary Index']]
    rank = rank[['Countries']]
    rank.rename(columns={'Countries': str(yr)}, inplace=True)
    rank_store.append(rank)
    score_store.append(econ)
#reporting became annual
for yr in range(2017, 2021+1, 1):
    filter= econf.loc[econf['Year'] == yr]
    rank = filter[['Countries', 'Economic Freedom Summary Index']].sort_values(by = ['Economic Freedom Summary Index'],
                                                                                     ignore_index = True, ascending=False)
    econ = rank[['Economic Freedom Summary Index']]
    rank = rank[['Countries']]
    rank.rename(columns={'Countries': str(yr)}, inplace=True)
    rank_store.append(rank)
    score_store.append(econ)

#Final preference profile
econf_profile_df = pd.concat(rank_store,  axis = 1)
econf_scores_df = pd.concat(score_store, axis = 1)

#drop nans for scores
econf_profile_df = econf_profile_df.loc[0:106, :] #Top 40 candidates for each ranker
econf_scores_df = econf_scores_df.loc[0:106, :] #Top 40 candidates for each ranker


#2. We have to create the item group dictionary
np_profile = econf_profile_df.to_numpy()
np_prof_flat = np_profile.flatten()
np_prof_flat = np_prof_flat[~pd.isnull(np_prof_flat)]
candidate_ids = list(np.unique(np_prof_flat))
ranked_grps = [econf[econf['Countries'] == e]['World Bank Region'].to_list()[0] for e in candidate_ids]
profile_item_group_dict = dict(zip(candidate_ids, ranked_grps))


print("There are ", len(candidate_ids), " candidates in the ranking profile.")

# Run experiment
profile_df = econf_profile_df
scores_df = econf_scores_df
fair_rep = 'EQUAL'
fusion = 'COMBmnz'
dataset_name = 'Econ-Freedom'
csv_name = 'results/econf/results_econfreedom.csv'

workflow(profile_df, scores_df, fair_rep, fusion, profile_item_group_dict, candidate_ids, dataset_name, csv_name)

