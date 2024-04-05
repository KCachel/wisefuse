import numpy as np
import pandas as pd
from experiment_core import *

#1. We have to create base rankings
adult = pd.read_csv('data/adult/adult.csv')
adult = adult.assign(candnumber = list(range(0, len(adult))))
adult = adult.loc[adult['class'] == '>50K']

# Rank by education-num
educationum_candrank = adult[['candnumber', 'education-num']].sort_values(by = ['education-num'],

                                                                          ignore_index = True, ascending=False)
educationnum= educationum_candrank[['education-num']]
educationum_candrank = educationum_candrank[['candnumber']]
educationum_candrank.rename(columns={'candnumber': 'RankByeducation-num'}, inplace=True)

# Rank by capital-gain
capitalgain_candrank = adult[['candnumber', 'capital-gain']].sort_values(by = ['capital-gain'],
                                                                         ignore_index = True, ascending=False)
capitalgain = capitalgain_candrank[['capital-gain']]
capitalgain_candrank = capitalgain_candrank[['candnumber']]
capitalgain_candrank.rename(columns={'candnumber': 'RankBycapital-gain'}, inplace=True)


# Rank by capital-loss
capitalloss_candrank = adult[['candnumber', 'capital-loss']].sort_values(by = ['capital-loss'],
                                                                         ignore_index = True, ascending=False)
capitalloss= capitalloss_candrank[['capital-loss']]
capitalloss_candrank = capitalloss_candrank[['candnumber']]
capitalloss_candrank.rename(columns={'candnumber': 'RankBycapital-loss'}, inplace=True)

# Rank by hours-per-week
hours_candrank = adult[['candnumber', 'hours-per-week']].sort_values(by = ['hours-per-week'],
                                                                     ignore_index = True, ascending=False)
hoursweek = hours_candrank[['hours-per-week']]
hours_candrank = hours_candrank[['candnumber']]
hours_candrank.rename(columns={'candnumber': 'RankByhours-per-week'}, inplace=True)

#Final ranking profile
adult_profile_df = pd.concat([educationum_candrank, capitalgain_candrank, capitalloss_candrank, hours_candrank], axis = 1)
adult_scores_df = pd.concat([educationnum, capitalgain, capitalloss, hoursweek], axis = 1)

#Uncomment to save to csv
# airbnb_profile_df.to_csv('data/airbnb/airbnb_profile_df.csv', index = False)


#2. We have to create the item group dictionary
np_profile = adult_profile_df.to_numpy()
unique_cands = list(np.unique(np_profile))
cand_race = [adult.loc[adult['candnumber'] == c]['race'].item() for c in unique_cands]
profile_item_group_dict = dict(zip(unique_cands, cand_race))


# Run experiment
profile_df = adult_profile_df
scores_df = adult_scores_df
fair_rep = 'EQUAL'
fusion = 'borda'
candidate_ids = list(np.unique(np_profile))
dataset_name = 'Adult'
csv_name = 'results/airbnb/results_adult.csv'

workflow(profile_df, scores_df, fair_rep, fusion, profile_item_group_dict, candidate_ids, dataset_name, csv_name)


