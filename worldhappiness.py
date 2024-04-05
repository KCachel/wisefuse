import numpy as np
import pandas as pd
from experiment_core import *
import pycountry_convert as pc

# Quick helper function to convert a country to its continent
def country_to_continent(country_name):
   #Manual outlies
    if country_name == 'Congo (Brazzaville)':
       return 'Africa'
    if country_name == 'Congo (Kinshasa)':
        return 'Africa'
    if country_name == 'Hong Kong S.A.R. of China':
        return 'Asia'
    if country_name == 'Kosovo':
        return 'Europe'
    if country_name == 'Somaliland region':
        return 'Africa'
    if country_name == 'State of Palestine':
        return 'Middle East'
    if country_name == 'Taiwan Province of China':
        return 'Asia'
    if country_name == 'Turkey':
        return 'Asia'
    else:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name


# Dataset Preparation
#1. Create ranking profile
happiness = pd.read_excel('data\world-happiness\DataForTable2.1WHR2023.xls')
happiness = happiness.dropna()


#Add the region sensitive attribute
countries_np = happiness['Country name'].to_numpy()
continent_regions = [country_to_continent(c) for c in countries_np]
happiness['Regions'] = continent_regions

#Drop countries where region is singleton or alike
happiness = happiness[~happiness['Regions'].isin(['Middle East', 'Oceania'])]


rank_store = []
score_store = []
for yr in range(2006, 2022+1, 1):
    filter= happiness.loc[happiness['year'] == yr]
    rank = filter[['Country name', 'Life Ladder']].sort_values(by = ['Life Ladder'],
                                                                                     ignore_index = True, ascending=False)
    scores = rank[['Life Ladder']]
    rank = rank[['Country name']]
    rank.rename(columns={'Country name': str(yr)}, inplace=True)
    rank_store.append(rank)
    score_store.append(scores)

#Final preference profile
happiness_profile_df = pd.concat(rank_store,  axis = 1)
scores_df = pd.concat(score_store, axis = 1)

np_profile = happiness_profile_df.to_numpy()
np_prof_flat = np_profile.flatten()
np_prof_flat = np_prof_flat[~pd.isnull(np_prof_flat)]
candidate_ids = list(np.unique(np_prof_flat))


print("There are ", len(candidate_ids), " candidates in the ranking profile.")

candidates_col = 'Country name'
sa_col = 'Regions'

#Profile item group dict
# grps = []
# for e in candidate_ids:
#     print(e)
#     grps.append(happiness.loc[happiness[candidates_col] == e][sa_col][0])

ranked_grps = [happiness[happiness[candidates_col] == e][sa_col].to_list()[0] for e in candidate_ids]
profile_item_group_dict = dict(zip(candidate_ids, ranked_grps))

# Run experiment
profile_df = happiness_profile_df

fusion = 'borda'
dataset_name = 'World Happiness'

fair_rep = 'PROPORTIONAL'
csv_name = 'results/world-happiness/results_worldhappiness.csv'
workflow(profile_df, scores_df, fair_rep, fusion, profile_item_group_dict, candidate_ids, dataset_name, csv_name)




