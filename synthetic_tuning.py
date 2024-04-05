import numpy as np
import pandas as pd
from experiment_core import *
import pycountry_convert as pc

dataset_df = pd.read_excel('data/synthetic-study/mallows_df.xlsx')
candidates_col = 'Candidates'
sa_col = 'Sensitive'
dataset_df[candidates_col] = dataset_df[candidates_col].apply(str)
dataset_df = pd.DataFrame(dataset_df.to_numpy())
dataset_df.rename(columns={0: candidates_col}, inplace=True)
dataset_df.rename(columns={1: sa_col}, inplace=True)
disp = .1
filename = "data\synthetic-study\profile_disp_" + str(disp) + "_.csv"
data = np.genfromtxt(filename, delimiter=',', dtype=int)
data = data.T
profile_df = pd.DataFrame(data)
profile_df = profile_df.astype(str)
np_profile = profile_df.to_numpy()
candidate_ids = list(np.unique(np_profile))
dataset_name = 'synthetic-' + str(disp)
ranked_grps = [dataset_df.loc[dataset_df[candidates_col] == e][sa_col].item() for e in candidate_ids]
profile_item_group_dict = dict(zip(candidate_ids, ranked_grps))
scores_df = pd.DataFrame(data)

fusion = 'borda'
csv_name = 'results/synthetic-study/tuningTestFQ.csv'
fair_rep = 'EQUAL'
workflow(profile_df, scores_df, fair_rep, fusion, profile_item_group_dict, candidate_ids, dataset_name, csv_name)