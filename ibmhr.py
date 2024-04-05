import numpy as np
import pandas as pd
from experiment_core import *

#1. We have to create base rankings
ibmhr = pd.read_csv('data/hr-analytics/IBMHRAnalytics.csv')

# Rank by years at the company
yearsatcompany_emprank = ibmhr[['EmployeeNumber', 'YearsAtCompany']].sort_values(by = ['YearsAtCompany'],

                                                                                 ignore_index = True, ascending=False)
yearsatcompany= yearsatcompany_emprank[['YearsAtCompany']]
yearsatcompany_emprank = yearsatcompany_emprank[['EmployeeNumber']]
yearsatcompany_emprank.rename(columns={'EmployeeNumber': 'RankByYearsAtCompany'}, inplace=True)

# Rank by years in current role
yearscurrentrole_emprank = ibmhr[['EmployeeNumber', 'YearsInCurrentRole']].sort_values(by = ['YearsInCurrentRole'],
                                                                                 ignore_index = True, ascending=False)
yearscurrentrole = yearscurrentrole_emprank[['YearsInCurrentRole']]
yearscurrentrole_emprank = yearscurrentrole_emprank[['EmployeeNumber']]
yearscurrentrole_emprank.rename(columns={'EmployeeNumber': 'RankByYearsInCurrentRole'}, inplace=True)


# Rank by years since last promotion
yearslastpromo_emprank = ibmhr[['EmployeeNumber', 'YearsSinceLastPromotion']].sort_values(by = ['YearsSinceLastPromotion'],
                                                                                 ignore_index = True, ascending=False)
yearslastpromo= yearslastpromo_emprank[['YearsSinceLastPromotion']]
yearslastpromo_emprank = yearslastpromo_emprank[['EmployeeNumber']]
yearslastpromo_emprank.rename(columns={'EmployeeNumber': 'RankByYearsSinceLastPromotion'}, inplace=True)

# Rank by years with current manager
yearscurrman_emprank = ibmhr[['EmployeeNumber', 'YearsWithCurrManager']].sort_values(by = ['YearsWithCurrManager'],
                                                                                 ignore_index = True, ascending=False)
yearscurrman = yearscurrman_emprank[['YearsWithCurrManager']]
yearscurrman_emprank = yearscurrman_emprank[['EmployeeNumber']]
yearscurrman_emprank.rename(columns={'EmployeeNumber': 'RankByYearsWithCurrManager'}, inplace=True)

#Final ranking profile
ibm_hr_profile_df = pd.concat([yearsatcompany_emprank, yearscurrentrole_emprank, yearslastpromo_emprank, yearscurrman_emprank],  axis = 1)
ibm_hr_scores_df = pd.concat([yearsatcompany, yearscurrentrole, yearslastpromo, yearscurrman],  axis = 1)

#Uncomment to save to csv
# ibm_hr_profile_df.to_csv('data/hr-analytics/ibm_hr_profile.csv', index = False)


#2. We have to create the item group dictionary
bins = [17, 20, 30, 40, 50, 60]
labels = ['10s','20s','30s','40s','50+']
ibmhr['BinnedAge'] = pd.cut(ibmhr['Age'], bins=bins, labels=labels)
#ibmhr = ibmhr.dropna(subset = ['BinnedAge'])
ibmhr = ibmhr.select_dtypes(exclude=['object']) #drop everything but numbers for this
np_profile = ibm_hr_profile_df.to_numpy()
unique_employees = list(np.unique(np_profile))
employee_age = [ibmhr.loc[ibmhr['EmployeeNumber'] == emp]['BinnedAge'].item() for emp in unique_employees]
profile_item_group_dict = dict(zip(unique_employees, employee_age))


# Run experiment
profile_df = ibm_hr_profile_df
scores_df = ibm_hr_scores_df
fair_rep = 'EQUAL'
fusion = 'COMBmnz'
candidate_ids = list(np.unique(np_profile))
dataset_name = 'IBMHR'
fair_rep = 'PROPORTIONAL'
csv_name = 'results/ibmhr/results_ibmhr.csv'
workflow(profile_df, scores_df, fair_rep, fusion, profile_item_group_dict, candidate_ids, dataset_name, csv_name)