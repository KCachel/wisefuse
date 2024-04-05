import comparedmethods as cm
import src as src
import numpy as np
import pandas as pd
import time
import copy

german_credit = pd.read_csv('data/disjoint-study/german_credit.csv')
german_credit = german_credit.assign(candnumber = list(range(0, len(german_credit))))
mapping = { 1 :'Male', 2:'Male',3:'Male',4:'Female' }
german_credit = german_credit.assign(Sex = german_credit['Sex & Marital Status'].map(mapping))
german_credit = german_credit[['candnumber', 'Sex', 'Credit Amount']]


times = []
ndkl = []
fairness = []
wig_rbo = []
rbo = []
method = []
dataset = []
disjointness = []
csv_name = 'results/disjoint-study/disjointFQ.csv'
trials = 5
base = 10
fixs = [1, .8, .6, .4, .2, 0]
disjoints = [0, .2, .4, .6, .8, 1]
#disjoint represents what proportion of the base rankings are disjoint.
for x in range(0, len(fixs)):
    fix = fixs[x]
    disjoint = disjoints[x]
    credit = pd.DataFrame(columns = german_credit.columns, data = copy.deepcopy(german_credit.values))
    fixed = credit.sample(n = int(fix*100), replace=False, random_state=1) #these will be shared across base rankings
    credit = credit.drop(fixed.index)
    unfixedpool = credit.sample(n = int(disjoint*100*base), replace=False, random_state=1)
    profile_dict = {}
    scores_dict = {}
    for i in range(0,base):
        unfixed = unfixedpool[0: int(disjoint*100)]
        unfixedpool = unfixedpool.drop(index=unfixedpool.index[:int(disjoint *100)], axis=0) #drop candidates
        baserank = pd.concat([fixed, unfixed], ignore_index=True)
        rank = baserank[['candnumber', 'Credit Amount']].sort_values(by=['Credit Amount'],
                                                                                         ignore_index=True,
                                                                                         ascending=False)
        profile_dict[i] = rank['candnumber'].tolist()
        scores_dict[i] = rank['Credit Amount'].tolist()

    profile_df = pd.DataFrame(profile_dict)
    scores_df = pd.DataFrame(scores_dict)

    print("disjointness param........", disjoint)
    np_profile = profile_df.to_numpy()
    candidate_ids = list(np.unique(np_profile))
    print('Disjoint: ', disjoint, 'needs ', (disjoint*100)*base+((1-disjoint)*100), 'candidates. Has ', len(candidate_ids))
    cand_sex = [german_credit.loc[german_credit['candnumber'] == e]['Sex'].item() for e in candidate_ids]
    profile_item_group_dict = dict(zip(candidate_ids, cand_sex))
    dataset_name = 'disjoint-' + str(disjoint)



    # BORDA
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_borda, _ = cm.BORDA(profile_df, candidate_ids)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_borda, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_borda))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_borda, profile_item_group_dict))
    method.append('BORDA')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))

    #
    # RAPF
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_rapf = cm.RAPF(profile_df, profile_item_group_dict, seed)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_rapf, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_rapf))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_rapf, profile_item_group_dict))
    method.append('RAPF')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))

    #
    # EPIRA
    if x == 0: #Can only run EPIRA on full rankings
        TIMES_ = []
        NDKL_ = []
        WIG_RBO_ = []
        RBO_ = []
        for i in range(0, trials):
            seed = i  # for repro
            start_time = time.time()
            cr_epira = cm.EPIRA(profile_df, profile_item_group_dict, .9)
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_epira, profile_item_group_dict, 'EQUAL'))
            RBO_.append(src.avg_rbo(profile_df, cr_epira))
            WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_epira, profile_item_group_dict))
        method.append('EPIRA')
        dataset.append(dataset_name)
        disjointness.append(disjoint)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append('EQUAL')
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))


    # FAIR FUSION
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_fairfusborda = src.fair_fusion(profile_df, scores_df, profile_item_group_dict, .9, 'EQUAL', 'borda')
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_fairfusborda, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_fairfusborda))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_fairfusborda, profile_item_group_dict))
    method.append('WISE')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # PRE-epsilon-greedy
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_pre_egs, _ = cm.pre_epsilongreedy(profile_df, scores_df, candidate_ids, profile_item_group_dict, .6, seed, 'borda')
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_pre_egs, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_pre_egs, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_pre_egs))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_pre_egs, profile_item_group_dict))
    method.append('PRE-EG')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # POST-epsilon-greedy
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_post_egs, _ = cm.post_epsilongreedy(profile_df, scores_df, candidate_ids, profile_item_group_dict, .6, seed,
                                           'borda')
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_post_egs, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_post_egs, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_post_egs))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_post_egs, profile_item_group_dict))
    method.append('POST-EG')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # PRE-fairq
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_pre_fq, _ = cm.pre_fairqueues(profile_df, scores_df, candidate_ids, profile_item_group_dict,  'borda', .05)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_pre_fq, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_pre_fq, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_pre_fq))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_pre_fq, profile_item_group_dict))
    method.append('PRE-FQ')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # POST-fair queues
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_post_fq, _ = cm.post_fairqueues(profile_df, scores_df, candidate_ids, profile_item_group_dict,  'borda', .05)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_post_fq, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_post_fq, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_post_fq))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_post_fq, profile_item_group_dict))
    method.append('POST-FQ')
    dataset.append(dataset_name)
    disjointness.append(disjoint)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # Save results
    dic = {'method': method,
           'dataset': dataset,
           'disjointness': disjointness,
           'times': times,
           'NDKL': ndkl,
           'rbo': rbo,
           'wig_rbo': wig_rbo,
           'fairness': fairness
           }

    results = pd.DataFrame(dic)
    results.to_csv(csv_name, index=False)

