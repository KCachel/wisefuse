import comparedmethods as cm
import src as src
import numpy as np
import pandas as pd
import time

dataset_df = pd.read_excel('data/synthetic-study/mallows_df.xlsx')
candidates_col = 'Candidates'
sa_col = 'Sensitive'
dataset_df[candidates_col] = dataset_df[candidates_col].apply(str)
dataset_df = pd.DataFrame(dataset_df.to_numpy())
dataset_df.rename(columns={0: candidates_col}, inplace=True)
dataset_df.rename(columns={1: sa_col}, inplace=True)

times = []
ndkl = []
ndkl_prefix = []
fairness = []
wig_rbo = []
rbo = []
method = []
dataset = []
dispersion = []
csv_name = 'results/synthetic-study/syntheticFQ.csv'
trials = 5

#for disjoint in [0, .2, .4, .6, .8, 1]:
for disp in [0, .02, .04, .06, .08, .1]:
    filename = "data\synthetic-study\profile_disp_" + str(disp) + "_.csv"
    data = np.genfromtxt(filename, delimiter=',', dtype=int)
    data = data.T
    profile_df = pd.DataFrame(data)
    profile_df = profile_df.astype(str)
    print("disjointness param........", disp)
    np_profile = profile_df.to_numpy()
    candidate_ids = list(np.unique(np_profile))
    dataset_name = 'synthetic-' + str(disp)
    ranked_grps = [dataset_df.loc[dataset_df[candidates_col] == e][sa_col].item() for e in candidate_ids]
    profile_item_group_dict = dict(zip(candidate_ids, ranked_grps))


    # BORDA
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_borda, _ = cm.BORDA(profile_df, candidate_ids)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_borda, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_borda, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_borda))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_borda, profile_item_group_dict))
    method.append('BORDA')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # RAPF
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_rapf = cm.RAPF(profile_df, profile_item_group_dict, seed)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_rapf, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_rapf, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_rapf))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_rapf, profile_item_group_dict))
    method.append('RAPF')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))


    # EPIRA
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_epira = cm.EPIRA(profile_df, profile_item_group_dict, .9)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_epira, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_epira, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_epira))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_epira, profile_item_group_dict))
    method.append('EPIRA')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))

    # FAIR FUSION
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_fairfusborda = src.fair_fusion(profile_df, profile_df, profile_item_group_dict, .99, 'EQUAL', 'borda')
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_fairfusborda, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_fairfusborda, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_fairfusborda))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_fairfusborda, profile_item_group_dict))
    method.append('WISE')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
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
        cr_pre_egs, _ = cm.pre_epsilongreedy(profile_df, profile_df, candidate_ids, profile_item_group_dict, .6, seed, 'borda')
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_pre_egs, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_pre_egs, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_pre_egs))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_pre_egs, profile_item_group_dict))
    method.append('PRE-EG')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
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
        cr_post_egs, _ = cm.post_epsilongreedy(profile_df, profile_df, candidate_ids, profile_item_group_dict, .6, seed,
                                           'borda')
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_post_egs, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_post_egs, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_post_egs))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_post_egs, profile_item_group_dict))
    method.append('POST-EG')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
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
        cr_pre_fq, _ = cm.pre_fairqueues(pd.DataFrame(np_profile), pd.DataFrame(data), candidate_ids, profile_item_group_dict,  'borda', .05)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_pre_fq, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_pre_fq, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_pre_fq))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_pre_fq, profile_item_group_dict))
    method.append('PRE-FQ')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))

    # POST-fair queues
    TIMES_ = []
    NDKL_ = []
    WIG_RBO_ = []
    RBO_ = []
    NDKLPRE_ = []
    for i in range(0, trials):
        seed = i  # for repro
        start_time = time.time()
        cr_post_fq, _ = cm.post_fairqueues(pd.DataFrame(np_profile), pd.DataFrame(data), candidate_ids, profile_item_group_dict,  'borda', .05)
        end_time = time.time()
        TIMES_.append(end_time - start_time)
        NDKL_.append(src.NDKL_onebyone(cr_post_fq, profile_item_group_dict, 'EQUAL'))
        NDKLPRE_.append(src.NDKL_groupchunks(cr_post_fq, profile_item_group_dict, 'EQUAL'))
        RBO_.append(src.avg_rbo(profile_df, cr_post_fq))
        WIG_RBO_.append(src.avg_wg_rbo(cr_borda, cr_post_fq, profile_item_group_dict))
    method.append('POST-FQ')
    dataset.append(dataset_name)
    dispersion.append(disp)
    times.append(np.mean(TIMES_))
    ndkl.append(np.mean(NDKL_))
    ndkl_prefix.append(np.mean(NDKLPRE_))
    fairness.append('EQUAL')
    wig_rbo.append(np.mean(WIG_RBO_))
    rbo.append(np.mean(RBO_))

    # Save results
    dic = {'method': method,
           'dataset': dataset,
           'dispersion': dispersion,
           'times': times,
           'NDKL': ndkl,
           'rbo': rbo,
           'wig_rbo': wig_rbo,
           'fairness': fairness
           }

    results = pd.DataFrame(dic)
    results.to_csv(csv_name, index=False)

