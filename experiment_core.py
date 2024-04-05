import numpy as np
import pandas as pd
import comparedmethods as cm
import src as src
import time

def workflow(profile_df, scores_df, fair_rep, fusion, profile_item_group_dict, candidate_ids, dataset_name, csv_name):
    times = []
    ndkl = []
    fairness = []
    wig_rbo = []
    rbo = []
    method = []
    dataset = []
    tuning = []
    trials = 5

    if dataset_name in ['synthetic-0.1']:
        rng = range(0, 5)
    else:
        rng = range(4,5)
    for t in rng:
        print("working on tuning parameter ", t)
        borda_tuning = [.1, .2, .3, .4, .5]
        combs_tuning = [.1, .2, .3, .4, .5]
        rapf_tuning = [.1, .2, .3, .4, .5]
        eg_tuning = [.2, .3, .4, .5, .6]
        if dataset_name in ['synthetic-0.1']:
            fairfuse_tuning = [0, .25, .5, .75, .99]
        else:
            fairfuse_tuning = [0, .25, .5, .75, .9]
        epira_tuning = [.5, .6, .7, .8, .9]
        fq_tuning = [.25, .2, .15, .1, .05]

        print("starting original...")
        if fusion == 'borda':
            # BORDA
            TIMES_ = []
            NDKL_ = []
            RBO_ = []
            WIG_RBO_ = []
            for i in range(0, trials):
                start_time = time.time()
                cr_original, _ = cm.BORDA(profile_df, candidate_ids)
                end_time = time.time()
                TIMES_.append(end_time - start_time)
                NDKL_.append(src.NDKL_onebyone(cr_original, profile_item_group_dict, fair_rep))
                RBO_.append(src.avg_rbo(profile_df, cr_original))
                WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_original, profile_item_group_dict))
            method.append('BORDA')
            dataset.append(dataset_name)
            times.append(np.mean(TIMES_))
            ndkl.append(np.mean(NDKL_))
            fairness.append(fair_rep)
            wig_rbo.append(np.mean(WIG_RBO_))
            rbo.append(np.mean(RBO_))
            tuning.append(borda_tuning[t])

        if fusion == 'COMBmnz':
            # COMBS
            TIMES_ = []
            NDKL_ = []
            RBO_ = []
            WIG_RBO_ = []
            for i in range(0, trials):
                start_time = time.time()
                cr_original, _ = cm.COMBmnz(profile_df, scores_df, candidate_ids)
                end_time = time.time()
                TIMES_.append(end_time - start_time)
                NDKL_.append(src.NDKL_onebyone(cr_original, profile_item_group_dict, fair_rep))
                RBO_.append(src.avg_rbo(profile_df, cr_original))
                WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_original, profile_item_group_dict))
            method.append('COMBmnz')
            dataset.append(dataset_name)
            times.append(np.mean(TIMES_))
            ndkl.append(np.mean(NDKL_))
            fairness.append(fair_rep)
            wig_rbo.append(np.mean(WIG_RBO_))
            rbo.append(np.mean(RBO_))
            tuning.append(combs_tuning[t])


        print("starting RAPF...")
        # RAPF
        TIMES_ = []
        NDKL_ = []
        RBO_ = []
        WIG_RBO_ = []
        for i in range(0, trials):
            seed = i  # for repro
            start_time = time.time()
            cr_rapf = cm.RAPF(profile_df, profile_item_group_dict, seed)
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_rapf, profile_item_group_dict, fair_rep))
            RBO_.append(src.avg_rbo(profile_df, cr_rapf))
            WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_rapf, profile_item_group_dict))
        method.append('RAPF')
        dataset.append(dataset_name)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append(fair_rep)
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))
        tuning.append(rapf_tuning[t])

        if dataset_name not in ['World Happiness', 'Econ-Freedom']:  # can't use EPIRA on non-overlapping
            print("starting EPIRA...")
            # EPIRA
            TIMES_ = []
            NDKL_ = []
            RBO_ = []
            WIG_RBO_ = []
            for i in range(0, trials):
                start_time = time.time()
                cr_epira = cm.EPIRA(profile_df, profile_item_group_dict, epira_tuning[t])
                end_time = time.time()
                TIMES_.append(end_time - start_time)
                NDKL_.append(src.NDKL_onebyone(cr_epira, profile_item_group_dict, fair_rep))
                RBO_.append(src.avg_rbo(profile_df, cr_epira))
                WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_epira, profile_item_group_dict))
            method.append('EPIRA')
            dataset.append(dataset_name)
            times.append(np.mean(TIMES_))
            ndkl.append(np.mean(NDKL_))
            fairness.append(fair_rep)
            wig_rbo.append(np.mean(WIG_RBO_))
            rbo.append(np.mean(RBO_))
            tuning.append(epira_tuning[t])

        print("starting WISE...")
        # FAIR FUSION
        TIMES_ = []
        NDKL_ = []
        RBO_ = []
        WIG_RBO_ = []
        for i in range(0, trials):
            start_time = time.time()
            cr_fairfus = src.fair_fusion(profile_df, scores_df, profile_item_group_dict, fairfuse_tuning[t], fair_rep, fusion)
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_fairfus, profile_item_group_dict, fair_rep))
            RBO_.append(src.avg_rbo(profile_df, cr_fairfus))
            WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_fairfus, profile_item_group_dict))
        method.append('WISE')
        dataset.append(dataset_name)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append(fair_rep)
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))
        tuning.append(fairfuse_tuning[t])

        print("starting pre epsilon-greedy...")
        # PRE-epsilon-greedy
        TIMES_ = []
        NDKL_ = []
        RBO_ = []
        WIG_RBO_ = []
        for i in range(0, trials):
            seed = i  # for repro
            start_time = time.time()
            cr_pre_egs, _ = cm.pre_epsilongreedy(profile_df, scores_df, candidate_ids, profile_item_group_dict, eg_tuning[t],
                                                 seed, fusion)
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_pre_egs, profile_item_group_dict, fair_rep))
            RBO_.append(src.avg_rbo(profile_df, cr_pre_egs))
            WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_pre_egs, profile_item_group_dict))
        method.append('PRE-EG')
        dataset.append(dataset_name)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append(fair_rep)
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))
        tuning.append(eg_tuning[t])

        print("starting post epsilon-greedy...")
        # POST-epsilon-greedy
        TIMES_ = []
        NDKL_ = []
        RBO_ = []
        WIG_RBO_ = []
        for i in range(0, trials):
            seed = i  # for repro
            start_time = time.time()
            cr_post_egs, _ = cm.post_epsilongreedy(profile_df, scores_df, candidate_ids, profile_item_group_dict, eg_tuning[t],
                                                   seed,
                                                   fusion)
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_post_egs, profile_item_group_dict, fair_rep))
            RBO_.append(src.avg_rbo(profile_df, cr_post_egs))
            WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_post_egs, profile_item_group_dict))
        method.append('POST-EG')
        dataset.append(dataset_name)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append(fair_rep)
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))
        tuning.append(eg_tuning[t])

        print("starting pre fq...")
        # PRE-fairq
        TIMES_ = []
        NDKL_ = []
        RBO_ = []
        WIG_RBO_ = []
        for i in range(0, trials):
            start_time = time.time()
            cr_pre_fq, _ = cm.pre_fairqueues(profile_df, scores_df, candidate_ids,
                                             profile_item_group_dict, fusion, fq_tuning[t])
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_pre_fq, profile_item_group_dict, fair_rep))
            RBO_.append(src.avg_rbo(profile_df, cr_pre_fq))
            WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_pre_fq, profile_item_group_dict))
        method.append('PRE-FQ')
        dataset.append(dataset_name)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append(fair_rep)
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))
        tuning.append(fq_tuning[t])

        print("starting post fq...")
        # POST-fair queues
        TIMES_ = []
        NDKL_ = []
        RBO_ = []
        WIG_RBO_ = []
        for i in range(0, trials):
            start_time = time.time()
            cr_post_fq, _ = cm.post_fairqueues(profile_df, scores_df, candidate_ids,
                                               profile_item_group_dict, fusion, fq_tuning[t])
            end_time = time.time()
            TIMES_.append(end_time - start_time)
            NDKL_.append(src.NDKL_onebyone(cr_post_fq, profile_item_group_dict, fair_rep))
            RBO_.append(src.avg_rbo(profile_df, cr_post_fq))
            WIG_RBO_.append(src.avg_wg_rbo(cr_original, cr_post_fq, profile_item_group_dict))
        method.append('POST-FQ')
        dataset.append(dataset_name)
        times.append(np.mean(TIMES_))
        ndkl.append(np.mean(NDKL_))
        fairness.append(fair_rep)
        wig_rbo.append(np.mean(WIG_RBO_))
        rbo.append(np.mean(RBO_))
        tuning.append(fq_tuning[t])

        # Save results
        dic = {'method': method,
               'dataset': dataset,
               'tuning': tuning,
               'times': times,
               'NDKL': ndkl,
               'rbo': rbo,
               'wig_rbo': wig_rbo,
               'fairness': fairness
               }

        results = pd.DataFrame(dic)
        results.to_csv(csv_name, index=False)