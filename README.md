# Within A Group Similiarity Fusion: WISE FUSION


Code and data for "WISE Fuse: Group Fairness Aware Rank Fusion". To reproduce
the experiments run `adult.py` (adult data), `econf.py` (economic freedom data), `worldhappiness.py` (world happiness data),
`ibmhr.py` (ibmhr data), `disjoint_study.py` (overlap credit data), `synthetic_tuning.py` (dataset for comparing tuning parameter predictability)  and `synthetic_mallows.py` (Mallows base rankings - note the code to
generate the Mallows profiles themselves are in `data\synthetic-study\generate_mallows.R`). Next to produce the plots
used in the paper run the script `plotting.R` in the `results/` folder.

All WISE source code is in the `src/` folder, and all compared methods are in the `comparedmethods/` folder (using code 
from [EPIRA](https://github.com/KCachel/Fairer-Together-Mitigating-Disparate-Exposure-in-Kemeny-Aggregation),
[RAPF](https://github.com/MouinulIslamNJIT/Rank-Aggregation_Proportionate_Fairness),
and [Fair Queues](https://github.com/ejohnson0430/fair_online_ranking/blob/master/algorithms/fair_queues.py)).

Each dataset is provided in the `data/` folder and are derived from publicly released data. However, our repo cannot directly contain the Economic Freedom data. The Fraser institute makes
the data publicly available, but users wishing to use it must download it themselves from https://www.fraserinstitute.org/economic-freedom/dataset.
Once the `efotw-2023-master-index-data-for-researchers-iso.xlsx` file is downloaded please place it into the `econf/` folder.