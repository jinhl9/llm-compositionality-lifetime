## After saving the resultss, we collect the results to make into one csv file.
import json
import os
import argparse
import numpy as np
from utils import *
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Collect all results to csv.")
    parser.add_argument("--results_path", type=str, help="Results path")

    return parser.parse_args()

args = parse_args()


fileformat = os.path.join(args.base_dir, '/EleutherAI_pythia-{}-deduped/ids_dataset_{}_step_{}.json') # size dataset step
file14m = os.path.join(args.base_dir, '/EleutherAI_pythia-14m/ids_dataset_{}_step_143000.json') # size

def load_data(size, dataset, step=None):
    filepath = fileformat.format(size, dataset, step) if size != '14m' else file14m.format(dataset, 143000)
    with open(filepath , 'r') as f:
        data = json.load(f)

    return data

SIZES = ['14m', '70m', '160m', '2.8b', '12b']
CKPTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
         1000, 2000, 3000, 4000, 8000, 13000, 23000 ,32000, 33000, 43000,
         53000, 63000, 64000, 73000, 83000, 93000, 103000, 113000, 123000, 133000,
         143000]
SCALING_SIZES = ['410m', '1.4b', '6.9b']
DATASETS = [1, 2, 3, 4, 'pile']
DATASET_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


HIDDEN_DIMS = {
    '14m': 128, 
    '70m': 512, 
    '160m': 768, 
    '410m': 1024,
    '1.4b': 2048,
    '2.8b': 2560, 
    '6.9b': 4096,
    '12b': 5120,
}

MODES = ['sane', 'shuffled']


RESULTS = []
for model in HIDDEN_DIMS:
    for ds in DATASETS:
        if model == '14m' and ds == 'pile': continue
        if model in SCALING_SIZES or ds == 'pile': 
            steps = CKPTS
        else:
            steps = [143000]
        for step in steps:
            try:
                results = load_data(model, ds, step)
            except FileNotFoundError:
                continue
            if ds != 'pile':
                for mode in MODES:
                    results_ = pd.DataFrame(results[mode])
                    results_['mode'] = mode
                    results_['words_coupled'] = ds
                    results_['step'] = step
                    results_['D'] = HIDDEN_DIMS[model]
                    results_['model'] = model
                    results_['layer'] = list(range(len(results_)))
                    RESULTS.append(results_)
            elif ds == 'pile':
                results_ = pd.DataFrame(results)
                results_['mode'] = 'pile'
                results_['words_coupled'] = 0
                results_['step'] = step
                results_['D'] = HIDDEN_DIMS[model]
                results_['model'] = model
                results_['layer'] = list(range(len(results_)))
                RESULTS.append(results_)
                
results_df = pd.concat(RESULTS)


results_df.to_csv("id_results_all.csv")
