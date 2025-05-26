import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SIZES = ['14m', '70m', '160m', '2.8b', '12b']
CKPTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
         1000, 2000, 3000, 4000, 8000, 13000, 23000 ,32000, 33000, 43000,
         53000, 63000, 64000, 73000, 83000, 93000, 103000, 113000, 123000, 133000,
         143000]
SCALING_SIZES = ['410m', '1.4b', '6.9b']
DATASETS = ['1', '2', '3', '4']
DATASET_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


HIDDEN_DIMS = {
    '14m': 128, 
    '70m': 512, 
    '160m': 768, 
    '410m': 1024,
    '1.4b': 2048,
    '6.9b': 4096,
    '12b': 5120,
}

MODES = ['sane', 'shuffled']


def plot_id_model_size(results_df):
    # Make figure
    Ds = [HIDDEN_DIMS[model] for model in HIDDEN_DIMS]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    for i, mode in enumerate(MODES):
        for ds in DATASETS:
            last_ckpt_df = results_df[(results_df['step']==143000) & (results_df['mode']==mode)]
            last_ckpt_df = last_ckpt_df[last_ckpt_df['words_coupled']==ds]
            
            # Mean
            agg_std = lambda x: np.sqrt(np.sum([x_**2 for x_ in x]))
            last_ckpt_mean = last_ckpt_df[['model', 'twonn_mean', 'pca_mean']]
            last_ckpt_std = last_ckpt_df[['model', 'twonn_std', 'pca_std']]
            means_df = last_ckpt_mean.groupby('model').agg(['mean'])
            std_df = last_ckpt_std.groupby('model').agg([agg_std])
            means_df = means_df.merge(std_df, on='model')
            means_df = means_df.reset_index()
            # Plot one line per datas
            for j, method in enumerate(['twonn', 'pca']):
                ys = np.array([means_df[means_df['model']==model][f'{method}_mean'] for model in HIDDEN_DIMS])[:,0,0]
                ystd = np.array([means_df[means_df['model']==model][f'{method}_std'] for model in HIDDEN_DIMS])[:,0,0]
                axs[j].plot(
                    Ds, 
                    ys, 
                    label=f'{ds}',
                    linestyle='--' if mode == 'shuffled' else None,
                    marker='o' if mode == 'sane' else '^',
                    alpha=0.8,
                    color=DATASET_COLORS[int(ds)-1]
                )
                axs[j].fill_between(Ds, ys - ystd, ys + ystd, alpha=0.4, color=DATASET_COLORS[int(ds)-1])
                if method == 'twonn':
                    axs[j].set_ylabel(r'TwoNN $I_d$', fontsize=13)
                else:
                    axs[j].set_ylabel(r'PCA $d$', fontsize=13)

                axs[j].set_xlabel(r'hidden dimension $D$', fontsize=14)
                axs[j].grid(True)

    # Custom legend handles for both subplots
    custom_lines = [Line2D([0], [0], color=color, marker='o', lw=2) for color in DATASET_COLORS] + \
        [Line2D([0], [0], color='gray', lw=2), Line2D([0], [0], color='gray', linestyle='--', lw=2)]
    
    fig.legend(custom_lines, 
            [1, 2, 3, 4, 'coherent', 'shuffled'], 
            loc='upper center', ncol=3, title='# words coupled', bbox_to_anchor=(0.705,0.95))
    fig.tight_layout()
    
    fig.savefig('id_model_size.png')



def plot_layerwise_id(results_df):
    sizes = list(HIDDEN_DIMS.keys())

    fig, axs = plt.subplots(2, len(sizes), figsize=(4 * len(sizes), 8))

    for i, method in enumerate(['twonn', 'pca']):
        for j, model in enumerate(sizes):
            # plot over layers
            ax = axs[i][j]

            # first get the x and y and std for sane
            model_df = results_df[(results_df['model']==model) & (results_df['step']==143000)]

            for mode in ['sane', 'shuffled']:
                setting_df = model_df[model_df['mode'] == mode]

                for ds in DATASETS:
                    df = setting_df[setting_df['words_coupled']==ds]
                    ax.plot(
                        df['layer'], 
                        df[f'{method}_mean'], 
                        color=DATASET_COLORS[int(ds)-1], 
                        marker='o' if mode == 'sane' else '^',
                        linestyle='--' if mode == 'shuffled' else None,
                        markersize=6
                    )
                    ax.fill_between(df['layer'], 
                        df[f'{method}_mean'] - df[f'{method}_std'],
                        df[f'{method}_mean'] + df[f'{method}_std'],
                        alpha=0.3,
                        color=DATASET_COLORS[int(ds)-1]
                                )
            if i == 0:
                ax.set_title(model, fontsize=30)
            ax.grid(True)
            ax.tick_params(axis='x', labelsize=23)
            ax.tick_params(axis='y', labelsize=23)
            if i == 1: 
                ax.set_xlabel('layer', fontsize=30)

            if j == 0:
                if i == 0:
                    ax.set_ylabel(r'TwoNN $I_d$', fontsize=30)
                else:
                    ax.set_ylabel(r'PCA $d$ (0.99)', fontsize=30)

    # Custom legend handles for both subplots
    custom_lines = [Line2D([0], [0], color=color, marker='o', lw=2) for color in DATASET_COLORS] + \
        [Line2D([0], [0], color='gray', lw=2), Line2D([0], [0], color='gray', linestyle='--', lw=2)]

    fig.legend(custom_lines, 
            [1, 2, 3, 4, 'sane', 'shuffled'], 
            ncol=1, title='# words coupled', bbox_to_anchor=(1.11, 0.96), fontsize=20, title_fontsize=23)
    fig.tight_layout()
    fig.savefig('nonlinear_linear_id_app.png', bbox_inches='tight')


results_df = pd.read_csv('id_results_all.csv')

plot_id_model_size(results_df)
plot_layerwise_id(results_df)