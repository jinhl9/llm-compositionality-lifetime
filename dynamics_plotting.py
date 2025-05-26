import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

SIZES = ['14m', '70m', '160m', '2.8b', '12b']
CKPTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 8000,
    13000, 23000, 32000, 33000, 43000, 53000, 63000, 64000, 73000, 83000,
    93000, 103000, 113000, 123000, 133000, 143000
]
SCALING_SIZES = ['410m', '1.4b', '6.9b']
DATASETS = ['1', '2', '3', '4']
DATASET_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_layers_over_checkpoints(results_df,
                                 model='6.9b',
                                 method='twonn',
                                 ckpts=[0, 2000, 143000],
                                 save_path=None):
    fig, axs = plt.subplots(2, len(ckpts), figsize=(4 * len(ckpts), 6))
    for i, mode in enumerate(['sane', 'shuffled']):
        for j, ckpt in enumerate(ckpts):
            ax = axs[i][j]
            model_df = results_df[(results_df['model'] == model)
                                  & (results_df['step'] == ckpt) &
                                  (results_df['length'] == 17)]
            setting_df = model_df[model_df['mode'] == mode]
            for ds in DATASETS:
                df = setting_df[setting_df['words_coupled'] == ds]
                if method == 'twonn':
                    ax.set_ylim(0, 60)
                if method == 'pca':
                    ax.set_ylim(0, 3000)
                ax.plot(df['layer'],
                        df[f'{method}_mean'],
                        color=DATASET_COLORS[int(ds) - 1],
                        marker='o' if mode == 'sane' else '^',
                        linestyle='--' if mode == 'shuffled' else None,
                        markersize=6)
                ax.fill_between(df['layer'],
                                df[f'{method}_mean'] - df[f'{method}_std'],
                                df[f'{method}_mean'] + df[f'{method}_std'],
                                alpha=0.3,
                                color=DATASET_COLORS[int(ds) - 1])
            if i == 0:
                ax.set_title(f'checkpoint: {ckpt}', fontsize=20)
            ax.grid(True)
            ax.tick_params(axis='x', labelsize=23)
            ax.tick_params(axis='y', labelsize=23)
            if i == 1:
                ax.set_xlabel('layer', fontsize=30)
            if method == 'twonn':
                if j == 0:
                    if i == 0:
                        ax.set_ylabel(r'Coherent, $I_d$', fontsize=30)
                    else:
                        ax.set_ylabel(r'Shuffled, $I_d$', fontsize=30)
            if method == 'pca':
                if j == 0:
                    if i == 0:
                        ax.set_ylabel(r'Coherent, $d$', fontsize=30)
                    else:
                        ax.set_ylabel(r'Shuffled, $d$', fontsize=30)
    custom_lines = [Line2D([0], [0], color=color, marker='o', lw=2) for color in DATASET_COLORS] + \
        [Line2D([0], [0], color='gray', lw=2, marker = 'o'), Line2D([0], [0], color='gray', linestyle='--', lw=2, marker = '^')]
    fig.legend(custom_lines, [1, 2, 3, 4, 'coherent', 'shuffled'],
               ncol=1,
               title='# words coupled',
               bbox_to_anchor=(1.3, 0.5),
               fontsize=20,
               title_fontsize=23)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_id_diff_over_ckpts(results_df,
                            models=['6.9b', '1.4b', '410m'],
                            method='twonn',
                            save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    diff_list = np.empty((2, len(models), len(CKPTS)))
    color_list = ['tab:pink', 'tab:olive', 'tab:cyan']
    for k, mode in enumerate(['sane', 'shuffled']):
        for i, model in enumerate(models):
            last_layer = max(results_df['layer'][results_df['model'] == model])
            cond = (results_df['model']
                    == model) & (results_df['layer']
                                 == last_layer) & (results_df['mode'] == mode)
            for j, ckpt in enumerate(CKPTS):
                id_k1 = results_df[cond
                                   & (results_df['words_coupled'] == str(1)) &
                                   (results_df['step']
                                    == ckpt)][f'{method}_mean']
                id_k4 = results_df[cond
                                   & (results_df['words_coupled'] == str(4)) &
                                   (results_df['step']
                                    == ckpt)][f'{method}_mean']
                if len(id_k1) != 1 or len(id_k4) != 1:
                    diff = np.nan
                else:
                    diff = id_k1.item() - id_k4.item()
                diff_list[k, i, j] = diff
    for m in range(2):
        for s in range(len(models)):
            ax.plot(
                np.array(CKPTS)[~np.isnan(diff_list[m, s])],
                diff_list[m, s][~np.isnan(diff_list[m, s])],
                color=color_list[s],
                marker='o' if m == 0 else '^',
                linestyle='--' if m == 1 else None,
                markersize=6,
            )
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('checkpoints', fontsize=25)
    if method == 'twonn':
        ax.set_ylabel(r'$\Delta I_d$', fontsize=25)
        ax.set_title(r'$I_{d,k=1} - I_{d,k=4}$', fontsize=30)
    if method == 'pca':
        ax.set_ylabel(r'$\Delta d$', fontsize=25)
        ax.set_title(r'$d_{k=1} - d_{k=4}$', fontsize=30)
    ax.grid(True)
    custom_lines = [Line2D([0], [0], color=color_list[i], marker='o', lw=2) for i in range(len(models))] + \
        [Line2D([0], [0], color='gray', lw=2, marker = 'o'), Line2D([0], [0], color='gray', linestyle='--', lw=2, marker = '^')]
    fig.legend(custom_lines,
               models + ['coherent', 'shuffled'],
               ncol=1,
               title='model size',
               bbox_to_anchor=(1.4, 0.96),
               fontsize=20,
               title_fontsize=23)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_length_diff(results_df,
                     models=['6.9b', '1.4b', '410m'],
                     method='twonn',
                     save_path=None):
    LENGTHS = [3, 6, 9, 11, 17]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    diff_list = np.empty((len(models), len(LENGTHS)))
    for i, model in enumerate(models):
        for j, l in enumerate(LENGTHS):
            last_layer = max(results_df['layer'][results_df['model'] == model])
            cond = (results_df['model'] == model) & (
                results_df['layer'] == last_layer) & (
                    results_df['length'] == l) & (results_df['step']
                                                  == results_df['step'].max())
            id_sane = results_df[cond & (results_df['mode'] == 'sane') &
                                 (results_df['words_coupled']
                                  == '1')][f'{method}_mean']
            id_shuffled = results_df[cond & (results_df['mode'] == 'shuffled')
                                     & (results_df['words_coupled']
                                        == '1')][f'{method}_mean']
            if len(id_sane) != 1 or len(id_shuffled) != 1:
                diff = np.nan
            else:
                diff = id_sane.item() - id_shuffled.item()
            diff_list[i, j] = diff
    for s in range(len(models)):
        ax.plot(
            [5, 8, 11, 15, 17],
            diff_list[s],
            color=f'C{s}',
            marker='o',
            linestyle=None,
            markersize=6,
        )
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('sequence length', fontsize=25)
    if method == 'twonn':
        ax.set_ylabel(r'$\Delta I_d$', fontsize=25)
        ax.set_title(r'$I_{coherent} - I_{shuffled}$', fontsize=30)
    if method == 'pca':
        ax.set_ylabel(r'$\Delta d$', fontsize=25)
        ax.set_title(r'$d_{coherent} - d_{shuffled}$', fontsize=30)
    ax.grid(True)
    custom_lines = [
        Line2D([0], [0], color=f'C{i}', marker='o', lw=2)
        for i in range(len(models))
    ]
    fig.legend(custom_lines,
               models,
               ncol=1,
               title='model size',
               bbox_to_anchor=(1.4, 0.96),
               fontsize=20,
               title_fontsize=23)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_pca_length_diff(results_df,
                         models=['6.9b', '1.4b', '410m'],
                         save_path=None):
    LENGTHS = [6, 9, 11, 17]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    diff_list = np.empty((2, len(models), len(LENGTHS)))
    method = 'pca'
    for i, model in enumerate(models):
        for j, l in enumerate(LENGTHS):
            for m, mode in enumerate(['sane', 'shuffled']):
                last_layer = max(
                    results_df['layer'][results_df['model'] == model])
                cond = (results_df['model']
                        == model) & (results_df['layer'] == last_layer) & (
                            results_df['length'] == l) & (
                                results_df['step'] == results_df['step'].max())
                id_k1 = results_df[cond & (results_df['mode'] == mode) &
                                   (results_df['words_coupled']
                                    == '1')][f'{method}_mean']
                id_k2 = results_df[cond & (results_df['mode'] == mode) &
                                   (results_df['words_coupled']
                                    == '2')][f'{method}_mean']
                if len(id_k1) != 1 or len(id_k2) != 1:
                    diff = np.nan
                else:
                    diff = id_k1.item() - id_k2.item()
                diff_list[m, i, j] = diff
    for m in range(2):
        for s in range(len(models)):
            ax.plot(
                LENGTHS,
                diff_list[m][s],
                color=f'C{s}',
                marker='o' if m == 0 else '^',
                linestyle='--' if m == 1 else None,
                markersize=6,
            )
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('sequence length', fontsize=25)
    ax.set_ylabel(r'$\Delta d$', fontsize=25)
    ax.set_title(r'$d_{k=1} - d_{k=2}$', fontsize=30)
    ax.grid(True)
    custom_lines = [Line2D([0], [0], color=f'C{i}', marker='o', lw=2) for i in range(len(models))] + \
        [Line2D([0], [0], color='gray', lw=2, marker = 'o'), Line2D([0], [0], color='gray', linestyle='--', lw=2, marker = '^')]
    fig.legend(custom_lines,
               models + ['coherent', 'shuffled'],
               ncol=1,
               title='model size',
               bbox_to_anchor=(1.4, 0.96),
               fontsize=20,
               title_fontsize=23)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_task_perf_and_id(results_df,
                          taskperf_result,
                          model='6.9b',
                          method='twonn',
                          save_path=None):
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    max_layers = max(
        results_df[results_df['model'] == model]['layer'].unique())
    layers = np.arange(max_layers)

    # --- Task performance plot ---
    ax = axs[0]
    TASKS = [
        t for t in taskperf_result['task'].unique()
        if "hendrycks" not in t and "crows" not in t and "wsc" not in t
    ]

    for i, t in enumerate(TASKS):
        task_cond = (taskperf_result['task']
                     == t) & (taskperf_result['model_size']
                              == model) & (taskperf_result['shot'] == 0)
        steps = taskperf_result[task_cond]['step']
        accs = taskperf_result[task_cond]['acc']
        stds = taskperf_result[task_cond]['acc_stderr']
        sort_steps_inds = np.argsort(steps.values)
        sorted_steps = steps.iloc[sort_steps_inds]
        sorted_accs = accs.iloc[sort_steps_inds]
        sorted_stds = stds.iloc[sort_steps_inds]
        ax.plot(sorted_steps, sorted_accs, marker='o', label=t, c=f'C{i}')

        ax.fill_between(sorted_steps,
                        sorted_accs - sorted_stds,
                        sorted_accs + sorted_stds,
                        alpha=0.3,
                        color=f'C{i}')
    ax.set_xscale('log')
    ax.set_ylabel('task performance', fontsize=15)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(title='task',
              bbox_to_anchor=[1, 0.8],
              fontsize=13,
              title_fontsize=13)

    # --- ID plot over checkpoints and layers ---
    ax = axs[1]
    colormap = plt.cm.viridis
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(layers) - 1)
    steps = np.sort(taskperf_result['step'].unique())
    ax.grid(True)
    mean_list = np.empty((len(layers), len(steps)))
    std_list = np.empty((len(layers), len(steps)))
    for i, layer in enumerate(layers):
        for j, s in enumerate(steps):
            id_cond = (results_df['words_coupled'] == '1') & (results_df['mode'] == "sane") & \
                      (results_df['layer'] == layer) & (results_df['model'] == model) & (results_df['step'] == s)
            mean = results_df[id_cond][f'{method}_mean']
            std = results_df[id_cond][f'{method}_std']
            if len(mean) == 1:
                mean_list[i, j] = mean.item()
                std_list[i, j] = std.item()
            else:
                mean_list[i, j] = np.nan
                std_list[i, j] = np.nan
    for i in range(len(layers)):
        a = ax.scatter(steps[~np.isnan(mean_list[i])],
                       mean_list[i, ~np.isnan(mean_list[i])],
                       marker='o',
                       color=colormap(normalize(i)))
        ax.fill_between(steps[~np.isnan(mean_list[i])],
                        mean_list[i, ~np.isnan(mean_list[i])] -
                        std_list[i, ~np.isnan(std_list[i])],
                        mean_list[i, ~np.isnan(mean_list[i])] +
                        std_list[i, ~np.isnan(std_list[i])],
                        alpha=0.3,
                        color=colormap(normalize(i)))
    ax.set_xscale('log')
    ax.set_xlabel('checkpoints', fontsize=18)
    if method == 'twonn':
        ax.set_ylabel(r'$I_{d}$', fontsize=18)
    elif method == 'pca':
        ax.set_ylabel(r'$d$', fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    cbar = fig.colorbar(a, ax=ax, label='layer', pad=0.5)
    cbar.ax.set_yticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', str((max_layers // 2)), str(max_layers)])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def main():
    results_df = pd.read_csv('id_results_all.csv')
    taskperf_result = pd.read_csv('task_performance_summary.csv')

    plot_layers_over_checkpoints(results_df,
                                 model="6.9b",
                                 method="twonn",
                                 save_path='layers_over_ckpts_twonn.png')
    plot_id_diff_over_ckpts(results_df,
                            models=['6.9b', '1.4b', '410m'],
                            method='twonn',
                            save_path='id_diff_ckpts_twonn.png')
    plot_length_diff(results_df,
                     models=['6.9b', '1.4b', '410m'],
                     method='twonn',
                     save_path='length_diff_twonn.png')
    plot_pca_length_diff(results_df,
                         models=['6.9b', '1.4b', '410m'],
                         save_path='pca_length_diff.png')
    plot_task_perf_and_id(results_df,
                          taskperf_result,
                          model='6.9b',
                          method='twonn',
                          save_path='task_perf_and_id.png')


if __name__ == "__main__":
    main()

