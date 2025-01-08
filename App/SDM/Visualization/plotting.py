from tokenize import group
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import os
from SDM.Configuration.configuration import normalize_column
from SDM.Feature_Engineering.get_feature_importances import get_feature_importances
from SDM.Visualization.plotting_utils import *
import numpy as np
from sklearn.tree import plot_tree


def plot_column(df, plot_folder, subid, condition, dataset_identifier, y_variable, time_variable, xlabel="Time (hours)",
                event_timestamps={}):
    ylabel = y_variable
    title = f'{y_variable} {subid} {condition}'
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.scatter(y=df[y_variable], x=df[time_variable])
    ax.set_title(title, fontsize=26, fontweight="semibold", pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plot_event_lines(df, ax, event_timestamps, time_variable, 'datetime')
    full_path = f'{plot_folder}/{y_variable} - {subid} - {condition}{dataset_identifier}.png'
    fig.savefig(full_path)
    plt.close('all')
    return full_path


def plot_TAC_curve(df, plot_folder, subid, condition, dataset_identifier, tac_variable, time_variable,
                   ylabel="TAC ug/L", xlabel="Time (hours)", event_timestamps={}):
    title = f'TAC Curve - {tac_variable}'
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.scatter(df[time_variable], df[tac_variable])
    ax.set_title(title, fontsize=26, fontweight="semibold", pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if df[tac_variable].max() < 15:
        ax.set_ylim(df[tac_variable].min() - 1, 15)
    plot_event_lines(df, ax, event_timestamps, time_variable, 'datetime')

    full_path = f'{plot_folder}/{tac_variable} - {subid} - {condition}{dataset_identifier}.png'
    fig.savefig(full_path)
    plt.close('all')
    return full_path


def plot_overlaid_TAC_curves(df, plot_folder, subid, condition, dataset_identifier, tac_variables, time_variable,
                             plot_name, ylabel="TAC ug/L", xlabel="Time (hours)", event_timestamps={}):
    title = f'TAC Curve - {plot_name}'
    fig, ax = plt.subplots(figsize=(16, 7))
    for variable in tac_variables:
        ax.plot(df[time_variable], df[variable])
    ax.set_title(title, fontsize=26, fontweight="semibold", pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plot_event_lines(df, ax, event_timestamps, time_variable, 'datetime')
    if ax.get_ylim()[1] < 10:
        ax.set_ylim(0, 10)

    full_path = f'{plot_folder}/{plot_name} - {subid} - {condition}{dataset_identifier}.png'
    fig.savefig(full_path)
    plt.close('all')
    return full_path


def plot_tac_and_temp(df, plot_folder, subid, condition, dataset_identifier, tac_variable, temp_variable, time_variable,
                      plot_title="TAC and Temperature", event_timestamps={}):
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax1.plot(df[time_variable], df[tac_variable], color="darkblue", label=tac_variable)
    ax2 = ax1.twinx()
    ax2.plot(df[time_variable], df[temp_variable], color="maroon", label=temp_variable)
    ax1.set_title(plot_title, fontsize=26, fontweight="semibold", pad=15)

    ax1.set_ylabel(tac_variable, color='darkblue')
    ax2.set_ylabel(temp_variable, color='maroon')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    if df[tac_variable].max() < 10:
        ax1.set_ylim(0, 10)
    if df[temp_variable].min() > 15:
        ax2.set_ylim(15, 40)

    plot_event_lines(df, ax1, event_timestamps, time_variable, 'datetime')

    path = f'{plot_folder}tac_and_temp_plot_{subid}_{condition}{dataset_identifier}.png'
    fig.savefig(path)
    plt.close('all')

    return path


def plot_device_removal(df, plot_folder, subid, condition, dataset_identifier, temp_variable, time_variable,
                        add_color=False, plot_title="Device Removal Detection", method='Temp Cutoff',
                        motion_variable=None, prediction_column=None, event_timestamps={}, temp_cutoff=27):
    if add_color:
        marker_colors = {
            'correct': ['darkblue', 'gray'],
            'incorrect': ['red', 'orange']
        }
    else:
        marker_colors = {
            'correct': ['black', 'black'],
            'incorrect': ['gray', 'gray']
        }

    fig, ax = plt.subplots(figsize=(16, 7))

    device_on_time, device_on_temp, device_off_time, device_off_temp = split_x_y_from_predictions(df, prediction_column,
                                                                                                  temp_variable,
                                                                                                  time_variable) if (
                                                                                                                                method == 'Model Predictions') or (
                                                                                                                                method == 'Ground Truth') else split_x_y_from_temp_cutoff(
        df, temp_variable, temp_variable, time_variable, cutoff=temp_cutoff)

    ax.scatter(device_on_time, device_on_temp, marker='o', c=marker_colors['correct'][0])
    ax.scatter(device_off_time, device_off_temp, marker='x', c=marker_colors['incorrect'][0])
    ax.set_xlabel('Time (hrs)', fontsize=20)
    ax.set_ylabel('Temperature (C)', fontsize=20)
    ax.set_title(plot_title, fontsize=24, fontweight="semibold", pad=25)
    ax.text(0.5, 1.02, f'Method: {method}', fontsize=12, style='italic',
            ha='center', va='center', transform=ax.transAxes)
    ax.legend(("Passed Temp", "Flagged Temp"), loc='upper left', fontsize=14)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if df[temp_variable].min() > 15:
        ax.set_ylim(15, 40)

    ax.hlines(y=27, xmin=0, xmax=df[time_variable].max(), color='black', linestyle='--')
    if (len(device_off_time) > 0) and (method == 'cutoff'):
        plt.text(df[time_variable].max() * 0.45, 27.0, "Device Not Worn", fontsize=18, fontstyle='italic')
    plot_event_lines(df, ax, event_timestamps, time_variable, 'datetime')

    if motion_variable:
        ax2 = ax.twinx()
        device_on_time, device_on_motion, device_off_time, device_off_motion = split_x_y_from_predictions(df,
                                                                                                          prediction_column,
                                                                                                          motion_variable,
                                                                                                          time_variable) if (
                                                                                                                                        method == 'Model Predictions') or (
                                                                                                                                        method == 'Ground Truth') else split_x_y_from_temp_cutoff(
            df, temp_variable, motion_variable, time_variable, cutoff=27)
        ax2.scatter(device_on_time, device_on_motion, marker='.', c=marker_colors['correct'][1])
        ax2.scatter(device_off_time, device_off_motion, marker='^', c=marker_colors['incorrect'][1])
        ax2.set_ylabel('Motion (G)', fontsize=20, rotation=-90, labelpad=25)
        ax2.legend(("Passed Motion", 'Flagged Motion'), loc='upper right', fontsize=14)
        ax2.tick_params(axis='y', labelsize=16)

    path = f'{plot_folder}{subid}_{condition}{dataset_identifier}_temp_cleaning_{method}.png'
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    return path


def plot_cropping(data, beginning_timestamp, end_timestamp, self, add_color=False, title="Raw Data Cropping"):
    if add_color:
        marker_colors = ['red', 'black']
    else:
        marker_colors = ['lightgray', 'black']

    hours = (data['datetime'] - beginning_timestamp) / np.timedelta64(1, 'h')
    cropped = data.loc[(data['datetime'] < beginning_timestamp) | (data['datetime'] > end_timestamp), 'TAC']
    cropped_time = hours.loc[(data['datetime'] < beginning_timestamp) | (data['datetime'] > end_timestamp)]

    kept = data.loc[(data['datetime'] > beginning_timestamp) & (data['datetime'] < end_timestamp), 'TAC']
    kept_time = hours.loc[(data['datetime'] > beginning_timestamp) & (data['datetime'] < end_timestamp)]

    try:
        cropped_at_max_duration = round(kept_time.max()) == round(self.max_duration)
        timestamps_valid = True
    except:
        timestamps_valid = False
        title = "Invalid Timestamps - Cropping Failed"

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.scatter(kept_time, kept, marker='.', c=marker_colors[1], s=12)
    ax.scatter(cropped_time, cropped, marker='x', c=marker_colors[0], s=14)
    #ax.legend(("Kept", "Cropped"), loc='upper right', fontsize=16)
    ax.set_xlabel('Time (hrs)', fontsize=24)
    ax.set_ylabel('TAC', fontsize=24, labelpad=3)
    ax.set_title(title, fontsize=36, fontweight="semibold", pad=15)
    if ax.get_ylim()[1] < 10:
        ax.set_ylim(0, 10)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    if timestamps_valid:
        if beginning_timestamp > data['datetime'].min():
            ax.vlines(x=0, ymin=-8, ymax=data['TAC'].max(), color='black', linestyle='--')
            plt.text(0.2, data['TAC'].max() * 0.9, "Episode", fontsize=28, fontstyle="italic")
            plt.text(0.2, data['TAC'].max() * 0.8, "Beginning", fontsize=28, fontstyle="italic")
        if end_timestamp < data['datetime'].max():
            ax.vlines(x=kept_time.max(), ymin=-8, ymax=data['TAC'].max(), color='black', linestyle='--')
            cutoff_label = f'{round(self.max_duration)}-Hr' if cropped_at_max_duration else 'Episode End'
            plt.text(kept_time.max() * 0.85, data['TAC'].max() * 0.9, cutoff_label, fontsize=28, fontstyle="italic")
            if cropped_at_max_duration:
                plt.text(kept_time.max() * 0.85, data['TAC'].max() * 0.8, "Cutoff", fontsize=28, fontstyle="italic")
    path = f'{self.plot_folder}/{self.subid}_{self.condition}{self.dataset_identifier}_cropping.png'
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    return path


def plot_cleaning_comparison(occasion, df, time_variable, add_color=False, title="TAC Signal Cleaning", size=(16, 7),
                             legend=True, snip=None, event_timestamps={}):
    if snip:
        df = df.iloc[snip[0]:snip[1]]

    tac_columns = ['TAC_Raw', 'TAC_sloped_start_reassigned', 'TAC_negative_reassigned', 'TAC_gaps_filled',
                   'TAC_extreme_values_imputed', 'TAC_device_off', 'TAC_processed']
    markers = [',', 'o', 'd', 'H', 'v', 's', 'p']
    names = ['Raw (passed)', 'Sloped Beginning Reassigned', 'Subzero Value Reassigned', 'Imputed (Missing Value)',
             'Imputed (Extreme Outlier)', 'Imputed (Device Off)', 'Imputed (Artifact)']

    marker_colors = ['dimgray', 'yellow', 'steelblue', 'peru', 'firebrick', 'mediumvioletred',
                     'maroon'] if add_color else ['black', 'dimgray', 'dimgray', 'dimgray', 'dimgray', 'dimgray',
                                                  'dimgray']

    all_imputed_idx = []

    tac_plot_specs = {}
    for i, col in enumerate(tac_columns):
        if col in df.columns.tolist():
            if i > 0 and (i < len(tac_columns) - 1):
                df_current_col = df[col]
                df_prior_cols = df[[col for col in tac_columns[:i] if col in df.columns]]
                df_post_cols = df[[col for col in tac_columns[i + 1:] if col in df.columns]]

                mask_different = df_prior_cols.ne(df_current_col, axis='index').all(axis=1)
                mask_equal = df_post_cols.eq(df_current_col, axis='index').all(axis=1)

                combined_mask = mask_different & mask_equal
                imputed_idx = combined_mask[combined_mask].index
                all_imputed_idx.extend(imputed_idx)

            elif i == len(tac_columns) - 1:
                df_current_col = df[col]
                df_prior_cols = df[[col for col in tac_columns[:i] if col in df.columns]]

                mask_different = df_prior_cols.ne(df_current_col, axis='index').all(axis=1)
                imputed_idx = mask_different[mask_different].index
                all_imputed_idx.extend(imputed_idx)

            else:
                imputed_idx = []

            if len(imputed_idx) > 0:
                tac_plot_specs[names[i]] = {
                    'x': df.loc[imputed_idx, time_variable].tolist(),
                    'y': df.loc[imputed_idx, col].tolist(),
                    'color': marker_colors[i],
                    'marker': markers[i],
                }

    tac_plot_specs['Raw (passed)'] = {
        'x': df.loc[[i for i in df.index.tolist() if i not in all_imputed_idx], time_variable].tolist(),
        'y': df.loc[[i for i in df.index.tolist() if i not in all_imputed_idx], 'TAC_Raw'].tolist(),
        'color': marker_colors[0],
        'marker': markers[0]
    }

    tac_plot_specs['Raw (removed)'] = {
        'x': df.loc[all_imputed_idx, time_variable].tolist(),
        'y': df.loc[all_imputed_idx, 'TAC_Raw'].tolist(),
        'color': 'crimson' if add_color else 'darkgrey',
        'marker': 'x'
    }

    smoothed_tac = df.loc[:, f'TAC_processed_smooth_{occasion.smoothing_window}'].tolist()
    smoothed_tac_x = df.loc[:, time_variable].tolist()
    fig, ax = plt.subplots(figsize=size)
    ax.plot(smoothed_tac_x, smoothed_tac, linestyle='solid', color='midnightblue' if add_color else 'dimgray',
            linewidth=2)
    for name, specs in tac_plot_specs.items():
        ax.scatter(specs['x'], specs['y'], c=specs['color'], marker=specs['marker'], s=20)
        if add_color and len(specs['x']) and name not in ['Raw (passed)', 'Raw (removed)']:
            ranges = find_ranges(specs['x'], threshold=0.05)
            for start, end in ranges:
                ax.axvspan(start, end, color=specs['color'], alpha=0.2, label='_nolegend_')

    ax.set_xlabel('Time (hrs)', fontsize=22)
    ax.set_ylabel('TAC', fontsize=22)
    if ax.get_ylim()[1] < 10:
        ax.set_ylim(0, 10)
    ax.set_title(title, fontsize=28, fontweight="semibold", pad=15)
    if not occasion.valid_occasion:
        split_text_index = int(round(len(occasion.invalid_reason) / 2))
        split_character_is_space = occasion.invalid_reason[split_text_index] == ' '
        while not split_character_is_space:
            split_text_index += 1
            split_character_is_space = occasion.invalid_reason[split_text_index] == ' '
            if split_character_is_space or split_text_index == len(occasion.invalid_reason) - 1:
                break
        invalid_reason_part_1 = occasion.invalid_reason[:split_text_index]
        invalid_reason_part_2 = occasion.invalid_reason[split_text_index:]
        plt.text(0.03, 0.95, f'Invalid: {invalid_reason_part_1}', ha='left', va='center', transform=plt.gca().transAxes,
                 fontdict={'size': 9})
        plt.text(0.03, 0.92, f'{invalid_reason_part_2}', ha='left', va='center', transform=plt.gca().transAxes,
                 fontdict={'size': 9})
    if legend:
        ax.legend(["Smooth (Final)"] + list(tac_plot_specs.keys()), loc='upper right', fontsize=16)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plot_event_lines(df, ax, event_timestamps, time_variable, 'datetime')

    filename = f'{occasion.plot_folder}cleaning - {occasion.subid} - {occasion.condition}{occasion.dataset_identifier}'
    if snip:
        filename = filename + f' - snip{snip[0]}_{snip[1]}'
    filename = filename + '.png'
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
    return filename


def plot_smoothed_curve(occasion, df, peak, curve_threshold, curve_begins, curve_ends, title="TAC Curve",
                        event_timestamps={}):
    peak_time = df.loc[df[f'TAC_processed_smooth_{occasion.smoothing_window}'] == peak, occasion.time_elapsed_column]
    # graph_cutoff = curve_ends + ((len(df) - curve_ends)*0.25)
    # df = df.loc[:graph_cutoff]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(df[occasion.time_elapsed_column].tolist(), df[f'TAC_processed_smooth_{occasion.smoothing_window}'].tolist(),
            c='black')
    ax.vlines(peak_time, ymin=curve_threshold, ymax=peak, color='black', linestyle='--')
    ax.hlines(curve_threshold, xmin=0, xmax=df[occasion.time_elapsed_column].max(), colors='black', linestyle='--')
    ax.set_xlabel('Time (hrs)', fontsize=24)
    ax.set_ylabel('TAC', fontsize=24)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_title(title, fontsize=36, fontweight="semibold", pad=15)
    if ax.get_ylim()[1] < 10:
        ax.set_ylim(0, 10)
    curve_timepoints = df.loc[[i for i in range(curve_begins, curve_ends)], occasion.time_elapsed_column].tolist()
    curve_tac_values = df.loc[
        [i for i in range(curve_begins, curve_ends)], f'TAC_processed_smooth_{occasion.smoothing_window}'].tolist()
    plt.fill_between(curve_timepoints, curve_tac_values, [curve_threshold for i in range(curve_begins, curve_ends)],
                     color='lightgray')
    plot_event_lines(df, ax, event_timestamps, occasion.time_elapsed_column, 'datetime')

    path = f'{occasion.plot_folder}{occasion.condition}{occasion.dataset_identifier}_smoothed_curve.png'
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    return path


def create_temp_histogram(self):
    temperatures = []
    for occasion in self.occasions:
        data = occasion.dataset['Temperature_C'].tolist()
        temperatures.extend(data)
    temperatures_below_threshold = [temp for temp in temperatures if temp < 27]

    import matplotlib.pyplot as plt
    plt.hist(temperatures, bins=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    plt.ylabel("Counts (1000's)")
    plt.xlabel('Temperature (Celsius)')
    plt.title('Temperature Distribution')
    plt.xticks([20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    plt.yticks(ticks=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000],
               labels=[2, 4, 6, 8, 10, 12, 14, 16, 18])
    plt.savefig(f'{self.analyses_out_folder}/temperature_histogram.png')
    plt.close('all')


def plot_box_whisker(features, variables, ground_truth_variable, plot_folder, cohort_name, filter={}):
    features = features[features['valid_occasion'] == 1].reset_index(drop=True)
    for variable_name, exclude_values in filter.items():
        features = features[~features[variable_name].isin(exclude_values)]
    data = {}
    for variable in variables:
        data[variable] = [[] for i in range(0, len(features[ground_truth_variable].unique()))]
        for i, truth in enumerate(features[ground_truth_variable].unique()):
            data[variable][i] = features[features[ground_truth_variable] == truth][variable].tolist()

    figure, axis = plt.subplots(1, len(variables))
    figure.set_figheight(12)
    figure.set_figwidth(10 * len(variables))
    colors = ['firebrick', 'silver', 'darkblue']
    ticker = 0
    positions = [i + 1 for i in range(0, len(features[ground_truth_variable].unique()))]
    for variable in variables:
        bp = axis[0 + ticker].boxplot(data[variable], positions=positions, widths=0.5, patch_artist=True)
        axis[0 + ticker].set_title(variable)
        axis[0 + ticker].set_xticks([(i + 1) for i in range(0, len(features[ground_truth_variable].unique()))])
        axis[0 + ticker].set_xticklabels(features[ground_truth_variable].unique().tolist())
        axis[0 + ticker].grid(color='lightgray', linestyle='--', linewidth=0.75, axis='y')
        ticker += 1
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)

    full_path = f'{plot_folder}/features_{cohort_name}_{ground_truth_variable}_box_whisker.png'
    figure.savefig(full_path, bbox_inches='tight')
    plt.close('all')


def plot_rf_feature_importances(model, model_name, model_figures_folder):
    forest_importance = get_feature_importances(model.best, model.predictors)
    fig, ax = plt.subplots()
    ax.grid(True, axis='x')
    forest_importance.plot.barh(ax=ax, color='dimgray')
    ax.set_title(f'Feature Importance - Random Forest', fontdict={'weight': 'bold', 'size': 15})
    ax.set_xlabel("Mean Decrease in Impurity", fontdict={'size': 13})
    ax.get_legend().remove()
    fig.tight_layout()

    plt.savefig(f'{model_figures_folder}/{model_name} - Feature Importances.png', dpi=55)
    plt.close('all')
    return forest_importance


def plot_rf_tree(rf, feature_names, cohort_name, model_figures_folder):
    fig = plt.figure(figsize=(15, 10))
    plot_tree(rf.best.estimators_[0],
              feature_names=feature_names, class_names=['Alc', 'Non'],
              filled=True, impurity=True, rounded=True)
    fig.savefig(f'{model_figures_folder}/Random Forest - {cohort_name} - Decision Tree.png')
    plt.close('all')
