from statistics import mean
import numpy as np
from SDM.Feature_Engineering.tac_features import get_peak
from SDM.Signal_Processing.impute import impute
from SDM.Feature_Engineering.get_tac_clusters import get_tac_clusters
from SDM.Signal_Processing.tac_slope import tac_slope


np.seterr(divide='ignore')


def replace_extreme_values(df, time_variable, method="cluster"):
    df['TAC_extreme_values_imputed'] = df['TAC'].tolist()
    tac_col = 'TAC_extreme_values_imputed'
    if method == "cluster":
        cluster_to_remove = None
        if df[tac_col].max() > 600:
            df = get_tac_clusters(df)
            cluster_minumums = [df[df.cluster == cluster_id][tac_col].min() for cluster_id in df['cluster'].unique()]
            cluster_maximums = [df[df.cluster == cluster_id][tac_col].max() for cluster_id in df['cluster'].unique()]
            for i, cluster_id in enumerate(df['cluster'].unique()):
                if i > 0:
                    cluster_separation = cluster_minumums[i] - cluster_maximums[i - 1]
                    if cluster_separation > 200:
                        cluster_to_remove = cluster_id
        error_indices = []
        if cluster_to_remove:
            df.loc[df['cluster'] == cluster_to_remove, tac_col] = np.nan
            error_indices.extend(df[df['cluster'] == cluster_to_remove].index.tolist())
            tac_cleaned, not_imputable = impute(df, df[tac_col].tolist(), time_variable, {}, how='both',
                                                threshold=False)
            df[tac_col] = tac_cleaned
            df['TAC'] = df[tac_col].tolist()

        return df, error_indices

    elif method == "cutoff":
        error_indices = df.index[df[tac_col] > 800]
        i = 0
        while i < len(error_indices):
            start_idx = error_indices[i]
            end_idx = start_idx

            while end_idx in df.index and df.loc[end_idx, tac_col] > 800:
                end_idx += 1

            prev_value = df.loc[start_idx - 1, tac_col] if start_idx - 1 in df.index else np.nan
            next_value = df.loc[end_idx, tac_col] if end_idx in df.index else np.nan

            if not np.isnan(prev_value) and not np.isnan(next_value):
                average = (prev_value + next_value) / 2
                df.loc[start_idx:end_idx - 1, tac_col] = average

            i += (end_idx - start_idx)

        df['TAC'] = df[tac_col].tolist()

        return df, error_indices


def remove_baseline_artifact(df):
    cleaned_idx = []
    if df.loc[0:5, 'TAC'].max() > 10:
        i = df.loc[0:5, 'TAC'].idxmax()
        while i < min(15, len(df) - 2):
            if (df['TAC'].iloc[i:i + 3] < 3).all():
                df.loc[0:i, 'TAC'] = 0
                cleaned_idx = [n for n in range(0, i)]
                break
            i += 1
    return df, cleaned_idx


def impute_artifacts(df, test_range, time_variable, extreme_outliers, major_threshold=0.75, minor_threshold=0.50):
    df_test = df.copy()

    tac_list = df_test['TAC'].tolist()
    tac_list_cleaned = [tac if (i not in extreme_outliers) else np.nan for i, tac in enumerate(tac_list.copy())]
    tac_list_imputed = tac_list.copy()

    counter = 0
    outlier_counter = {
        'major': [],
        'minor': []
    }
    tac_lists_repo = {
        'imputed': {counter: tac_list_imputed.copy()},
        'cleaned': {counter: tac_list_cleaned.copy()}
    }
    cannot_impute = []
    starting_index = {0: 0}
    index_check_count = {}
    for i, tac in enumerate(tac_list_imputed):
        index_check_count[i] = 0

    cycle = 0
    outliers_in_cycle = {}

    all_data_points_checked = False
    while not all_data_points_checked:
        tac_list_current = tac_lists_repo['imputed'][counter]
        start = starting_index[counter]
        if start == 0:
            cycle += 1
            outliers_in_cycle[cycle] = False

        for index, tac in enumerate(tac_list_current):
            if (index > test_range) and (index >= start):
                previous_tac_list = []
                for x in range(1, test_range):
                    previous_tac_list.append(tac_list_current[index - x])

                previous_diff_tac_list = []
                for i in range(0, len(previous_tac_list)):
                    if i > 0:
                        previous_diff_tac_list.append(abs(previous_tac_list[i] - previous_tac_list[i - 1]))

                tac_difference_past_3 = abs(tac_list_current[index] - mean(tac_list_current[index - 3:index]))
                tac_difference_past_1 = abs(tac_list_current[index] - tac_list_current[index - 1])
                if tac_difference_past_1 > 5:
                    local_peak = get_peak(df, 'TAC', window={'index': index, 'window': 100})

                    log_percent_change_from_peak_major = (np.log(tac_difference_past_3)) / (np.log(local_peak) + 0.0001)
                    major_threshold_passed = (log_percent_change_from_peak_major >= major_threshold) and (
                            local_peak > 15) and (index not in cannot_impute)

                    log_percent_change_from_peak_minor = (np.log(tac_difference_past_1)) / (np.log(local_peak) + 0.0001)
                    # cycle count == 1 ---> this ensures all major errors are detected before minor errors
                    minor_threshold_passed = (log_percent_change_from_peak_minor >= minor_threshold) and (
                            local_peak > 5) and (index not in cannot_impute) and (~major_threshold_passed) and (
                                                     cycle > 1)

                    if major_threshold_passed:
                        index_ticker = 0
                        threshold_exceeded = True
                        slope = tac_slope(tac_list_current, index, 10)
                        direction_of_tac_change = 1 if (tac_list_current[index] - tac_list_current[
                            index - 1]) > 0 else -1
                        threshold_projection_direction = direction_of_tac_change

                        while threshold_exceeded:
                            threshold = (((abs(slope)) * threshold_projection_direction) * index_ticker) + \
                                        tac_list_current[index - 1] + (0.05 * threshold_projection_direction)
                            tac_for_testing = tac_list_current[index + index_ticker]
                            threshold_passed = (tac_for_testing > threshold) if (direction_of_tac_change == 1) else (
                                    tac_for_testing < threshold)
                            if threshold_passed:
                                tac_list_current[index + index_ticker] = np.nan
                                tac_list_cleaned[index + index_ticker] = np.nan
                                outlier_counter['major'].append(index + index_ticker)
                                index_ticker += 1
                            if not threshold_passed or ((index + index_ticker) == len(tac_list)):
                                remaining_duration = df['Duration_Hrs'].max() - df.loc[index, 'Duration_Hrs']
                                sides_to_learn_from = 'left' if remaining_duration < 1 else 'both'
                                tac_list_current, not_imputable = impute(df, tac_list_current, time_variable,
                                                                         index_check_count, how=sides_to_learn_from,
                                                                         threshold=False, gap_proportional_limit=0.5)
                                if len(not_imputable) > 0:
                                    for i in not_imputable:
                                        cannot_impute.append(i)
                                tac_lists_repo['imputed'][counter + 1] = tac_list_current
                                tac_lists_repo['cleaned'][counter + 1] = tac_list_cleaned
                                counter += 1
                                if index_ticker > 0:
                                    outliers_in_cycle[cycle] = True
                                else:
                                    outliers_in_cycle[cycle] = False
                                threshold_exceeded = False

                    if minor_threshold_passed:
                        index_ticker = 0
                        threshold_exceeded = True
                        while threshold_exceeded:
                            threshold_tests = []
                            temp_df = df.copy()
                            temp_df.loc[:, 'TAC'] = tac_list_cleaned
                            current_difference = abs(
                                tac_list_current[index + index_ticker] - tac_list_current[index - 1])
                            current_local_peak = get_peak(temp_df, 'TAC',
                                                          window={'window': 100, 'index': index + index_ticker})
                            log_percent_change_from_peak = (np.log(current_difference)) / (
                                    np.log(current_local_peak) + 1)
                            test_1 = (log_percent_change_from_peak > 0.35) and (current_difference > 2)
                            threshold_tests.append(test_1)
                            neighbor_difference = abs(
                                tac_list_current[index + index_ticker] - tac_list_current[index + index_ticker - 1])
                            log_percent_change_from_peak = (np.log(neighbor_difference)) / (
                                    np.log(current_local_peak) + 1)
                            test_2 = (log_percent_change_from_peak > 0.35) and (neighbor_difference > 2)
                            threshold_tests.append(test_2)

                            if any(threshold_tests):
                                tac_list_current[index + index_ticker] = np.nan
                                tac_list_cleaned[index + index_ticker] = np.nan
                                outlier_counter['minor'].append(index + index_ticker)
                                index_ticker += 1
                                if (index + index_ticker) in outlier_counter['major']:
                                    outlier_counter['major'].remove(index + index_ticker)

                            if ~any(threshold_tests) or ((index + index_ticker) == len(tac_list)):
                                tac_list_current, not_imputable = impute(df, tac_list_current, time_variable,
                                                                         index_check_count, how='left', threshold=False,
                                                                         gap_proportional_limit=0.5)
                                if len(not_imputable) > 0:
                                    for i in not_imputable:
                                        cannot_impute.append(i)
                                tac_lists_repo['imputed'][counter + 1] = tac_list_current
                                tac_lists_repo['cleaned'][counter + 1] = tac_list_cleaned
                                counter += 1
                                if index_ticker > 0:
                                    outliers_in_cycle[cycle] = True
                                else:
                                    outliers_in_cycle[cycle] = False
                                threshold_exceeded = False

                tac_review_complete = (index >= (len(tac_list) - 1))
                if tac_review_complete and not outliers_in_cycle[cycle] and (cycle > 1):
                    all_data_points_checked = True
                    break
                if tac_review_complete and (outliers_in_cycle[cycle]) or (tac_review_complete and (cycle == 1)):
                    starting_index[counter] = 0

    return tac_lists_repo['imputed'][counter], tac_lists_repo['cleaned'][counter], set(outlier_counter['major']), set(
        outlier_counter['minor'])
