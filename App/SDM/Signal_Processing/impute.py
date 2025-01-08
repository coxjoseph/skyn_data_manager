import pandas as pd
import numpy as np
from SDM.Feature_Engineering.tac_features import get_peak
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
import scipy.interpolate
from sklearn.gaussian_process.kernels import DotProduct, Matern, ConstantKernel
import math


# TODO: refactor
def impute_tac_in_gaps(df, tac_variable, time_elapsed_variable, sampling_rate, hours_elapsed_threshold):
    gaps_filled_df = df.copy()
    gap_rows_filled = 0
    for idx, row in df.iterrows():
        if idx > 0:
            hours_elapsed_between_readings = row[time_elapsed_variable] - df.loc[idx - 1, time_elapsed_variable]
            if hours_elapsed_between_readings > hours_elapsed_threshold:
                rows_to_add = math.floor(((hours_elapsed_between_readings * 60) / sampling_rate))
                gap_rows = pd.DataFrame(dict(zip(gaps_filled_df.columns.tolist(),
                                                 [[None for _ in range(0, rows_to_add)] for _ in
                                                  range(0, len(gaps_filled_df.columns))])))
                gap_rows['Duration_Hrs'] = [
                    gaps_filled_df.loc[gap_rows_filled + idx - 1, 'Duration_Hrs'] + ((sampling_rate * (i + 1)) / 60) for
                    i in range(0, rows_to_add)]
                gap_rows['datetime'] = [gaps_filled_df.loc[gap_rows_filled + idx - 1, 'datetime'] + pd.Timedelta(
                    minutes=(sampling_rate * (i + 1))) for i in range(0, rows_to_add)]
                gap_rows[tac_variable] = [np.nan for _ in range(0, rows_to_add)]
                gap_rows['device_id'] = [row['device_id'] for _ in range(0, rows_to_add)]
                gap_rows['Firmware Version'] = [row['Firmware Version'] for _ in range(0, rows_to_add)]
                if 'app version' in df.columns:
                    gap_rows['app version'] = [row['app version'] for _ in range(0, rows_to_add)]
                if 'device time zone' in df.columns:
                    gap_rows['device time zone'] = [row['device time zone'] for _ in range(0, rows_to_add)]
                gap_rows['gap_imputed'] = [1 for _ in range(0, rows_to_add)]
                if 'gap_imputed' not in gaps_filled_df.columns:
                    gaps_filled_df['gap_imputed'] = [0 for _ in range(0, len(gaps_filled_df))]

                before_gap = gaps_filled_df.iloc[:idx + gap_rows_filled]
                after_gap = gaps_filled_df.iloc[idx + gap_rows_filled:]
                gaps_filled_df = pd.concat([before_gap, gap_rows, after_gap], ignore_index=True)
                gaps_filled_df.reset_index(inplace=True, drop=True)

                imputed_tac_list, not_imputable = impute(gaps_filled_df, gaps_filled_df[tac_variable].tolist(),
                                                         'Duration_Hrs', {}, gap_proportional_limit=0.7,
                                                         override_index_check_count=True)
                gaps_filled_df['TAC_gaps_filled'] = imputed_tac_list
                gaps_filled_df[tac_variable] = gaps_filled_df['TAC_gaps_filled']
                gap_rows_filled += rows_to_add

    return gaps_filled_df


def impute(df_prior, tac_list, time_variable, index_check_count, knot_proportion=0.10, variable='TAC', how='both',
           threshold=False, gap_proportional_limit=0.40, override_index_check_count=False, extend_missing_idx=0):
    training_data = None  # TODO: Holdover for pep - rewrite after refactor
    df = df_prior.copy()
    cannot_impute = []

    # get a list of indices where artifacts have been removed
    missing_idx = [i for (i, tac) in enumerate(tac_list) if np.isnan([tac])]

    # artifact gap cannot take more than X% of dataset
    gap_limit = len(df) * gap_proportional_limit

    # create a list of each gap
    gaps = [[]]
    gap_count = 0
    for i, missing_tac_idx in enumerate(missing_idx):
        gaps[gap_count].append(missing_tac_idx)
        if i + 1 < len(missing_idx):
            # if there is a gap between the current missing_idx and the next
            if (missing_tac_idx + 1) != missing_idx[i + 1]:
                missing_idx_to_add = [n for n in range(missing_tac_idx + 1, missing_tac_idx + 1 + extend_missing_idx) if
                                      (n < len(df)) and (n < missing_idx[i + 1])]
                if len(missing_idx_to_add):
                    for idx in missing_idx_to_add:
                        gaps[gap_count].append(idx)
                    # if the newly added idx still leaves a gap until the next missing idx
                    if max(missing_idx_to_add) + 1 != missing_idx[i + 1]:
                        gap_count += 1
                        gaps.append([])
        else:
            missing_idx_to_add = [n for n in
                                  range(missing_tac_idx + 1, missing_tac_idx + 1 + round(extend_missing_idx / 3)) if
                                  (n < len(df))]
            if len(missing_idx_to_add):
                for idx in missing_idx_to_add:
                    gaps[gap_count].append(idx)

    # If artifacts/device off gaps are less than 15 values apart, then take those two smaller gaps
    # and combine to make large gap
    i = 0
    while i < len(gaps) - 1:
        current_gap = gaps[i]
        next_gap = gaps[i + 1]
        gap_distance = next_gap[0] - current_gap[-1]
        # if gap is less than or equal to 15, then combine lists
        if 0 < gap_distance <= 15:
            filled_gap = list(range(current_gap[-1] + 1, next_gap[0]))
            combined_list = current_gap + filled_gap + next_gap
            gaps[i] = combined_list
            gaps.pop(i + 1)
        else:
            i += 1

    if len(missing_idx) > 0:
        for gap in gaps:
            if len(gap) < gap_limit:
                if len(index_check_count) > 0:
                    gap_index_check_count = {k: index_check_count[k] for k in gap}
                    max_impute_attempts = max(gap_index_check_count.values())
                elif override_index_check_count:
                    max_impute_attempts = 1
                else:
                    max_impute_attempts = 0
                    # how many data points to use for building spline? X % of the dataset.
                # for front of gap
                first_missing_id = gap[0]
                data_limit = (int(len(df) * knot_proportion)) if (len(df) * knot_proportion) > 30 else 30
                front_ticker = 0
                # while the index is not beyond the length of df or greater than our local data limit
                while (first_missing_id - front_ticker > 0) and (front_ticker <= data_limit):
                    front_ticker += 1
                front_index = first_missing_id - front_ticker
                data_before_gap = df.loc[front_index:first_missing_id - 1]
                data_before_gap.reset_index(inplace=True)

                # for behind the gap
                last_missing_id = gap[-1]
                data_limit = (int(gap_limit / 2)) if gap_limit > 75 else 30
                back_ticker = 0
                # while the index is not beyond the length of df or greater than our local data limit
                while (last_missing_id + back_ticker < len(df)) and (back_ticker <= data_limit):
                    back_ticker += 1
                back_index = last_missing_id + back_ticker
                data_after_gap = df.loc[last_missing_id + 1:back_index]
                data_after_gap.reset_index(inplace=True)
                # before and after gap

                # sometimes the data after gap has values that are artifacts. while we do not want to impute for
                # those now, we do not want those to be used to train the imputation model.
                # this below loop detects those artifacts for removal
                outlier_indices = {'before': [], 'after': []}
                if threshold:
                    for position, dataset in {'before': data_before_gap, 'after': data_after_gap}.items():
                        local_peak = get_peak(dataset, 'TAC')
                        for i, row in dataset.iterrows():
                            if i > 0:
                                tac_difference = abs(row['TAC'] - dataset.loc[i - 1, 'TAC'])
                                if (local_peak > 1) and (tac_difference > 1):
                                    if ((np.log(tac_difference) / (np.log(local_peak) + 0.0001)) > threshold) and (
                                            local_peak > 3):
                                        outlier_indices[position].append(i)
                data_before_gap.loc[outlier_indices['before'], 'TAC'] = np.nan
                data_after_gap.loc[outlier_indices['after'], 'TAC'] = np.nan
                data_around_gap = pd.concat([data_before_gap, data_after_gap])

                if len(data_after_gap) == 0:
                    how = 'left'
                elif how == 'flex':
                    remaining_duration = df_prior['Duration_Hrs'].max() - data_after_gap['Duration_Hrs'].min()
                    tac_diff_gap = data_after_gap.loc[min(data_after_gap.index.tolist()), 'TAC'] - data_before_gap.loc[
                        max(data_before_gap.index.tolist()), 'TAC']
                    local_peak = get_peak(df, 'TAC',
                                          window={'index': max(data_before_gap.index.tolist()), 'window': 100})
                    tac_diff_gap_too_big = (abs(tac_diff_gap) / local_peak) > 0.5 and tac_diff_gap > 10
                    if remaining_duration < 1 or tac_diff_gap_too_big:
                        training_data = data_before_gap
                    else:
                        training_data = data_around_gap

                if how != 'flex':
                    key = {
                        'left': data_before_gap,
                        'right': data_after_gap,
                        'both': data_around_gap
                    }
                    training_data = key[how]

                x = training_data[~pd.isna(training_data[variable])][time_variable]
                y = training_data[~pd.isna(training_data[variable])][variable]
                x_with_gap = df[front_index:back_index][time_variable]

                # 4 models can be attempted
                # if model fails to bring value below artifact threshold, then next model will be attempted

                if max_impute_attempts == 1:
                    kernel = Matern(length_scale=0.4, nu=0.3, length_scale_bounds=(1e-3, 1e3)) * ConstantKernel(
                        constant_value=5)
                    model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x.to_numpy().reshape(-1, 1), y)
                    predictions = model.predict(x_with_gap.to_numpy().reshape(-1, 1))
                    data_to_insert = predictions[front_ticker:front_ticker + (last_missing_id - first_missing_id + 1)]
                    tac_list[first_missing_id:last_missing_id + 1] = data_to_insert
                elif max_impute_attempts == 2:
                    model = LinearRegression().fit(x.to_numpy().reshape(-1, 1), y)
                    predictions = model.predict(x_with_gap.to_numpy().reshape(-1, 1))
                    data_to_insert = predictions[front_ticker:front_ticker + (last_missing_id - first_missing_id + 1)]
                    tac_list[first_missing_id:last_missing_id + 1] = data_to_insert
                elif (max_impute_attempts == 3) and (how == 'both'):
                    y_interp = scipy.interpolate.interp1d(x.tolist(), y.tolist())
                    predictions = y_interp(x_with_gap)
                    data_to_insert = predictions[front_ticker:front_ticker + (last_missing_id - first_missing_id + 1)]
                    tac_list[first_missing_id:last_missing_id + 1] = data_to_insert
                else:
                    model = GaussianProcessRegressor(kernel=DotProduct(), random_state=0).fit(
                        x.to_numpy().reshape(-1, 1), y)
                    predictions = model.predict(x_with_gap.to_numpy().reshape(-1, 1))
                    data_to_insert = predictions[front_ticker:front_ticker + (last_missing_id - first_missing_id + 1)]
                    tac_list[first_missing_id:last_missing_id + 1] = data_to_insert
                    for i in gap:
                        cannot_impute.append(i)

            else:
                for i in gap:
                    cannot_impute.append(i)

    tac_list = [tac if tac >= 0 else 0 for tac in tac_list]

    return tac_list, cannot_impute
