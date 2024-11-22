from typing import Optional

import numpy as np
import statistics as stats_package
from sklearn.metrics import r2_score


def get_auc(df, variable, time_variable) -> tuple[Optional[float], Optional[float]]:
    if len(df):
        tac = df[variable]
        tac = tac.astype(float)
        total_auc = np.trapezoid(tac, dx=0.1)
        auc_per_hour = total_auc / df[time_variable].max()
        return total_auc, auc_per_hour
    else:
        return None, None


def get_mean_stdev_sem(df, variable):
    if len(df):
        tac = df[variable]
        return tac.mean(), tac.std(), tac.sem()
    else:
        return None, None, None


def get_peak(df, variable, window=None):
    if window is None:
        window = {}
    if len(df):
        if len(window) > 0:
            window_length = window['window']
            index = window['index']
            previous_window = window_length if (index - window_length) > 0 else (index - 1)
            post_window = window_length if (index + window_length) < len(df) else ((len(df) - window_length) - 1)
            peak = df.loc[index - previous_window:index + post_window, variable].max()
        else:
            tac = df[variable]
            peak = tac.max()
        # prevent extremely low peaks for division purposes in getting relative peak changes
        return peak if peak > 0.5 else 1
    else:
        return None


def get_peak_index(df, variable):
    if len(df):
        data_series = df[variable]
        return df.index[df[variable] == data_series.max()].tolist()[0]
    else:
        return None


def get_baseline_mean_stdev(df, variable, baseline_count=10):
    if len(df) > baseline_count:
        data_series = df[variable]
        baseline_values = data_series.loc[0:baseline_count]
        baseline_mean = baseline_values.mean()
        baseline_stdev = baseline_values.std()
        return baseline_mean, baseline_stdev
    else:
        return None, None


def get_curve_threshold(df, variable, baseline_count):
    base_mean, base_stdev = get_baseline_mean_stdev(df, variable, baseline_count=baseline_count)
    curve_threshold = base_mean + base_stdev
    return curve_threshold


# TODO: refactor this and next method?
def get_rise_duration(df, variable, time_variable, peak_index, curve_threshold):
    if len(df):
        curve_begins_index = None
        curve_threshold_reached = False
        index_count = 1
        while not curve_threshold_reached:
            if peak_index == 0 | peak_index - index_count == 0:
                curve_begins_index = 0
                break
            if df.loc[peak_index - index_count, variable] < curve_threshold:
                curve_begins_index = peak_index - index_count
                curve_threshold_reached = True
            index_count += 1

        rise_duration = df.loc[peak_index, time_variable] - df.loc[curve_begins_index, time_variable]
        curve_begins_time = df.loc[curve_begins_index, 'Duration_Hrs']
        return rise_duration, curve_begins_index, curve_begins_time
    else:
        return 0, None


def get_fall_duration(df, variable, time_variable, peak_index, curve_threshold):
    if len(df):
        post_peak_data = df.loc[peak_index:]
        unadjusted_curve_threshold = curve_threshold
        curve_threshold_reached = False
        index_count = 1
        curve_ends_index = None
        if (index_count + peak_index) < len(df):
            if peak_index == len(df[variable].tolist()) - 1:
                curve_ends_index = len(df[variable].tolist()) - 1
            else:
                while not curve_threshold_reached:
                    if (peak_index + index_count) == (len(df[variable]) - 1):
                        curve_ends_index = (len(df[variable]) - 1)
                        break
                    if df[variable].loc[peak_index + index_count] < curve_threshold:
                        curve_ends_index = peak_index + index_count
                        curve_threshold_reached = True
                    index_count += 1
                    # slightly increase curve threshold if it hasn't increased more than 5 TAC
                    if unadjusted_curve_threshold + 5 > curve_threshold:
                        curve_threshold += 0.005
        else:
            curve_ends_index = (len(df) - 1)

        fall_duration = post_peak_data.loc[curve_ends_index, time_variable] - post_peak_data.loc[
            peak_index, time_variable]

        if fall_duration == 0:
            fall_duration = 0

        return fall_duration, curve_ends_index, curve_threshold
    else:
        return 0, None


# TODO: how is this method different than the next?
def get_fall_completion(df, tac_variable, curve_ends_index, relative_peak, curve_fall_threshold):
    relative_tac_curve_end = (df.loc[curve_ends_index, tac_variable] - curve_fall_threshold)
    unfinished_fall = relative_tac_curve_end > 0
    if unfinished_fall:
        return (relative_peak - relative_tac_curve_end) / relative_peak
    else:
        return 1


def get_rise_completion(df, tac_variable, curve_start_index, relative_peak, curve_threshold):
    relative_tac_curve_start = (df.loc[curve_start_index, tac_variable] - curve_threshold)
    unfinished_rise = relative_tac_curve_start > 0
    if unfinished_rise:
        return (relative_peak - relative_tac_curve_start) / relative_peak
    else:
        return 1


def get_rise_rate(rise_duration, relative_peak):
    if rise_duration and (relative_peak is not None):
        return relative_peak / rise_duration
    elif rise_duration == 0:
        return 0
    else:
        return None


def get_fall_rate(fall_duration, relative_peak):
    if fall_duration and (relative_peak is not None):
        return relative_peak / fall_duration
    elif fall_duration == 0:
        return 0
    else:
        return None


def get_curve_duration(rise_duration, fall_duration):
    return rise_duration + fall_duration


def get_curve_auc(df, variable, curve_begins_index, curve_ends_index, curve_threshold):
    if curve_begins_index == curve_ends_index:
        curve_ends_index += 1
    if len(df):
        df_curve_only = df.loc[curve_begins_index:curve_ends_index]
        tac_cropped = df_curve_only[variable]
        tac_cropped = tac_cropped.astype(float)
        tac_cropped = np.clip(tac_cropped, curve_threshold, None)
        curve_auc = np.trapezoid(tac_cropped, dx=0.1)
        return curve_auc
    else:
        return None


def get_curve_auc_per_hour(curve_auc, curve_duration):
    if curve_auc and curve_duration:
        return curve_auc / curve_duration
    else:
        return None


def get_avg_tac_diff(df, variable):
    """takes absolute value of the difference between each consecutive TAC difference, then returns the mean of all
    differences and a list of each difference
    """
    if len(df):
        tac = df[variable]
        differences = []
        for i, value in enumerate(tac):
            if i >= 1:
                absolute_difference = abs(value - tac[i - 1])
                differences.append(absolute_difference)
        return stats_package.fmean(differences), differences
    else:
        return None, None


def get_tac_directional_alterations(df, variable):
    """returns the number of times that TAC changes directions
    """
    current_change = None
    if len(df):
        tac = df[variable].tolist()

        alterations = 0
        changes = []
        for i, value in enumerate(tac):
            if i > 0:
                difference = value - tac[i - 1]
                if difference >= 0:
                    current_change = 'positive'
                else:
                    current_change = 'negative'
                changes.append(current_change)
            if i > 1:
                prior_change = changes[i - 2]
                if current_change != prior_change:
                    alterations += 1

        return alterations, len(changes)

    else:
        return None, None


def get_tac_directional_alteration_percent(df, variable):
    """
  returns % of TAC changes that reverse the direction of previous values
  """
    if len(df):
        alterations, n_datapoints = get_tac_directional_alterations(df, variable)

        alteration_percent = (alterations / n_datapoints) * 100
        return alteration_percent
    else:
        return None


def get_discrete_curve_count(df, variable, sampling_rate, curve_threshold, min_curve_duration_hours,
                             max_curve_separation_hours=0.1, min_relative_peak=5):
    """provides the count of discrete curves, determined by consective values above the curve threshold. To count as
    a curve, consecutive values must reach the Minumum Curve Duration (Hrs). A new curve is identified whenever the
    previous curve drops below the curve threshold for longer than the Max Curve Separation (Hrs)"""
    curve_count = 0
    current_curve_duration = 0
    df_tac_curves = df[df[variable] > curve_threshold]
    previous_index = df_tac_curves.index.tolist()[0] - 1 if len(df_tac_curves.index.tolist()) else None
    curve_begin_index = df_tac_curves.index.tolist()[0] - 1 if len(df_tac_curves.index.tolist()) else None

    for i, row in df_tac_curves.iterrows():
        if (((i - previous_index) * sampling_rate) / 60) < max_curve_separation_hours:
            current_curve_duration += ((sampling_rate * (i - previous_index)) / 60)
        else:
            if (current_curve_duration >= min_curve_duration_hours) and (
                    df.loc[curve_begin_index:i, variable].max() > min_relative_peak):
                curve_count += 1
            current_curve_duration = 0
            curve_begin_index = i
        previous_index = i

    return curve_count


def get_value_proportion(df, value, variable):
    """returns proportion (float) of column that equals provided value"""
    if len(df):
        return len(df[df[variable] == value][variable]) / len(df)
    else:
        return None


def get_r2(df, truth, test):
    """Provide dataframe, column name for truth, and column name to test. returns r squared (coefficient of
    determination)"""
    if len(df):
        return r2_score(df[truth].tolist(), df[test].tolist())
    else:
        return None
