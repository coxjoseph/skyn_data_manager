# FIXME: huh?
ordered_feature_columns = [
    "subid", "dataset_identifier", "episode_identifier", "condition", "drinks", "binge", "AUD",
    "RF_Alc_vs_Non_prediction", "LR_Alc_vs_Non_prediction", "RF_Binge_prediction", "LR_Binge_prediction",
    "RF_Alc_vs_Non_default_prediction", "RF_Alc_vs_Non_default_correct", "LR_Alc_vs_Non_default_prediction",
    "LR_Alc_vs_Non_default_correct", "RF_Binge_default_prediction", "RF_Binge_default_correct",
    "LR_Binge_default_prediction", "LR_Binge_default_correct", "valid_occasion", "device_start", "crop_begin",
    "crop_end", "cropped_device_off_end", "cropped_device_off_start", "removal_detect_method", "device_off_N",
    "not_worn_percent", "not_worn_duration", "worn_duration", "worn_duration_percent", "device_inactive_perc",
    "device_inactive_duration", "device_active_perc",
    "device_active_duration", "subzero_reassigned_zero_count", "extreme_outlier_N", "major_outlier_N",
    "minor_outlier_N", "imputed_N", "tac_imputed_perc", "fall_completion", "baseline_range", "curve_threshold_RAW",
    "baseline_mean_RAW", "baseline_stdev_RAW", "mean_RAW", "stdev_RAW", "sem_RAW", "auc_total_RAW", "auc_per_hour_RAW",
    "peak_RAW", "relative_peak_RAW", "rise_duration_RAW", "curve_start_time", "fall_duration_RAW", "rise_rate_RAW",
    "fall_rate_RAW", "duration_RAW", "curve_duration_RAW", "curve_auc_RAW", "curve_auc_per_hour_RAW", "no_motion_RAW",
    "mean_motion_RAW", "stdev_motion_RAW", "sem_motion_RAW", "mean_temp_RAW", "stdev_temp_RAW", "sem_temp_RAW",
    "avg_tac_diff_RAW", "tac_alt_perc_RAW", "curve_alterations_RAW", "completed_curve_count_RAW",
    "curve_begins_index_RAW",
    "curve_ends_index_RAW", "curve_threshold_CLN", "TAC_N_CLN", "baseline_mean_CLN", "baseline_stdev_CLN", "mean_CLN",
    "stdev_CLN", "sem_CLN", "auc_total_CLN", "auc_per_hour_CLN", "peak_CLN", "relative_peak_CLN", "rise_duration_CLN",
    "fall_duration_CLN", "rise_rate_CLN", "fall_rate_CLN", "duration_CLN", "curve_duration_CLN", "curve_auc_CLN",
    "curve_auc_per_hour_CLN", "no_motion_CLN", "mean_motion_CLN", "stdev_motion_CLN", "sem_motion_CLN", "mean_temp_CLN",
    "stdev_temp_CLN", "sem_temp_CLN", "avg_tac_diff_CLN", "tac_alt_perc_CLN", "curve_alterations_CLN",
    "completed_curve_count_CLN", "curve_begins_index_CLN", "curve_ends_index_CLN"
]


def get_ordered_dataset_columns(smoothing_window):
    return [
        "SubID", "Dataset_Identifier", "Episode_Identifier", "Full_Identifier", "Row_ID", "email", "datetime",
        "device_id", "TAC", "Motion", "Temperature_C", "Firmware Version", "Duration_Hrs", "TAC_Raw", "device_on",
        "temp", "temp_a_pre", "temp_b_pre", "temp_c_pre", "temp_a_post", "temp_b_post", "temp_c_post",
        "temp_mean_change_pre",
        "temp_mean_change_post", "temp_change_pre", "temp_change_post", "motion", "motion_a_pre", "motion_b_pre",
        "motion_c_pre", "motion_a_post", "motion_b_post", "motion_c_post", "motion_mean_change_pre",
        "motion_mean_change_post", "motion_change_pre", "motion_change_post", "TAC_sloped_start_reassigned",
        "sloped_start",
        "TAC_negative_reassigned", "negative_reassigned_zero", "gap_imputed", "TAC_gaps_filled",
        "TAC_extreme_values_imputed",
        "extreme_outlier", "device_on_pred", "TAC_device_off", "TAC_device_off_imputed", "TAC_processed",
        "TAC_artifacts_removed", "major_outlier", "minor_outlier", f'TAC_smooth_{smoothing_window}',
        f'TAC_processed_smooth_{smoothing_window}'
    ]
