from .skyn_datapoint import skynDatapoint
from ..Configuration.configuration import *
from ..Crop.crop import *
from ..Signal_Processing.smooth_signal import *
from ..Signal_Processing.remove_outliers import *
from ..Signal_Processing.impute import impute_tac_in_gaps, impute
from ..Visualization.plotting import *
from ..Feature_Engineering.tac_features import *
from ..Configuration.file_management import *
from ..Documenting.dataset_workbook import export_skyn_workbook
from ..Signal_Processing.revise_incomplete_features import revise_fall_features, revise_rise_features
import pandas as pd
import numpy as np
import traceback
from typing import Any, Optional
from pandas import NaT


class SkynDataset:
    def __init__(self, path, data_out_folder, graphs_out_folder, subid, dataset_identifier, episode_identifier='e1',
                 disable_crop_start=True, disable_crop_end=True, skyn_upload_timezone='CST', max_duration=18,
                 metadata_path='', metadata=pd.DataFrame()):
        self.path = path

        #Subid, Dataset ID, Episode ID
        self.subid = subid
        self.dataset_identifier = dataset_identifier
        self.episode_identifier = episode_identifier

        #COHORT METADATA
        if len(metadata) > 0:
            self.metadata = metadata
            self.metadata_index = self.get_metadata_index()
        elif metadata_path == '':
            self.metadata = pd.DataFrame(
                columns=['SubID', 'Condition', 'Dataset_Identifier', 'Episode_Identifier', 'Use_Data', 'Notes',
                         'Crop Begin Date', 'Crop Begin Time', 'Crop End Date', 'Crop End Time', 'Time Zone'],
                data=[
                    [self.subid, 'Unk', self.dataset_identifier, int(self.episode_identifier[1:]), 'Y', '', '', '', '',
                     '', '']]
            )
            self.metadata = configure_timestamps(self.metadata)
            self.metadata_index = 0
        else:
            self.metadata = configure_timestamps(pd.read_excel(metadata_path))
            self.metadata_index = self.get_metadata_index()

        self.event_timestamps = self.get_event_timestamps(metadata_path)  #will return {} if no path

        #outcomes/variables of interest/self-reported info
        self.condition = self.load_metadata(column='Condition')
        if self.condition != 'Non' and self.condition != 'Alc':
            self.condition = 'Unk'
        self.drinks = self.load_metadata(column='TotalDrks')
        self.sex = self.load_metadata(column='Sex')
        self.binge = self.is_binge()
        self.aud = self.load_metadata(column='AUD')
        self.metadata_note = self.load_metadata(column='Notes')

        #Full Identifier, cohort info
        self.full_identifier = get_full_identifier(self.subid, self.dataset_identifier, self.episode_identifier)
        self.cohort_full_identifiers = get_cohort_full_identifiers(self.metadata)

        #load data
        self.unprocessed_dataset = self.load_dataset()
        self.sampling_rate = 1  #biosensor readings per minute. this is updated in the command below
        self.dataset: pd.DataFrame = self.configure_raw_data()

        #DISABLE
        self.disabled_by_multiple_device_ids = includes_multiple_device_ids(self.dataset)

        #CROPPING INFO
        self.begin = self.dataset['datetime'].min()
        self.end = self.dataset['datetime'].max()
        self.max_duration = max_duration

        self.time_elapsed_column = 'Duration_Hrs'  #updated after cropping/processing
        self.cropping_timestamp_available_begin = self.timestamp_available('Begin')
        self.cropping_timestamp_available_end = self.timestamp_available('End')

        self.crop_start_with_timestamps = (~disable_crop_start) and self.cropping_timestamp_available_begin
        self.crop_end_with_timestamps = (~disable_crop_end) and self.cropping_timestamp_available_end

        self.skyn_upload_timezone = skyn_upload_timezone
        self.crop_begin_adjustment = 0
        self.crop_end_adjustment = 0
        self.device_start_date, self.device_start_datetime = get_start_date_time(self.dataset)
        self.crop_begin = None  #updated after cropping
        self.crop_end = None  #updated after cropping
        self.baseline_corrected = False
        self.data_cropped = False

        #EXPORT PATH INFO
        self.data_out_folder = data_out_folder
        self.plot_folder = graphs_out_folder
        self.simple_plot_paths = []
        self.complex_plot_paths = []
        self.plot_paths = self.simple_plot_paths + self.complex_plot_paths
        self.official_curve_plot = None
        self.device_removal_model_preds_plot = None
        self.device_removal_temp_cutoff_plot = None
        self.device_removal_ground_truth_plot = None
        self.device_removal_plot = None
        self.tac_and_temp_plot = None
        self.cropping_plot = None
        self.cleaning_comparison_plot = None
        self.motion_plot = None

        #CLEANING INFO
        self.device_removal_detection_method = None
        self.stats = {}

        # self.smoothing_window = nearest_odd(51/self.sampling_rate)
        self.smoothing_window = 51

        """gaps in data shorter than this duration will be filled via imputation"""
        self.gap_fill_max_duration = 0.5  #hours

        #these two thresholds represent proportion between consecutive TAC differences and local peak
        self.major_cleaning_threshold = 0.8
        self.minor_cleaning_threshold = 0.5
        self.baseline_range = 10
        self.fall_revision_threshold = 0.5  #hours; if exceeded, fall features will be revised using algorithm
        self.rise_revision_threshold = 0.5

        #EXCLUSION CRITERIA
        self.min_duration_active = 1  #hours
        self.max_percentage_inactive = 30
        self.max_percentage_imputed = 65
        self.valid_occasion = 1
        self.invalid_reason = None

        #MODEL PREDICTIONS
        self.predictions = {}

    def process_with_default_settings(self, make_plots=False, export=True):
        self.create_output_folders()

        print(self.subid)
        print(self.dataset_identifier)
        print(self.episode_identifier)
        print(self.metadata)

        self.dataset['TAC_Raw'] = self.dataset['TAC'].tolist()

        self.valid_occasion, self.invalid_reason = self.determine_initial_validity()

        if self.valid_occasion:
            self.crop_dataset_based_on_temperature()
            if self.crop_start_with_timestamps or self.crop_end_with_timestamps:
                self.crop_dataset_with_timestamps()
                self.data_cropped = True
            elif self.max_duration:
                self.crop_dataset_with_max_duration()
                self.data_cropped = True
            if make_plots:
                self.plot_tac_and_temp()
                self.motion_plot = self.plot_column('Motion')

        if self.valid_occasion:
            self.valid_occasion = 1 if len(self.dataset) > 20 else 0
            self.invalid_reason = 'Not enough data' if len(self.dataset) < 20 else None

        if self.valid_occasion:
            self.get_row_features()
            self.set_curve_threshold(dataset_version='RAW')
            self.get_episode_features(dataset_version='RAW')

        if self.valid_occasion:
            self.clean_tac_signal()
            self.smooth_tac_signal(self.smoothing_window, 3, ['TAC', 'TAC_processed'])
            self.valid_occasion, self.invalid_reason = self.determine_post_cleaning_validity()

        if self.valid_occasion:
            self.set_curve_threshold(dataset_version='CLN', validate=True)

        if self.valid_occasion:
            self.get_episode_features(dataset_version='CLN')
            revise_fall_features(self, fall_revision_threshold=self.fall_revision_threshold)
            revise_rise_features(self, rise_revision_threshold=self.rise_revision_threshold)
            #calculate if device off / gaps in data exceed 50% of dataset

        if make_plots:
            try:
                self.plot_cleaning_comparison()
            except:
                pass
            try:
                self.plot_official_tac()
            except:
                pass
            try:
                self.plot_device_removal()
            except:
                pass

        if export:
            try:
                self.export_workbook()
            except:
                print(traceback.format_exc())

    def clean_tac_signal(self):

        #CLEANING STEP 1: Reassign start of data to 0 if on large downward slope
        df = self.dataset.copy()
        tac_starting_values = df.loc[:12, 'TAC']
        slope, intercept = np.polyfit(np.arange(len(tac_starting_values)), tac_starting_values, 1)
        if slope < -3 and df.loc[:12, 'TAC'].min() < 10:
            df.loc[:12, 'TAC'] = 0
            df['TAC_sloped_start_reassigned'] = [0 for i in range(0, 12)] + df.loc[12:, 'TAC'].tolist()
            df['sloped_start'] = [1 for i in range(0, 12)] + [0 for i in range(12, len(df))]
        else:
            df['TAC_sloped_start_reassigned'] = df['TAC'].tolist()
            df['sloped_start'] = 0
        self.dataset = df

        #CLEANING STEP 2: Reassign negative values to 0
        df = self.dataset.copy()
        negative_TAC_reassigned = [tac if tac > 0 else 0 for tac in df['TAC'].tolist()]
        df['TAC'] = negative_TAC_reassigned
        df['TAC_negative_reassigned'] = negative_TAC_reassigned
        df['negative_reassigned_zero'] = [1 if tac > 0 else 0 for tac in df['TAC'].tolist()]
        self.dataset = df

        #CLEANING STEP 3: Fill in "gaps" (i.e. missing data)
        df = self.dataset.copy()
        if (len(df) * self.sampling_rate) > 60:
            df = impute_tac_in_gaps(df, 'TAC', 'Duration_Hrs', self.sampling_rate, self.gap_fill_max_duration)
        if 'gap_imputed' not in self.dataset.columns:
            df['gap_imputed'] = [0 for i in range(0, len(df))]
            df['TAC_gaps_filled'] = df['TAC']
        self.dataset = df

        #CLEANING STEP 4: replace extreme values, i.e., values impossibly high
        if (len(self.dataset) > 30) and (max(self.dataset['TAC']) - min(self.dataset['TAC']) > 5):
            self.dataset, extreme_outliers = replace_extreme_values(self.dataset, 'Duration_Hrs', method='cutoff')
            # self.dataset, cleaned_idx = remove_baseline_artifact(self.dataset)
            self.dataset['extreme_outlier'] = [1 if i in extreme_outliers else 0 for i in range(0, len(self.dataset))]
        else:
            extreme_outliers = []
            self.dataset['extreme_outlier'] = [0 for i in range(0, len(self.dataset))]

        #CLEANING STEP 5: Impute values where device is removed
        if self.device_removal_detection_method:
            self.detect_device_removal()

        #CLEANING STEP 6: Impute environmental artifacts and excessive noise/jumps
        df = self.dataset.copy()
        if (len(self.dataset) > 30) and (max(df['TAC']) - min(df['TAC']) > 5) and (self.valid_occasion == 1):
            imputed_TAC, cleaned_TAC, major_outliers, minor_outliers = impute_artifacts(df, self.baseline_range,
                                                                                        self.time_elapsed_column,
                                                                                        extreme_outliers,
                                                                                        self.major_cleaning_threshold,
                                                                                        self.minor_cleaning_threshold)
            df['TAC_processed'] = imputed_TAC
            df['TAC_artifacts_removed'] = cleaned_TAC
            df['major_outlier'] = [(1 if i in major_outliers else 0) for i in range(0, len(df))]
            df['minor_outlier'] = [(1 if i in minor_outliers else 0) for i in range(0, len(df))]
            df['TAC'] = df['TAC_processed'].tolist()

        else:
            df['TAC_processed'] = df['TAC']
            df['TAC_artifacts_removed'] = df['TAC']
            df['major_outlier'] = 0
            df['minor_outlier'] = 0
        self.dataset = df

        if 'TAC_processed' not in self.dataset.columns:
            self.dataset['TAC_processed'] = self.dataset['TAC'].tolist()
            self.dataset['TAC_artifacts_removed'] = self.dataset['TAC'].tolist()

    def smooth_tac_signal(self, window_length, polyorder, variables):
        smoothed_tac_variables = smooth_signals(self.dataset, window_length, polyorder, variables)
        for variable in [v for v in variables if v in self.dataset.columns]:
            smoothed_tac = smoothed_tac_variables[f'{variable}_{window_length}']
            smoothed_variable_name = f'{variable}_smooth_{window_length}'
            if min(smoothed_tac) < 0:
                smoothed_tac = [tac - min(smoothed_tac) for tac in smoothed_tac]
            self.dataset[smoothed_variable_name] = smoothed_tac

    def crop_dataset_based_on_temperature(self):
        self.dataset, cropped_device_off_end = crop_device_off_end(self.dataset)
        self.stats['cropped_device_off_end'] = cropped_device_off_end
        self.dataset, cropped_device_off_start = crop_device_off_start(self.dataset)
        self.stats['cropped_device_off_start'] = cropped_device_off_start
        self.dataset.reset_index(drop=True, inplace=True)

    def crop_dataset_with_timestamps(self):
        self.dataset, cropped_plot_path = crop_using_timestamps(self, self.dataset)
        self.dataset = get_time_elapsed(self.dataset, 'datetime')
        self.complex_plot_paths.append(cropped_plot_path)
        self.cropping_plot = cropped_plot_path
        if len(self.dataset) < 10:
            self.valid_occasion = 0
            self.invalid_reason = 'No data within cropping timestamps'

    def crop_dataset_with_max_duration(self):
        min_timestamp = self.dataset['datetime'].min()
        max_timestamp = min_timestamp + pd.DateOffset(hours=self.max_duration)
        self.dataset = self.dataset[
            (self.dataset['datetime'] >= min_timestamp) & (self.dataset['datetime'] <= max_timestamp)]
        self.dataset.reset_index(drop=True, inplace=True)
        self.dataset = get_time_elapsed(self.dataset, 'datetime')

    def detect_device_removal(self):
        """
    Will detect device removal based on Temp value or trained algorithm (which assesses recent temp and motion changes), then impute regions where device was removed or unstable. 
    """
        self.stats['removal_detect_method'] = self.device_removal_detection_method
        df = self.dataset
        if 'Temp' in self.device_removal_detection_method:
            temp_cutoff = int(self.device_removal_detection_method[-11:-10])
            df.loc[:, 'device_on_pred'] = df.apply(lambda row: 1 if row['Temperature_C'] > temp_cutoff else 0, axis=1)
            counts = df['device_on_pred'].value_counts()
        elif self.device_removal_detection_method == 'ground_truth':
            df.loc[:, 'device_on_pred'] = np.nan  #no predictions relevant
        elif self.device_removal_detection_method == 'Built-in Algorithm':
            try:
                model = load_default_model('worn_vs_removed')
                prediction_features = df.loc[:, [col for col in df.columns if col in model.predictors]].dropna()
                predictions = model.predict(prediction_features)
                nan_indices = df[[col for col in df.columns if col in model.predictors]].isna().any(axis=1)
                prediction_indices = ~nan_indices
                df['device_on_pred'] = 1
                df.loc[prediction_indices, 'device_on_pred'] = predictions
                counts = df['device_on_pred'].value_counts()
            except:
                temp_cutoff = 27
                self.device_removal_detection_method = f'Temp Cutoff ({temp_cutoff} Celsius)'
                df.loc[:, 'device_on_pred'] = df.apply(lambda row: 1 if row['Temperature_C'] > temp_cutoff else 0,
                                                       axis=1)
                counts = df['device_on_pred'].value_counts()
        else:
            df.loc[:, 'device_on_pred'] = np.nan

        try:
            self.stats['not_worn_percent'] = (len(df) - counts[1]) / len(df)
        except:
            self.stats['not_worn_percent'] = 0

        total_duration = df[self.time_elapsed_column].max()
        self.stats[f'not_worn_duration'] = self.stats[f'not_worn_percent'] * total_duration
        self.stats[f'worn_duration'] = total_duration - self.stats[f'not_worn_duration']
        self.stats[f'worn_duration_percent'] = self.stats[f'worn_duration'] / total_duration

        df_copy = df.copy()
        tac_removed_device_off = df.apply(lambda row: row['TAC'] if row['device_on_pred'] == 1 else np.nan, axis=1)
        imputed_tac_list, not_imputable = impute(df_copy, tac_removed_device_off.tolist(), 'Duration_Hrs', {},
                                                 override_index_check_count=True, extend_missing_idx=15,
                                                 gap_proportional_limit=0.9, how='flex')

        imputed_tac_list = [tac if tac > 0 else 0 for tac in imputed_tac_list]
        df['TAC_device_off'] = imputed_tac_list
        df['TAC_device_off_imputed'] = [1 if df.loc[i, 'TAC_device_off'] != df.loc[i, 'TAC'] else 0 for i in
                                        range(0, len(df))]
        df['TAC'] = df['TAC_device_off']

    def export_workbook(self):
        export_skyn_workbook(self.dataset, self.unprocessed_dataset, self.stats, self.subid, self.condition,
                             self.dataset_identifier, self.predictions, self.data_out_folder, self.simple_plot_paths,
                             self.complex_plot_paths)

    def set_curve_threshold(self, dataset_version='CLN', validate=False):
        #simply saving the baseline range to stats
        self.stats[f'baseline_range_{dataset_version}'] = self.baseline_range

        tac_variable = f'TAC_processed_smooth_{self.smoothing_window}' if dataset_version == 'CLN' else 'TAC_Raw'

        curve_threshold = get_curve_threshold(self.dataset, tac_variable, self.baseline_range)
        peak = get_peak(self.dataset, tac_variable)
        floor = self.dataset[tac_variable].min()
        relative_peak = peak - curve_threshold
        tac_range = peak - floor

        backup_dataset = self.dataset.copy()

        if tac_range < 2:
            curve_threshold = curve_threshold = (peak + floor) / 2
        elif (dataset_version == 'CLN') and validate:
            curve_threshold_verified = False
            drop_count = 0
            while not curve_threshold_verified:
                df = self.dataset.iloc[drop_count:]
                df.reset_index(inplace=True)
                curve_threshold = get_curve_threshold(df, tac_variable, self.baseline_range)
                peak = get_peak(df, tac_variable)
                floor = df[tac_variable].min()
                relative_peak = peak - curve_threshold
                tac_range = peak - floor
                peak_to_floor_required_ratio = (1 / 7) * (tac_range - 0.5) + 1
                # print('CT: ', curve_threshold)
                # print('Peak: ', peak)
                # print('Floor: ', floor)
                # print('must be below: ', tac_range / peak_to_floor_required_ratio)
                # print('ratio: ', peak_to_floor_required_ratio)
                # print('Pass: ', (curve_threshold < tac_range / peak_to_floor_required_ratio))
                # print('Relative Peak: ', relative_peak)
                # print('tac_range: ', tac_range)
                # if curve threshold is (about) half way between the floor and peak
                #or if relative peak < 1
                if (curve_threshold < (tac_range / peak_to_floor_required_ratio) + floor) or (tac_range < 3) or (
                        tac_range < 5 and relative_peak < 1):

                    if curve_threshold > peak or curve_threshold < floor:
                        curve_threshold = curve_threshold = floor + (tac_range * 0.8)

                    if (relative_peak < 8) and peak in self.dataset.loc[:self.baseline_range + 1,
                                                       tac_variable].tolist():
                        curve_threshold = (tac_range / peak_to_floor_required_ratio) + floor
                    # elif (relative_peak < 5) and peak not in self.dataset.loc[:self.baseline_range+1, tac_variable].tolist():
                    #   curve_threshold = (tac_range / 4) + floor

                    self.dataset = self.dataset.iloc[drop_count:].reset_index(drop=True)
                    self.dataset['Duration_Hrs'] = self.dataset['Duration_Hrs'] - self.dataset['Duration_Hrs'].min()
                    curve_threshold_verified = True
                    break
                else:
                    drop_count += 1
                    if drop_count == len(self.dataset) - 1 - self.baseline_range:
                        self.dataset = self.dataset.iloc[drop_count:].reset_index(drop=True)
                        self.dataset['Duration_Hrs'] = self.dataset['Duration_Hrs'] - self.dataset['Duration_Hrs'].min()
                        break

            if self.dataset['Duration_Hrs'].max() < 1 or (drop_count * self.sampling_rate) > (len(self.dataset) / 3):
                self.dataset = backup_dataset
                curve_threshold = get_curve_threshold(self.dataset, tac_variable, self.baseline_range)
                peak = get_peak(self.dataset, tac_variable)
                floor = self.dataset[tac_variable].min()
                curve_threshold = (peak + floor) / 2
                # self.valid_occasion = 0
                # self.invalid_reason = f'Unable to ascertain valid curve threshold. Curve threshold is established by the first {self.baseline_range} values, and must be half way (or lower) between peak TAC and floor TAC. SDM will attempt to trim the dataset until curve threshold is valid, but if too much data is trimmed, then dataset is considered invalid.'

        else:
            curve_threshold = get_curve_threshold(self.dataset, tac_variable, self.baseline_range)

        self.stats[f'curve_threshold_{dataset_version}'] = curve_threshold

    def get_episode_features(self, dataset_version='RAW'):

        df = self.dataset
        tac_variable = f'TAC_processed_smooth_{self.smoothing_window}' if dataset_version == 'CLN' else 'TAC_Raw'

        self.stats[f'TAC_N_{dataset_version}'] = len(df)
        self.stats[f'baseline_mean_{dataset_version}'], self.stats[
            f'baseline_stdev_{dataset_version}'] = get_baseline_mean_stdev(df, tac_variable)
        self.stats[f'mean_{dataset_version}'], self.stats[f'stdev_{dataset_version}'], self.stats[
            f'sem_{dataset_version}'] = get_mean_stdev_sem(df, tac_variable)
        self.stats[f'auc_total_{dataset_version}'], self.stats[f'auc_per_hour_{dataset_version}'] = get_auc(df,
                                                                                                            tac_variable,
                                                                                                            self.time_elapsed_column)
        self.stats[f'peak_{dataset_version}'] = get_peak(df, tac_variable)
        peak_index = get_peak_index(df, tac_variable)
        self.stats[f'relative_peak_{dataset_version}'] = self.stats[f'peak_{dataset_version}'] - self.stats[
            f'curve_threshold_{dataset_version}']
        self.stats[f'rise_duration_{dataset_version}'], curve_begins_index, self.stats[
            'curve_start_time'] = get_rise_duration(df, tac_variable, self.time_elapsed_column, peak_index,
                                                    self.stats[f'curve_threshold_{dataset_version}'])
        self.stats[f'fall_duration_{dataset_version}'], curve_ends_index, curve_fall_threshold = get_fall_duration(df,
                                                                                                                   tac_variable,
                                                                                                                   self.time_elapsed_column,
                                                                                                                   peak_index,
                                                                                                                   self.stats[
                                                                                                                       f'curve_threshold_{dataset_version}'])
        self.stats[f'rise_rate_{dataset_version}'] = get_rise_rate(self.stats[f'rise_duration_{dataset_version}'],
                                                                   self.stats[f'relative_peak_{dataset_version}'])
        self.stats[f'fall_rate_{dataset_version}'] = get_fall_rate(self.stats[f'fall_duration_{dataset_version}'],
                                                                   self.stats[f'relative_peak_{dataset_version}'])
        self.stats[f'duration_{dataset_version}'] = df[self.time_elapsed_column].max()
        self.stats[f'curve_duration_{dataset_version}'] = get_curve_duration(
            self.stats[f'rise_duration_{dataset_version}'], self.stats[f'fall_duration_{dataset_version}'])
        self.stats[f'curve_auc_{dataset_version}'] = get_curve_auc(df, tac_variable, curve_begins_index,
                                                                   curve_ends_index,
                                                                   self.stats[f'curve_threshold_{dataset_version}'])
        self.stats[f'curve_auc_per_hour_{dataset_version}'] = get_curve_auc_per_hour(
            self.stats[f'curve_auc_{dataset_version}'], self.stats[f'curve_duration_{dataset_version}'])

        #MOTION
        self.stats[f'no_motion_{dataset_version}'] = get_value_proportion(df, 0, 'Motion')
        self.stats[f'mean_motion_{dataset_version}'], self.stats[f'stdev_motion_{dataset_version}'], self.stats[
            f'sem_motion_{dataset_version}'] = get_mean_stdev_sem(df, 'Motion')

        #TEMPERATURE
        self.stats[f'mean_temp_{dataset_version}'], self.stats[f'stdev_temp_{dataset_version}'], self.stats[
            f'sem_temp_{dataset_version}'] = get_mean_stdev_sem(df, 'Temperature_C')

        #SIGNAL VARIABILITY
        self.stats[f'avg_tac_diff_{dataset_version}'], differences = get_avg_tac_diff(df, tac_variable)
        self.stats[f'tac_alt_perc_{dataset_version}'] = get_tac_directional_alteration_percent(df, tac_variable)
        self.stats[f'curve_alterations_{dataset_version}'], total_n = get_tac_directional_alterations(
            df.loc[curve_begins_index:curve_ends_index], tac_variable)
        self.stats[f'completed_curve_count_{dataset_version}'] = get_discrete_curve_count(df, tac_variable,
                                                                                          self.sampling_rate,
                                                                                          self.stats[
                                                                                              f'curve_threshold_{dataset_version}'],
                                                                                          min_curve_duration_hours=0.5)

        self.stats[f'curve_begins_index_{dataset_version}'] = curve_begins_index
        self.stats[f'curve_ends_index_{dataset_version}'] = curve_ends_index

        if dataset_version == 'CLN':
            #OUTLIERS AND CLEANING
            self.stats[f'subzero_reassigned_zero_count'] = len(df[df['negative_reassigned_zero'] == 0])
            self.stats[f'device_off_N'] = len(df[df['device_on_pred'] == 0])
            self.stats[f'extreme_outlier_N'] = len(df[df['extreme_outlier'] == 1])
            self.stats[f'major_outlier_N'] = len(df[df['major_outlier'] == 1])
            self.stats[f'minor_outlier_N'] = len(df[df['minor_outlier'] == 1])

            #proportion of how much of the tac falls towards baseline
            self.stats['fall_completion'] = get_fall_completion(self.dataset, tac_variable, curve_ends_index,
                                                                self.stats['relative_peak_CLN'], curve_fall_threshold)
            self.rise_completion = get_rise_completion(self.dataset, tac_variable, curve_begins_index,
                                                       self.stats['relative_peak_CLN'],
                                                       self.stats['curve_threshold_CLN'])

            #Reassign null stats to their raw stat; if raw stat is also null, the dataset is invalid
            for variable in ['auc_total', 'auc_per_hour', 'curve_auc', 'curve_auc_per_hour', 'avg_tac_diff']:
                if not self.stats[f'{variable}_CLN']:
                    if self.stats[f'{variable}_RAW']:
                        self.stats[f'{variable}_CLN'] = self.stats[f'{variable}_RAW']
                    else:
                        self.valid_occasion = 0
                        self.invalid_reason = 'feature engineering failed'

    def get_row_features(self):

        labels_off_on = []
        temp = []
        motion = []

        if len(list(self.event_timestamps.keys())) > 0:
            self.label_device_removed_with_timestamps()
        else:
            self.dataset['device_on'] = ['unk' for i in range(0, len(self.dataset))]

        for i, row in self.dataset.iterrows():
            temp.append(skynDatapoint('Temperature_C', 'datetime', i, self.dataset, 10, self.sampling_rate,
                                      label=row['device_on'], use_after=True, use_before=True))
            motion.append(
                skynDatapoint('Motion', 'datetime', i, self.dataset, 10, self.sampling_rate, label=row['device_on'],
                              use_after=True, use_before=True))
            labels_off_on.append(row['device_on'])

        self.dataset['device_on'] = labels_off_on
        self.dataset['temp'] = [t.value if hasattr(t, 'value') else np.nan for t in temp]
        self.dataset['temp_a_pre'] = [t.a_pre if hasattr(t, 'a_pre') else np.nan for t in temp]
        self.dataset['temp_b_pre'] = [t.b_pre if hasattr(t, 'b_pre') else np.nan for t in temp]
        self.dataset['temp_c_pre'] = [t.c_pre if hasattr(t, 'c_pre') else np.nan for t in temp]
        self.dataset['temp_a_post'] = [t.a_post if hasattr(t, 'a_post') else np.nan for t in temp]
        self.dataset['temp_b_post'] = [t.b_post if hasattr(t, 'b_post') else np.nan for t in temp]
        self.dataset['temp_c_post'] = [t.c_post if hasattr(t, 'c_post') else np.nan for t in temp]
        self.dataset['temp_mean_change_pre'] = [t.mean_change_before if hasattr(t, 'mean_change_before') else np.nan for
                                                t in temp]
        self.dataset['temp_mean_change_post'] = [t.mean_change_after if hasattr(t, 'mean_change_after') else np.nan for
                                                 t in temp]
        self.dataset['temp_change_pre'] = [t.difference_from_prior if hasattr(t, 'difference_from_prior') else np.nan
                                           for t in temp]
        self.dataset['temp_change_post'] = [t.difference_from_next if hasattr(t, 'difference_from_next') else np.nan for
                                            t in temp]
        self.dataset['motion'] = [m.value if hasattr(m, 'value') else np.nan for m in motion]
        self.dataset['motion_a_pre'] = [m.a_pre if hasattr(m, 'a_pre') else np.nan for m in motion]
        self.dataset['motion_b_pre'] = [m.b_pre if hasattr(m, 'b_pre') else np.nan for m in motion]
        self.dataset['motion_c_pre'] = [m.c_pre if hasattr(m, 'c_pre') else np.nan for m in motion]
        self.dataset['motion_a_post'] = [m.a_post if hasattr(m, 'a_post') else np.nan for m in motion]
        self.dataset['motion_b_post'] = [m.b_post if hasattr(m, 'b_post') else np.nan for m in motion]
        self.dataset['motion_c_post'] = [m.c_post if hasattr(m, 'c_post') else np.nan for m in motion]
        self.dataset['motion_mean_change_pre'] = [m.mean_change_before if hasattr(m, 'mean_change_before') else np.nan
                                                  for m in motion]
        self.dataset['motion_mean_change_post'] = [m.mean_change_after if hasattr(m, 'mean_change_after') else np.nan
                                                   for m in motion]
        self.dataset['motion_change_pre'] = [m.difference_from_prior if hasattr(m, 'difference_from_prior') else np.nan
                                             for m in motion]
        self.dataset['motion_change_post'] = [m.difference_from_next if hasattr(m, 'difference_from_next') else np.nan
                                              for m in motion]

    # MAKE PLOTS
    def plot_official_tac(self):
        plot_path = plot_smoothed_curve(self, self.dataset, self.stats[f'peak_CLN'], self.stats['curve_threshold_CLN'],
                                        self.stats['curve_begins_index_CLN'], self.stats['curve_ends_index_CLN'],
                                        event_timestamps=self.event_timestamps)
        self.complex_plot_paths.append(plot_path)
        self.official_curve_plot = plot_path

    def plot_column(self, variable_name):
        plot_path = plot_column(self.dataset, self.plot_folder, self.subid, self.condition, self.dataset_identifier,
                                variable_name, self.time_elapsed_column, event_timestamps=self.event_timestamps)
        self.simple_plot_paths.append(plot_path)
        return plot_path

    def plot_device_removal(self):
        plot_path = None
        if 'Temp' in self.device_removal_detection_method:
            plot_path = plot_device_removal(self.dataset, self.plot_folder, self.subid, self.condition,
                                            self.dataset_identifier, 'Temperature_C', self.time_elapsed_column,
                                            event_timestamps=self.event_timestamps, motion_variable='Motion',
                                            add_color=True,
                                            temp_cutoff=int(self.device_removal_detection_method[-11:-10]))
            self.device_removal_temp_cutoff_plot = plot_path
        elif self.device_removal_detection_method == 'Built-in Algorithm':
            plot_path = plot_device_removal(self.dataset, self.plot_folder, self.subid, self.condition,
                                            self.dataset_identifier, 'Temperature_C', self.time_elapsed_column,
                                            event_timestamps=self.event_timestamps, motion_variable='Motion',
                                            prediction_column='device_on_pred', method='Model Predictions',
                                            add_color=True)
            self.device_removal_model_preds_plot = plot_path
        elif self.device_removal_detection_method == 'ground_truth':
            plot_path = plot_device_removal(self.dataset, self.plot_folder, self.subid, self.condition,
                                            self.dataset_identifier, 'Temperature_C', self.time_elapsed_column,
                                            event_timestamps=self.event_timestamps, motion_variable='Motion',
                                            prediction_column='device_on', method='Ground Truth', add_color=True)
            self.device_removal_ground_truth_plot = plot_path
        if plot_path not in self.complex_plot_paths and plot_path != None:
            self.complex_plot_paths.append(plot_path)
        self.device_removal_plot = plot_path

    def plot_tac_and_temp(self):
        plot_path = plot_tac_and_temp(self.dataset, self.plot_folder, self.subid, self.condition,
                                      self.dataset_identifier, 'TAC', 'Temperature_C', self.time_elapsed_column,
                                      event_timestamps=self.event_timestamps)
        self.complex_plot_paths.append(plot_path)
        self.tac_and_temp_plot = plot_path

    def plot_cleaning_comparison(self):
        plot_path = plot_cleaning_comparison(self, self.dataset, self.time_elapsed_column,
                                             event_timestamps=self.event_timestamps, add_color=True)
        self.complex_plot_paths.append(plot_path)
        self.cleaning_comparison_plot = plot_path

    def plot_TAC_curves(self, additional_groupings={}):
        dataset_columns = self.dataset.columns.tolist()
        for variable_name in [col for col in self.dataset.columns.tolist() if 'TAC' in col]:
            plot_path = plot_TAC_curve(self.dataset,
                                       self.plot_folder, self.subid, self.condition, self.dataset_identifier,
                                       variable_name, self.time_elapsed_column, event_timestamps=self.event_timestamps)
            self.complex_plot_paths.append(plot_path)

        groupings = {
            'cleaned_smooth_comparison': [col for col in dataset_columns if 'cleaned_smooth' in col]
        }
        groupings.update(additional_groupings)

        for group_name, variable_grouping in groupings.items():
            if len(variable_grouping) > 0:
                plot_path = plot_overlaid_TAC_curves(self.dataset, self.plot_folder, self.subid, self.condition,
                                                     self.dataset_identifier, variable_grouping,
                                                     self.time_elapsed_column, group_name,
                                                     event_timestamps=self.event_timestamps)
                self.plot_paths.append(plot_path)

    def label_device_removed_with_timestamps(self):
        labels = []
        prior_timestamp_index = 0
        for timestamp_label, timestamp in self.event_timestamps.items():
            timestamp_index = get_closest_index_with_timestamp(self.dataset, timestamp, 'datetime')
            if timestamp_index:
                if "off" in timestamp_label.lower():  #if timestamp indicates device was taken off, then prior data corresponds to device on
                    labels.extend([1 for i in range(prior_timestamp_index, timestamp_index)])
                elif "on" in timestamp_label.lower():
                    labels.extend([0 for i in range(prior_timestamp_index, timestamp_index)])
                else:
                    labels.extend(['unk' for i in range(prior_timestamp_index, timestamp_index)])
                prior_timestamp_index = timestamp_index

                if timestamp_label == list(self.event_timestamps.keys())[len(list(self.event_timestamps.keys())) - 1]:
                    if "off" in timestamp_label.lower():  #if timestamp indicates device was taken off, then prior data corresponds to device on
                        labels.extend([0 for i in range(timestamp_index, len(self.dataset))])
                    elif "on" in timestamp_label.lower():
                        labels.extend([1 for i in range(timestamp_index, len(self.dataset))])
                    else:
                        labels.extend(['unk' for i in range(timestamp_index, len(self.dataset))])

        try:
            self.dataset['device_on'] = labels
        except:
            self.dataset['device_on'] = 'unk'

    def make_prediction(self, models, export=False):
        predictions = {}
        for model in models:
            try:
                data = pd.DataFrame([self.stats[feature] for feature in model.predictors], index=model.predictors).T
                prediction = model.predict(data)
                if model.outcome == 'condition':
                    predictions[model.model_name + '_prediction'] = 'Alc' if prediction == 1 else 'Non'
                    if self.condition != 'Unk':
                        predictions[model.model_name + '_self_report_or_ground_truth_confirmed'] = 'yes (correct)' if \
                            predictions[model.model_name + '_prediction'] == self.condition else 'no (incorrect)'
                    else:
                        predictions[
                            model.model_name + '_self_report_or_ground_truth_confirmed'] = 'Cannot confirm - no self-report or ground truth provided for condition'
                elif model.outcome == 'binge':
                    predictions[model.model_name + '_prediction'] = 'Heavy' if prediction == 1 else 'Light'
                    if self.binge != "None" and self.binge != "Unk":
                        predictions[model.model_name + '_self_report_or_ground_truth_confirmed'] = 'yes (correct)' if \
                            predictions[model.model_name + '_prediction'] == self.binge else 'no (incorrect)'
                    elif self.binge == 'None':
                        predictions[
                            model.model_name + '_self_report_or_ground_truth_confirmed'] = 'Cannot confirm - Binge prediction may not be relevant. Self-report/ground truth is non-alcohol episode.'
                    else:
                        predictions[
                            model.model_name + '_self_report_or_ground_truth_confirmed'] = 'Cannot confirm - no self-report or ground truth determined for binge.'
                else:
                    predictions[model.model_name] = prediction
                    predictions[
                        model.model_name + '_self_report_or_ground_truth_confirmed'] = 'Cannot confirm - model is exploratory and has not been implemented into SDM.'

            except Exception:
                print(traceback.format_exc())
                return traceback.format_exc()

        self.predictions = pd.DataFrame([predictions])

    def get_metadata_index(self):
        print(self.subid)
        print(self.episode_identifier)
        print(self.dataset_identifier)
        try:
            filtered_metadata = self.metadata[(self.metadata['SubID'] == self.subid) & (
                    self.metadata['Episode_Identifier'] == int(self.episode_identifier[1:])) &
                                              (self.metadata['Dataset_Identifier'] == int(
                                                  self.dataset_identifier))]
            return filtered_metadata.index.tolist()[0]
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error getting metadata index: {e}")
            return None

    def load_metadata(self, column: str = 'TotalDrks') -> Optional[Any]:
        try:
            return self.metadata.loc[self.metadata_index, column]
        except (KeyError, IndexError):
            return None

    def is_binge(self) -> str:
        if self.drinks is None or self.sex is None:
            return "Unk"
        if self.drinks == 0:
            return "None"
        if self.sex == 1 and self.drinks >= 5:
            return "Heavy"
        if self.sex == 2 and self.drinks >= 4:
            return "Heavy"
        if self.drinks > 0:
            return "Light"
        return "Unk"

    # TODO: go through and use Paths not strings
    def create_output_folders(self):
        if not os.path.exists(self.data_out_folder):
            os.mkdir(self.data_out_folder)
        if not os.path.exists(self.plot_folder):
            os.mkdir(self.plot_folder)
        subid_plot_folder = f'{self.plot_folder}/{self.subid}/'
        if not os.path.exists(subid_plot_folder):
            os.mkdir(subid_plot_folder)
        full_plot_folder = (f'{self.plot_folder}/{self.subid}/{self.dataset_identifier}'
                            f'{self.condition if self.condition else ""}/')
        if not os.path.exists(full_plot_folder):
            os.mkdir(full_plot_folder)
        self.plot_folder = full_plot_folder

    # TODO: save as paths and use .extensions and stuff!!
    def load_dataset(self):
        if self.path[-3:] == 'csv':
            return pd.read_csv(self.path, index_col=False)
        else:
            return pd.read_excel(self.path, index_col=False)

    def configure_raw_data(self):
        print(f'initializing \n{self.subid} \n{self.condition} {self.dataset_identifier}')
        print(self.path)
        self.unprocessed_dataset['SubID'] = self.subid
        self.unprocessed_dataset['Dataset_Identifier'] = self.dataset_identifier
        self.unprocessed_dataset['Episode_Identifier'] = self.episode_identifier
        self.unprocessed_dataset['Full_Identifier'] = get_full_identifier(self.subid, self.dataset_identifier,
                                                                          self.episode_identifier)

        df_raw = update_column_names(self.unprocessed_dataset)
        df_raw = rename_tac_column(df_raw)

        try:
            df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], unit='s')
        except (ValueError, TypeError) as e:
            print(f"Error configuring raw data: {e}")
            df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])

        df_raw = df_raw.sort_values(by="datetime", ignore_index=True)
        df_raw.reset_index(inplace=True, drop=True)
        df_raw['Row_ID'] = df_raw['Full_Identifier'].astype(str) + '_' + df_raw.index.astype(str)

        df_raw = df_raw[['SubID', 'Dataset_Identifier', 'Episode_Identifier', 'Full_Identifier', 'Row_ID'] +
                        [col for col in df_raw.columns.tolist() if col not in ['SubID', 'Dataset_Identifier',
                                                                               'Episode_Identifier', 'Full_Identifier',
                                                                               'Row_ID']]]

        sampling_rate = get_sampling_rate(df_raw, 'datetime')
        if sampling_rate > 1:
            df_raw = reduce_sampling_rate(df_raw, 'datetime')

        df_raw = get_time_elapsed(df_raw, 'datetime')

        df_raw = remove_junk_columns(df_raw)

        return df_raw

    # TODO: figure out what the hell this does
    def timestamp_available(self, begin_or_end='Begin'):
        filter_ = (self.metadata['SubID'] == self.subid) & (
                (self.metadata['Dataset_Identifier'] == str(self.dataset_identifier)) | (
                self.metadata['Dataset_Identifier'] == int(self.dataset_identifier))) & (
                          self.metadata['Episode_Identifier'] == int(self.episode_identifier[1:]))
        return (self.metadata.loc[filter_, f'Crop {begin_or_end} Date'].notnull().any() and
                self.metadata.loc[filter_, f'Crop {begin_or_end} Time'].notnull().any())

    def get_event_timestamps(self, metadata_path: str) -> dict[str, Any]:
        try:
            timestamps = pd.read_excel(metadata_path, sheet_name="Additional_Timestamps")
            # TODO: why > 5? why not all? rewrite
            timestamp_columns = [col for i, col in enumerate(timestamps.columns) if i > 5]

            timestamps[timestamp_columns] = timestamps[timestamp_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M',
                                                                                errors='coerce')

            filtered_timestamps = timestamps[
                (timestamps['SubID'] == self.subid) &
                (timestamps['Dataset_Identifier'] == self.dataset_identifier) &
                (timestamps['Episode_Identifier'] == int(self.episode_identifier[1:]))
                ]

            event_timestamps = {
                col: filtered_timestamps.loc[filtered_timestamps.index[0], col]
                for col in filtered_timestamps.select_dtypes(include='datetime64').columns
            }

            return {key: value for key, value in event_timestamps.items() if value is not NaT}

        except (FileNotFoundError, ValueError, KeyError, IndexError, TypeError) as e:
            print(f"An error occurred while processing the timestamps: {e}")
            return {}

    def determine_initial_validity(self) -> tuple[int, Optional[str]]:
        """Checks metadata and device to see if event should be checked
        """
        # TODO: does this need to return anything? It seems like it's only setting class attributes which should be read
        if self.valid_occasion and self.metadata_index is None:
            self.valid_occasion = 0
            self.invalid_reason = 'Not in metadata'

        use_data = self.load_metadata('Use_Data')
        if use_data == 'N':
            self.valid_occasion = 0
            self.invalid_reason = f'Excluded by metadata: {self.metadata_note}'
            return self.valid_occasion, self.invalid_reason
        if self.disabled_by_multiple_device_ids:
            self.valid_occasion = 0
            self.invalid_reason = 'Multiple devices detected within dataset'
            return self.valid_occasion, self.invalid_reason

        self.valid_occasion = 1
        self.invalid_reason = None
        return self.valid_occasion, self.invalid_reason

    def calculate_device_inactivity_stats(self) -> None:
        device_off_or_removed = self.get_device_off_or_removed()
        self.stats['device_inactive_perc'] = (device_off_or_removed.sum() / len(self.dataset)) * 100
        self.stats['device_inactive_duration'] = (device_off_or_removed.sum() * self.sampling_rate) / 60
        self.stats['device_active_perc'] = 100 - self.stats['device_inactive_perc']
        self.stats['device_active_duration'] = (
                                                       self.dataset['Duration_Hrs'].max() -
                                                       self.dataset['Duration_Hrs'].min()
                                               ) - (device_off_or_removed.sum() * self.sampling_rate) / 60

    def calculate_imputation_stats(self, all_imputations: pd.Series) -> None:
        self.stats['imputed_N'] = all_imputations.sum()
        self.stats['tac_imputed_duration'] = (all_imputations.sum() * self.sampling_rate) / 60
        self.stats['tac_imputed_perc'] = (all_imputations.sum() / len(self.dataset)) * 100

    def get_device_off_or_removed(self) -> pd.Series:
        return (self.dataset['gap_imputed'] == 1) | (self.dataset['TAC_device_off_imputed'] == 1)

    def get_all_imputations(self) -> pd.Series:
        device_off_or_removed = self.get_device_off_or_removed()
        return (
                device_off_or_removed |
                (self.dataset['major_outlier'] == 1) |
                (self.dataset['minor_outlier'] == 1) |
                (self.dataset['sloped_start'] == 1) |
                (self.dataset['extreme_outlier'] == 1)
        )

    def validate_device_usage(self) -> tuple[int, Optional[str]]:
        device_off_or_removed = self.get_device_off_or_removed()
        device_active_duration = ((len(self.dataset) - device_off_or_removed.sum()) * self.sampling_rate) / 60
        enough_device_on = (
                self.stats['device_inactive_perc'] < self.max_percentage_inactive
                and device_active_duration > self.min_duration_active
        )

        if not enough_device_on:
            self.valid_occasion = 0
            self.invalid_reason = (
                f'Duration of device inactivity (device non-wear, device off) is too great. Max allowed percentage of '
                f'inactivity ({self.max_percentage_inactive}%) was exceeded or the minimum required duration of '
                f'{self.min_duration_active} hours was not met. Device inactive for '
                f'{round(self.stats["device_inactive_perc"], 1)}% and there is '
                f'{round(device_active_duration, 1)} hours of valid data.'
            )
            return self.valid_occasion, self.invalid_reason

        return 1, None

    def validate_imputation(self) -> tuple[int, Optional[str]]:
        """Determine if the device data is valid based on the imputation percentage."""
        too_many_imputations = self.stats['tac_imputed_perc'] > self.max_percentage_imputed

        if too_many_imputations:
            self.valid_occasion = 0
            self.invalid_reason = (
                  f'Device contains too many artifacts or noisy data. Specifically, '
                  f'{round(self.stats["tac_imputed_perc"], 1)}% of the data was imputed, exceeding the limit of '
                  f'{self.max_percentage_imputed}%.'
              )
            return self.valid_occasion, self.invalid_reason

        return 1, None

    def determine_post_cleaning_validity(self) -> tuple[int, Optional[str]]:
        all_imputations = self.get_all_imputations()

        self.calculate_device_inactivity_stats()
        self.calculate_imputation_stats(all_imputations)

        valid_occasion, invalid_reason = self.validate_device_usage()
        if not valid_occasion:
            return valid_occasion, invalid_reason

        valid_occasion, invalid_reason = self.validate_imputation()
        return valid_occasion, invalid_reason
