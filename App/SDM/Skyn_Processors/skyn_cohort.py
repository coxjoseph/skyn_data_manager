from SDM.Configuration.file_management import save_to_computer
from SDM.Machine_Learning.binary_model_dev import train_and_test_model_with_holdout, cv_group_k_fold
from SDM.Machine_Learning.pca import pca_with_features
from SDM.Visualization.plotting import plot_rf_feature_importances, plot_box_whisker
from skyn_cohort_tester import SkynCohortTester
from skyn_dataset import SkynDataset
from SDM.Configuration.configuration import configure_timestamps, get_cohort_full_identifiers
from SDM.Documenting.cohort_workbook import SDMReport
from SDM.Machine_Learning.feature_estimator import train_feature_estimator
from SDM.User_Interface.Utils.filename_tools import create_metadata_from_cohort_folder, extract_subid, \
    extract_dataset_identifier
import glob
import pandas as pd
import fnmatch
import os
import numpy as np


class SkynCohort:
    """
    TODO: match the cut comments from the init function to here for doc
    path to datasets to be analyzed
    path to find metadata
    give a name to this collection  of datasets
    folder to save processed data
    folder to save graphs
    folder to save features and model results
    signal processing customization
    model customization
    """

    def __init__(self,
                 data_folder: str,
                 metadata_path: str = None,
                 cohort_name: str = None,
                 merge_variables=None,
                 disable_crop_start=True,
                 disable_crop_end=True,
                 data_out_folder=None,
                 graphs_out_folder=None,
                 analyses_out_folder=None,
                 max_dataset_duration=18,
                 skyn_upload_timezone='CST',
                 predictors=None
                 ):

        if predictors is None:
            predictors = ['rise_rate_CLN', 'rise_duration_CLN', 'fall_rate_CLN', 'fall_duration_CLN', 'peak_CLN',
                          'relative_peak_CLN', 'completed_curve_count_CLN', 'curve_alterations_CLN', 'auc_total_CLN',
                          'major_outlier_N', 'minor_outlier_N', 'rise_rate_RAW', 'rise_duration_RAW',
                          'fall_rate_RAW', 'fall_duration_RAW', 'completed_curve_count_RAW', 'curve_alterations_RAW',
                          'auc_total_RAW', 'avg_tac_diff_RAW', 'tac_alt_perc_RAW', 'fall_completion',
                          'curve_start_time']
        if merge_variables is None:
            merge_variables = {}

        self.data_folder = data_folder
        self.cohort_name = cohort_name
        self.metadata_path = metadata_path
        if self.metadata_path == '':
            self.metadata = configure_timestamps(pd.DataFrame(create_metadata_from_cohort_folder(data_folder)))
        else:
            self.metadata = configure_timestamps(pd.read_excel(self.metadata_path))
        self.cohort_identifiers = get_cohort_full_identifiers(self.metadata)
        self.duplicate_identifiers = len(self.cohort_identifiers) != len(set(self.cohort_identifiers))
        self.merge_variables = merge_variables
        self.disable_crop_start = disable_crop_start
        self.disable_crop_end = disable_crop_end
        self.occasion_paths = glob.glob(f'{data_folder}*')
        self.sub_ids = []
        self.occasions = []
        self.invalid_occasions = []

        # export settings
        self.data_out_folder = data_out_folder
        self.graphs_out_folder = graphs_out_folder
        self.analyses_out_folder = analyses_out_folder
        self.model_figures_folder = self.analyses_out_folder + '/Figures'
        if not os.path.exists(self.model_figures_folder):
            os.mkdir(self.model_figures_folder)
        self.python_object_folder = self.analyses_out_folder + '/Python_Objects'
        self.default_filename = f'SDM_report_{self.cohort_name}'
        self.exported_reports = fnmatch.filter(os.listdir(self.analyses_out_folder), f"{self.default_filename}*")
        self.filename = f'{self.default_filename}_{len(self.exported_reports)}.xlsx' if len(
            self.exported_reports) else f'{self.default_filename}.xlsx'

        # signal processing customization
        self.smoothing_window = 51
        self.major_cleaning_threshold = 0.8
        self.minor_cleaning_threshold = 0.5
        self.device_removal_detection_method = None
        self.crop_begin_adjustment = -0.75
        self.crop_end_adjustment = 0

        # cropping customization
        self.max_dataset_duration = max_dataset_duration
        self.skyn_upload_timezone = skyn_upload_timezone

        # feature selection
        self.predictors = predictors
        self.prediction_features = predictors
        self.training_features = predictors

        # quality control
        self.min_duration_active = 1  # hours
        self.max_percentage_inactive = 30
        self.max_percentage_imputed = 65

        # results
        self.master_dataset = pd.DataFrame()
        self.invalid_features = pd.DataFrame()
        self.features = pd.DataFrame()
        self.PCA = pd.DataFrame()
        self.models = []
        self.default_model_accuracies = {}
        self.device_removal_models = {
            'RF': None,
            'LR': None
        }
        self.feature_estimators = {}

    def process_cohort(self, export_python_object=True):
        self.master_dataset = pd.DataFrame()
        if self.duplicate_identifiers:
            print('Error in metadata: Duplicate Identifiers')
        elif len(self.occasion_paths) == 0:
            print(f'No Skyn files detected in {self.data_folder}')
        else:
            self.occasions = []
            for path in self.occasion_paths:
                subid = extract_subid(os.path.basename(path))
                dataset_identifier = extract_dataset_identifier(os.path.basename(path))
                matching_episode_ids = self.metadata[(self.metadata['SubID'] == int(subid)) & (
                        self.metadata['Dataset_Identifier'] == int(dataset_identifier))][
                    'Episode_Identifier'].tolist()
                for episode_identifier in matching_episode_ids:
                    occasion = SkynDataset(path, self.data_out_folder, self.graphs_out_folder, int(subid),
                                           int(dataset_identifier), ('e' + str(episode_identifier)),
                                           self.disable_crop_start, self.disable_crop_end, self.skyn_upload_timezone,
                                           metadata_path=self.metadata_path, metadata=self.metadata)
                    occasion.max_duration = self.max_dataset_duration
                    occasion.major_cleaning_threshold = self.major_cleaning_threshold
                    occasion.minor_cleaning_threshold = self.minor_cleaning_threshold
                    occasion.smoothing_window = self.smoothing_window
                    occasion.device_removal_detection_method = self.device_removal_detection_method
                    occasion.crop_begin_adjustment = self.crop_begin_adjustment
                    occasion.crop_end_adjustment = self.crop_end_adjustment
                    occasion.min_duration_active = self.min_duration_active
                    occasion.max_percentage_inactive = self.max_percentage_inactive
                    occasion.max_percentage_imputed = self.max_percentage_imputed
                    occasion.process_with_default_settings(make_plots=True)
                    self.occasions.append(occasion) if occasion.valid_occasion else self.invalid_occasions.append(
                        occasion)
                    if len(self.master_dataset) == 0 and len(occasion.dataset) > 0:
                        self.master_dataset = occasion.dataset
                    else:
                        self.master_dataset = pd.concat([self.master_dataset, occasion.dataset], ignore_index=True)

            self.load_features()

        if not os.path.exists(self.python_object_folder):
            os.mkdir(self.python_object_folder)
        if export_python_object:
            save_to_computer(self, self.cohort_name, self.python_object_folder)
        self.export_sdm_report()

    def export_sdm_report(self):
        sdm_report = SDMReport(self)
        sdm_report.run_export()

    def make_predictions(self, models, prediction_type='binary', save_processor=True):
        if prediction_type == 'binary':
            tester = SkynCohortTester(self, models)
            # TODO: this doesnt exist either??
            tester.make_binary_predictions(predictors=self.prediction_features)
            tester.create_feature_and_predictions_report()
            if save_processor:
                save_to_computer(self, self.cohort_name, self.python_object_folder)

    def train_models_using_episode_features(self, model_names, force_refresh=False, save_processor=True,
                                            save_models=True, export_report=True):
        for model_name in model_names:
            if model_name == 'RF_Alc_vs_Non' or model_name == 'LR_Alc_vs_Non':
                self.cross_validation(model_name, 'Alc_vs_Non')
                self.export_feature_plots(ground_truth_variable='condition')

            if model_name == 'RF_AUD' or model_name == 'LR_AUD':
                self.cross_validation(model_name, 'AUD_vs_Not')
                self.export_feature_plots(ground_truth_variable='AUD')

            if model_name == 'RF_Binge' or model_name == 'LR_Binge':
                self.cross_validation(model_name, 'Light_vs_Heavy')
                self.export_feature_plots(ground_truth_variable='binge', filter_conditions={'binge': ["None"]})

        self.load_features(force_refresh=force_refresh)

        if save_processor:
            save_to_computer(self, self.cohort_name, self.python_object_folder)

        if save_models:
            for model in self.models:
                save_to_computer(model, self.cohort_name + model.model_name, self.python_object_folder,
                                 extension='sdmtm')

        if export_report:
            self.export_sdm_report()

    def create_feature_estimators(self):
        fall_rate_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'rise_duration_CLN',
                                                                    'rise_rate_CLN'], outcome='fall_rate_CLN')
        self.feature_estimators['fall_rate'] = fall_rate_model

        fall_duration_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'rise_duration_CLN',
                                                                        'rise_rate_CLN'], outcome='fall_duration_CLN')
        self.feature_estimators['fall_duration'] = fall_duration_model

        rise_rate_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'fall_duration_CLN',
                                                                    'fall_rate_CLN'], outcome='rise_rate_CLN')
        self.feature_estimators['fall_rate'] = rise_rate_model

        rise_duration_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'fall_duration_CLN',
                                                                        'fall_rate_CLN'], outcome='rise_duration_CLN')
        self.feature_estimators['fall_duration'] = rise_duration_model

    def train_device_removal_model(self, exclude_beginning=10, exclude_ending=10, holdout=0.3,
                                   model_name='worn_vs_removed_RF'):

        grouped_df = self.master_dataset.groupby('Full_Identifier')
        first_10_indices = grouped_df.head(exclude_beginning).index
        last_10_indices = grouped_df.tail(exclude_ending).index
        indices_for_model_training = first_10_indices.union(last_10_indices)
        data_for_model_training = self.master_dataset.loc[~self.master_dataset.index.isin(indices_for_model_training)]
        self.master_dataset['device_removal_model'] = 'excluded'
        self.master_dataset.loc[
            ~self.master_dataset.index.isin(indices_for_model_training), 'device_removal_model'] = 'included'

        model = train_and_test_model_with_holdout(data_for_model_training,
                                                  ['temp', 'temp_a_pre', 'temp_b_pre', 'temp_c_pre', 'temp_a_post',
                                                   'temp_b_post', 'temp_c_post', 'temp_mean_change_pre',
                                                   'temp_mean_change_post', 'temp_change_pre', 'temp_change_post',
                                                   'motion', 'motion_a_pre', 'motion_b_pre', 'motion_c_pre',
                                                   'motion_a_post', 'motion_b_post', 'motion_c_post',
                                                   'motion_mean_change_pre', 'motion_mean_change_post',
                                                   'motion_change_pre', 'motion_change_post'],
                                                  holdout=holdout,
                                                  model_name=model_name)
        if holdout > 0:
            model.feature_importance = plot_rf_feature_importances(model, model_name, self.model_figures_folder)

            self.master_dataset = self.master_dataset.merge(model.cv_results['All_Features'], how='outer', on='Row_ID',
                                                            suffixes=('', '_right'))
            self.master_dataset = self.master_dataset.drop(
                columns=[col for col in self.master_dataset.columns if col.endswith('_right')])

            for occasion in self.occasions:
                occasion.dataset = self.master_dataset[
                    self.master_dataset['Full_Identifier'] == occasion.full_identifier]
                if occasion.full_identifier in model.cv_results['Training_Features']['Full_Identifier'].tolist():
                    occasion.device_removal_detection_method = 'ground_truth'
                else:
                    occasion.device_removal_detection_method = 'model'

                occasion.plot_device_removal()

        self.models.append(model)
        save_to_computer(self, self.cohort_name, self.python_object_folder)
        save_to_computer(model, model_name, self.python_object_folder, extension='sdmtm')

    # noinspection DuplicatedCode
    def load_features(self, force_refresh=True):
        features = {
            'subid': [occasion.subid for occasion in self.occasions],
            'condition': [occasion.condition for occasion in self.occasions],
            'dataset_identifier': [occasion.dataset_identifier for occasion in self.occasions],
            'episode_identifier': [occasion.episode_identifier for occasion in self.occasions],
            'valid_occasion': [occasion.valid_occasion for occasion in self.occasions],
            'device_start': [occasion.device_start_datetime for occasion in self.occasions],
            'crop_begin': [occasion.crop_begin for occasion in self.occasions],
            'crop_end': [occasion.crop_end for occasion in self.occasions],
            'drinks': [occasion.drinks for occasion in self.occasions],
            'binge': [occasion.binge for occasion in self.occasions],
            'AUD': [occasion.aud for occasion in self.occasions],
        }

        if force_refresh:
            self.features = pd.DataFrame()
            self.invalid_features = pd.DataFrame()

        stat_keys = []

        for occasion in self.occasions:
            for key, value in occasion.stats.items():
                if key not in stat_keys:
                    stat_keys.append(key)
                # TODO: refactor these
                refresh = (key not in self.features.columns.tolist()) or force_refresh
                if refresh:
                    if key not in features.keys():
                        features[key] = []
                        features[key].append(value)
                    else:
                        features[key].append(value)

        if len(self.features) > 0:
            self.features = self.features.merge(pd.DataFrame(features))
        else:
            self.features = pd.DataFrame(features)

        invalid_features = {
            'subid': [occasion.subid for occasion in self.invalid_occasions],
            'condition': [occasion.condition for occasion in self.invalid_occasions],
            'dataset_identifier': [occasion.dataset_identifier for occasion in self.invalid_occasions],
            'episode_identifier': [occasion.episode_identifier for occasion in self.invalid_occasions],
            'valid_occasion': [occasion.valid_occasion for occasion in self.invalid_occasions],
            'invalid_reason': [occasion.invalid_reason for occasion in self.invalid_occasions],
            'device_start': [occasion.device_start_datetime for occasion in self.invalid_occasions],
            'crop_begin': [occasion.crop_begin for occasion in self.invalid_occasions],
            'crop_end': [occasion.crop_end for occasion in self.invalid_occasions],
            'drinks': [occasion.drinks for occasion in self.invalid_occasions],
            'binge': [occasion.binge for occasion in self.invalid_occasions],
            'AUD': [occasion.aud for occasion in self.invalid_occasions]
        }

        for occasion in self.invalid_occasions:
            for key in stat_keys:
                if key not in list(occasion.stats.keys()):
                    occasion.stats[key] = ''
            for key, value in occasion.stats.items():
                refresh = (key not in self.invalid_features.columns.tolist()) or force_refresh
                if refresh:
                    if key not in invalid_features.keys():
                        invalid_features[key] = []
                        invalid_features[key].append(value)
                    else:
                        invalid_features[key].append(value)

        if len(self.invalid_features) > 0:
            self.invalid_features = self.invalid_features.merge(pd.DataFrame(invalid_features))
        else:
            self.invalid_features = pd.DataFrame(invalid_features)

    def export_feature_plots(self, ground_truth_variable='condition', filter_conditions=None) -> None:
        if filter_conditions is None:
            filter_conditions = {}
        plot_box_whisker(self.features, self.predictors, ground_truth_variable, self.model_figures_folder,
                         self.cohort_name, filter=filter_conditions)

    def principal_component_analysis(self):
        explained_variances = pca_with_features(self.features, self.predictors)
        self.PCA = explained_variances

    def cross_validation(self, model_name, model_design):
        print('Training ' + model_name + ' ' + model_design)
        cv_group_k_fold(model_design, model_name, self)

        if 'RF' in model_name:
            index_of_model = next((i for i, m in enumerate(self.models) if m.model_name == model_name), None)
            self.models[index_of_model].feature_importance = plot_rf_feature_importances(self.models[index_of_model],
                                                                                         model_name,
                                                                                         self.model_figures_folder)

    def make_predictions_using_default_models(self, models, export_report=True):
        filter_logic = None
        pred_labels = None
        ground_truth_variable = None
        for model in models:
            name = model.model_name
            if 'Alc_vs_Non' in name:
                ground_truth_variable = 'condition' if 'Unk' not in self.features['condition'] else None
                filter_logic = {'valid_occasion': [0]}
                pred_labels = {1: 'Alc', 0: 'Non'}
            elif 'Binge' in name:
                ground_truth_variable = 'binge' if 'Light' in self.features['binge'] or 'Heavy' in self.features[
                    'binge'] else None
                filter_logic = {
                    'valid_occasion': [0],
                }
                alc_vs_non_pred_column = 'RF_Alc_vs_Non_default_prediction' \
                    if 'RF_Alc_vs_Non_default_prediction' in self.features.columns \
                    else 'LR_Alc_vs_Non_default_prediction' \
                    if 'LR_Alc_vs_Non_default_prediction' in self.features.columns \
                    else ''
                if alc_vs_non_pred_column:
                    filter_logic[alc_vs_non_pred_column] = [0, 'Non']
                pred_labels = {
                    1: 'Heavy', 0: 'Light'
                }

            if 'default' not in model.model_name:
                model.model_name = model.model_name + '_default'
            self.make_binary_predictions(model, model.model_name, pred_labels=pred_labels, filter_logic=filter_logic,
                                         ground_truth_variable=ground_truth_variable)

        if export_report:
            self.export_sdm_report()

    # noinspection PyTypeChecker
    def make_binary_predictions(self, model, model_name, filter_logic=None,
                                pred_labels=None, ground_truth_variable='condition'):
        if pred_labels is None:
            pred_labels = {1: 'Alc', 0: 'Non'}
        if filter_logic is None:
            filter_logic = {'valid_occasion': [0]}
        excluded = None
        features = self.features.copy()

        for column, exclude_values in filter_logic.items():
            excluded = features[features[column].isin(exclude_values)]
            features = features[~features[column].isin(exclude_values)]

        predictions = model.predict(features[self.prediction_features])
        features[model_name + '_prediction'] = [pred_labels[x] for x in predictions]
        excluded[model_name + '_prediction'] = ['excluded' for _ in range(len(excluded))]

        if ground_truth_variable:
            features[model_name + '_correct'] = np.where(
                features[model_name + '_prediction'] == features[ground_truth_variable], 'correct', 'incorrect')
            excluded[model_name + '_correct'] = ['excluded' for _ in range(len(excluded))]

        features = pd.concat([features, excluded])
        features.sort_values(by=['subid', 'dataset_identifier', 'episode_identifier'], inplace=True)

        for i, occasion in enumerate(self.occasions):
            filter_logic = (features['subid'] == occasion.subid) & (
                    features['dataset_identifier'] == occasion.dataset_identifier) & (
                             features['episode_identifier'] == occasion.episode_identifier)
            if len(filter_logic):
                occasion.predictions[model_name] = features.loc[filter_logic, model_name + '_prediction'].tolist()[0]
        self.features = features
