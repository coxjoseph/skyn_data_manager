from .skyn_cohort_tester import skynCohortTester
from .skyn_dataset import SkynDataset
from ..Configuration.configuration import *
from ..Visualization.plotting import *
from ..Configuration.file_management import *
from ..Documenting.cohort_workbook import SDM_Report
from ..Machine_Learning.model_optimization import *
from ..Machine_Learning.pca import *
from ..Machine_Learning.feature_estimator import train_feature_estimator
from ..Feature_Engineering.tac_features import *
from ..Machine_Learning.binary_model_dev import *
from ..User_Interface.Utils.filename_tools import *
import glob
import pandas as pd
import fnmatch


class skynCohort:
  def __init__(self,
      #data in: tac data, metadata, redcap data, etc.
      data_folder, #path to datasets to be analyzed
      metadata_path = None, #path to find metadata
      cohort_name = None, #give a name to this collection  of datasets
      merge_variables = {}, 
      disable_crop_start = True, 
      disable_crop_end = True, 
      #export settings
      data_out_folder = None, #folder to save processed data
      graphs_out_folder = None, #folder to save graphs
      analyses_out_folder = None, #folder to save features and model results
      
      #signal processing customization
      max_dataset_duration = 18, 
      skyn_upload_timezone = 'CST',
      
      #model customization
      predictors = ['rise_rate_CLN', 'rise_duration_CLN', 'fall_rate_CLN', 'fall_duration_CLN', 'peak_CLN', 'relative_peak_CLN', 'completed_curve_count_CLN', 'curve_alterations_CLN', 'auc_total_CLN','major_outlier_N', 'minor_outlier_N', 'rise_rate_RAW', 'rise_duration_RAW', 'fall_rate_RAW', 'fall_duration_RAW', 'completed_curve_count_RAW', 'curve_alterations_RAW', 'auc_total_RAW','avg_tac_diff_RAW', 'tac_alt_perc_RAW', 'fall_completion', 'curve_start_time']
      ):
    
    #data in: tac data, metadata, redcap data, etc.
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
    self.subids = []
    self.occasions = []
    self.invalid_occasions = []

    #export settings
    self.data_out_folder = data_out_folder
    self.graphs_out_folder = graphs_out_folder
    self.analyses_out_folder = analyses_out_folder
    self.model_figures_folder = self.analyses_out_folder + '/Figures'
    if not os.path.exists(self.model_figures_folder):
      os.mkdir(self.model_figures_folder)
    self.python_object_folder = self.analyses_out_folder + '/Python_Objects'
    self.default_filename = f'SDM_report_{self.cohort_name}'
    self.exported_reports = fnmatch.filter(os.listdir(self.analyses_out_folder), f"{self.default_filename}*")
    self.filename = f'{self.default_filename}_{len(self.exported_reports)}.xlsx' if len(self.exported_reports) else f'{self.default_filename}.xlsx'

    #signal processing customization
    self.smoothing_window = 51
    self.major_cleaning_threshold = 0.8
    self.minor_cleaning_threshold = 0.5
    self.device_removal_detection_method = None
    self.crop_begin_adjustment = -0.75
    self.crop_end_adjustment = 0

    #cropping customization
    self.max_dataset_duration = max_dataset_duration
    self.skyn_upload_timezone = skyn_upload_timezone

    #feature selection
    self.predictors = predictors
    self.prediction_features = predictors
    self.training_features = predictors

    #quality control
    self.min_duration_active = 1 #hours
    self.max_percentage_inactive = 30
    self.max_percentage_imputed = 65

    #results
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
        matching_episode_ids = self.metadata[(self.metadata['SubID'] == int(subid)) & (self.metadata['Dataset_Identifier'] == int(dataset_identifier))]['Episode_Identifier'].tolist()
        for episode_identifier in matching_episode_ids:
          occasion = SkynDataset(path, self.data_out_folder, self.graphs_out_folder, int(subid), int(dataset_identifier), ('e' + str(episode_identifier)), self.disable_crop_start, self.disable_crop_end, self.skyn_upload_timezone, metadata_path = self.metadata_path, metadata = self.metadata)
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
          self.occasions.append(occasion) if occasion.valid_occasion else self.invalid_occasions.append(occasion)
          if len(self.master_dataset) == 0 and len(occasion.dataset) > 0:
            self.master_dataset = occasion.dataset
          else:
            self.master_dataset = pd.concat([self.master_dataset, occasion.dataset], ignore_index=True)

      self.load_features()
        
    if not os.path.exists(self.python_object_folder):
      os.mkdir(self.python_object_folder)
    if export_python_object:
        save_to_computer(self, self.cohort_name, self.python_object_folder)
    self.export_SDM_report()
  
  def export_SDM_report(self):
    sdm_report = SDM_Report(self)
    sdm_report.run_export()

  def make_predictions(self, models, prediction_type='binary', save_processor=True):
    if prediction_type == 'binary':
      tester = skynCohortTester(self, models)
      tester.make_binary_predictions(predictors=self.prediction_features)
      tester.create_feature_and_predictions_report()
      if save_processor:
          save_to_computer(self, self.cohort_name, self.python_object_folder)

  # def train_custom_model_using_episode_features(self, model_name, save_model=True, save_processor=True, refresh_stats=True):
  #   self.load_features(force_refresh=refresh_stats)
  #   self.cross_validation(model_name)

  def train_models_using_episode_features(self, model_names, force_refresh=False, save_processor=True, save_models=True, export_report=True):
    # self.load_features(force_refresh=force_refresh)

    for model_name in model_names:
      if model_name == 'RF_Alc_vs_Non'or model_name == 'LR_Alc_vs_Non':
          self.cross_validation(model_name, 'Alc_vs_Non')
          self.export_feature_plots(ground_truth_variable='condition')

      if model_name == 'RF_AUD' or model_name == 'LR_AUD':
        self.cross_validation(model_name, 'AUD_vs_Not')
        self.export_feature_plots(ground_truth_variable='AUD')

      if model_name == 'RF_Binge' or model_name == 'LR_Binge':
          self.cross_validation(model_name, 'Light_vs_Heavy')
          self.export_feature_plots(ground_truth_variable='binge', filter={'binge': ["None"]})
      
      #Custom Variable Creation Not READY

    self.load_features(force_refresh=force_refresh)

    if save_processor:
        save_to_computer(self, self.cohort_name, self.python_object_folder)

    if save_models:
      for model in self.models:
          save_to_computer(model, self.cohort_name + model.model_name, self.python_object_folder, extension='sdmtm')

    if export_report:
      self.export_SDM_report()

  def create_feature_estimators(self):
    fall_rate_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'rise_duration_CLN', 'rise_rate_CLN'], outcome='fall_rate_CLN')
    self.feature_estimators['fall_rate'] = fall_rate_model

    fall_duration_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'rise_duration_CLN', 'rise_rate_CLN'], outcome='fall_duration_CLN')
    self.feature_estimators['fall_duration'] = fall_duration_model

    rise_rate_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'fall_duration_CLN', 'fall_rate_CLN'], outcome='rise_rate_CLN')
    self.feature_estimators['fall_rate'] = rise_rate_model

    rise_duration_model = train_feature_estimator(self, predictors=['relative_peak_CLN', 'fall_duration_CLN', 'fall_rate_CLN'], outcome='rise_duration_CLN')
    self.feature_estimators['fall_duration'] = rise_duration_model

  def train_device_removal_model(self, exclude_beginning=10, exclude_ending=10, holdout=0.3, model_name = 'worn_vs_removed_RF'):
    
    grouped_df = self.master_dataset.groupby('Full_Identifier')
    first_10_indices = grouped_df.head(exclude_beginning).index
    last_10_indices = grouped_df.tail(exclude_ending).index
    indices_for_model_training = first_10_indices.union(last_10_indices)
    data_for_model_training = self.master_dataset.loc[~self.master_dataset.index.isin(indices_for_model_training)]
    self.master_dataset['device_removal_model'] = 'excluded'
    self.master_dataset.loc[~self.master_dataset.index.isin(indices_for_model_training), 'device_removal_model'] = 'included'

    model = train_and_test_model_with_holdout(data_for_model_training, ['temp','temp_a_pre','temp_b_pre','temp_c_pre','temp_a_post','temp_b_post','temp_c_post','temp_mean_change_pre','temp_mean_change_post','temp_change_pre','temp_change_post','motion','motion_a_pre','motion_b_pre','motion_c_pre','motion_a_post','motion_b_post','motion_c_post','motion_mean_change_pre','motion_mean_change_post','motion_change_pre','motion_change_post'],
                                              holdout=holdout,
                                              model_name=model_name)
    if holdout > 0:
      model.feature_importance = plot_rf_feature_importances(model, model_name, self.model_figures_folder)

      self.master_dataset = self.master_dataset.merge(model.cv_results['All_Features'], how='outer', on='Row_ID', suffixes=('', '_right'))
      self.master_dataset = self.master_dataset.drop(columns=[col for col in self.master_dataset.columns if col.endswith('_right')])

      for occasion in self.occasions:
        occasion.dataset = self.master_dataset[self.master_dataset['Full_Identifier']==occasion.full_identifier]
        if occasion.full_identifier in model.cv_results['Training_Features']['Full_Identifier'].tolist():
          occasion.device_removal_detection_method = 'ground_truth'
        else:
          occasion.device_removal_detection_method = 'model'

        occasion.plot_device_removal()

    self.models.append(model)
    save_to_computer(self, self.cohort_name, self.python_object_folder)
    save_to_computer(model, model_name, self.python_object_folder, extension='sdmtm')

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
        #statistical features will be added here...
      }
    
    if force_refresh:
      self.features = pd.DataFrame()
      self.invalid_features = pd.DataFrame()

    stat_keys = []

    for occasion in self.occasions:
      for key, value in occasion.stats.items():
        if key not in stat_keys:
          stat_keys.append(key)
        refresh = (key not in self.features.columns.tolist()) or (force_refresh)
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
        refresh = (key not in self.invalid_features.columns.tolist()) or (force_refresh)
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
    
  def export_feature_plots(self, ground_truth_variable='condition', filter={}):
    #invalid occasion are removed within plot function
    plot_folder = plot_box_whisker(self.features, self.predictors, ground_truth_variable, self.model_figures_folder, self.cohort_name, filter=filter)

  def principal_component_analysis(self):
    explained_variances = PCA_with_features(self.features, self.predictors)
    self.PCA = explained_variances

  def cross_validation(self, model_name, model_design):
    print('Training ' + model_name + ' ' + model_design)
    CV_group_k_fold(model_design, model_name, self)
    
    if 'RF' in model_name:
      index_of_model = next((i for i, m in enumerate(self.models) if m.model_name == model_name), None)
      self.models[index_of_model].feature_importance = plot_rf_feature_importances(self.models[index_of_model], model_name, self.model_figures_folder)
  
  #in development
  def iterative_near_real_time_testing(self):
  
    features = pd.DataFrame(columns=['subid', 'condition', 'curve_duration', 'peak', 'rise_rate', 'fall_rate', 'rise_duration', 'fall_duration', 'avg_tac_diff'])
    for i in range(5, 120, 5):
      features['curve_duration'].append()
      for occasion in self.occasions:

        baseline_mean, baseline_sd = get_baseline_mean_stdev(occasion.dataset, f'TAC_processed_smooth_{occasion.smoothing_window}')
        peak_index = get_peak_index(occasion.dataset, f'TAC_processed_smooth_{occasion.smoothing_window}')
        rise_duration, curve_beginning = get_rise_duration(occasion.dataset, f'TAC_processed_smooth_{occasion.smoothing_window}', occasion.time_variable, peak_index)
        data = occasion.dataset.iloc[curve_beginning: curve_beginning + i]
        #find beginning of curve index
        
        #crop dataset from here to i
        #calculate rise rate, peak, fall rate, rise duration, fall duration, duration, average tac change

  def make_predictions_using_default_models(self, models, export_report=True, save_processor=False):

    for model in models:
      name = model.model_name
      if 'Alc_vs_Non' in name:
        ground_truth_variable = 'condition' if 'Unk' not in self.features['condition'] else None
        filter = {'valid_occasion': [0]}
        pred_labels = {1: 'Alc', 0: 'Non'}
      elif 'Binge' in name:
        ground_truth_variable = 'binge' if 'Light' in self.features['binge'] or 'Heavy' in self.features['binge'] else None
        filter = {
          'valid_occasion': [0], 
          }
        alc_vs_non_pred_column = 'RF_Alc_vs_Non_default_prediction' if 'RF_Alc_vs_Non_default_prediction' in self.features.columns else 'LR_Alc_vs_Non_default_prediction' if 'LR_Alc_vs_Non_default_prediction' in self.features.columns else ''
        if alc_vs_non_pred_column:
          filter[alc_vs_non_pred_column] = [0, 'Non']
        pred_labels = {
          1: 'Heavy', 0: 'Light'
        }
      
      if 'default' not in model.model_name:
        model.model_name = model.model_name + '_default'
      self.make_binary_predictions(model, model.model_name, pred_labels=pred_labels, filter=filter, ground_truth_variable=ground_truth_variable)

    if export_report:
      self.export_SDM_report()


  def make_binary_predictions(self, model, model_name, filter={'valid_occasion': [0]}, pred_labels = {1: 'Alc', 0: 'Non'}, ground_truth_variable='condition'):
    features = self.features.copy()
    
    for column, exclude_values in filter.items():
      excluded = features[features[column].isin(exclude_values)]
      features = features[~features[column].isin(exclude_values)]

    predictions = model.predict(features[self.prediction_features])
    features[model_name + '_prediction'] = [pred_labels[x] for x in predictions]
    excluded[model_name + '_prediction'] = ['excluded' for i in range(0, len(excluded))]

    if ground_truth_variable:
      features[model_name + '_correct'] = np.where(features[model_name + '_prediction'] == features[ground_truth_variable], 'correct', 'incorrect')
      excluded[model_name + '_correct'] = ['excluded' for i in range(0, len(excluded))]

    features = pd.concat([features, excluded])
    # except:
    #   features[model_name + '_pred'] = ['prediction failed' for i in range(0, len(features))]
    #   features[model_name + '_correct'] = ['' for i in range(0, len(features))]
    features.sort_values(by=['subid', 'dataset_identifier', 'episode_identifier'], inplace=True)

    for i, occasion in enumerate(self.occasions):
      filter = (features['subid'] == occasion.subid) & (features['dataset_identifier'] == occasion.dataset_identifier) & (features['episode_identifier'] == occasion.episode_identifier)
      if len(filter):
        occasion.predictions[model_name] = features.loc[filter, model_name + '_prediction'].tolist()[0]

    self.features = features


        
    # python_object_folder = self.analyses_out_folder + '/Python_Objects'
    # if not os.path.exists(python_object_folder):
    #   os.mkdir(python_object_folder)

  # def make_binary_predictions(self, predictors = ['curve_auc', 'rise_rate', 'fall_duration', 'peak', 'fall_rate', 'rise_duration', 'TAC_N', 'avg_tac_diff', 'tac_alt_perc', 'major_outlier_N', 'minor_outlier_N']):
  #   for version in ['CLN', 'RAW']:
  #     features = self.stats[version]
  #     if 'major_outlier_N' not in features.columns.tolist():
  #       features['major_outlier_N'] = [0 for i in range(0, len(features))]
  #     if 'minor_outlier_N' not in features.columns.tolist():
  #       features['minor_outlier_N'] = [0 for i in range(0, len(features))]
  #     for model in self.models:
  #       try:
  #         self.predictions[model.model_name + '_' + version] = model.predict(features[predictors])
  #         predictions = self.predictions[model.model_name + '_' + version]
  #         self.stats[version][model.model_name + '_prediction'] = ['Alc' if x==1 else 'Non' for x in predictions]
  #         self.stats[version][model.model_name + '_correct'] = np.where(
  #           self.stats[version]['condition'] == 'Unk', 
  #             'unknown',
  #           np.where(self.stats[version][model.model_name + '_prediction'] == self.stats[version]['condition'], 
  #             'correct', 'incorrect'))
  #       except:
  #         self.stats[version][model.model_name + '_prediction'] = ['prediction failed' for i in range(0, len(features))]
  #         self.stats[version][model.model_name + '_correct'] = ['' for i in range(0, len(features))]
        
  #   python_object_folder = self.analyses_out_folder + '/Python_Objects'
  #   if not os.path.exists(python_object_folder):
  #     os.mkdir(python_object_folder)

    
    




    
