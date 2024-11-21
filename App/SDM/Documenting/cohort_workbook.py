import pandas as pd
import numpy as np
import xlsxwriter
from SDM.Configuration.file_management import merge_using_subid
from SDM.Documenting.ordered_feature_columns import *


class SDM_Report:
    def __init__(self, skyn_cohort):
        self.skyn_cohort = skyn_cohort
        self.writer = pd.ExcelWriter(f'{self.skyn_cohort.analyses_out_folder}/{self.skyn_cohort.filename}',
                                     engine='xlsxwriter')
        self.ordered_feature_columns = ordered_feature_columns
        self.ordered_dataset_columns = get_ordered_dataset_columns(self.skyn_cohort.smoothing_window)

        self.cropping_plot_available = 1 if any(
            [occasion.crop_start_with_timestamps or occasion.crop_end_with_timestamps for occasion in
             self.skyn_cohort.occasions]) else 0

    def save_and_close(self):
        self.writer.close()

    def run_export(self):
        self.export_variable_key()  #KEY tab
        self.export_dataset_features()  #Summaries and Plots Tab
        self.export_episode_plots()  #Summaries and Plots Tab
        self.export_dataset_features(invalid_occasions=True)  #Invalid Summaries Tab
        self.export_episode_plots(invalid_occasions=True)  # Invalid Summaries Tab
        self.export_features()  # Features and Invalid tab
        self.export_model_performance()  #Model Results tab
        self.export_feature_importance()  #Feature Importance tab
        self.export_master_dataset()  #All Data tab
        self.save_and_close()

    def export_variable_key(self):
        key_path = 'App/SDM/Documenting/FeatureKey.xlsx'
        variable_key = pd.read_excel(key_path, index_col='Name', sheet_name='Dictionary')
        categories = pd.read_excel(key_path, index_col='Dictionary Categories', sheet_name='Categories')
        tab_description = pd.read_excel(key_path, index_col='Tab Name', sheet_name='TabDescription')

        tab_description.to_excel(self.writer, sheet_name='KEY', startcol=1)
        categories.to_excel(self.writer, sheet_name='KEY', startrow=len(tab_description) + 2, startcol=1)
        variable_key.to_excel(self.writer, sheet_name='KEY', startrow=len(categories) + len(tab_description) + 4)

    def export_dataset_features(self, invalid_occasions=False):
        counter = 0
        occasions = self.skyn_cohort.occasions if not invalid_occasions else self.skyn_cohort.invalid_occasions
        sheet_name = 'Summaries and Plots' if not invalid_occasions else 'Invalid Summaries'

        for occasion in occasions:
            stats = occasion.stats
            row = (counter * 20) + 2
            col = 15

            try:
                basic_info = pd.Series(name='Basic Info',
                                       data=[occasion.subid, occasion.condition, occasion.dataset_identifier,
                                             occasion.episode_identifier, occasion.drinks, occasion.binge, occasion.aud,
                                             'Yes' if occasion.crop_start_with_timestamps else 'No',
                                             'Yes' if occasion.crop_end_with_timestamps else 'No'],
                                       index=['SubID', 'Condition', 'Dataset ID', 'Episode ID', 'Drink Total', 'Binge',
                                              'AUD', 'Cropped Start', 'Cropped End'])
                basic_info.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=1)
            except:
                basic_info = pd.DataFrame({'Basic Info': [f'Unable to generate: {occasion.invalid_reason}']})
                basic_info.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=1)

            try:
                general_stats = pd.Series(name='General Stats (CLN)',
                                          data=[stats['TAC_N_CLN'], occasion.stats['fall_completion'],
                                                'Yes' if occasion.stats[
                                                             'fall_completion'] < occasion.fall_revision_threshold else 'No',
                                                'Yes' if occasion.stats['fall_completion'] < 1 else 'No',
                                                stats['not_worn_percent'], stats['worn_duration'],
                                                stats['worn_duration_percent'], occasion.valid_occasion],
                                          index=['TAC Count', 'Fall Completion', 'Fall Rate Revised',
                                                 'Fall Duration Revised', 'Not Worn %', 'Valid Duration', 'Valid %',
                                                 'Valid Occasion'])
                for model_name, prediction in occasion.predictions.items():
                    general_stats[model_name] = prediction
                general_stats.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=4)
            except:
                general_stats = pd.DataFrame(
                    {'General Stats (CLN)': [f'Unable to generate: {occasion.invalid_reason}']})
                general_stats.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=4)

            try:
                curve_features = pd.Series(name='Curve Features (CLN)',
                                           data=[stats['peak_CLN'], stats['auc_total_CLN'], stats['auc_per_hour_CLN'],
                                                 stats['curve_duration_CLN'], stats['rise_duration_CLN'],
                                                 stats['fall_duration_CLN'], stats['rise_rate_CLN'],
                                                 stats['fall_rate_CLN'], stats['completed_curve_count_CLN'],
                                                 stats['curve_alterations_CLN']],
                                           index=['Peak', 'AUC Total', 'AUC / Hr', 'Curve Duration', 'Rise Duration',
                                                  'Fall Duration', 'Rise Rate', 'Fall Rate', 'Curve Count',
                                                  'Alterations'])
                curve_features.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=7)
            except:
                curve_features = pd.DataFrame(
                    {'Curve Features (CLN)': [f'Unable to generate: {occasion.invalid_reason}']})
                curve_features.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=7)

            try:
                cleaning_stats = pd.Series(name='Cleaning Stats',
                                           data=[stats['TAC_N_CLN'], stats['major_outlier_N'],
                                                 occasion.major_cleaning_threshold, stats['minor_outlier_N'],
                                                 occasion.minor_cleaning_threshold, stats['imputed_N']],
                                           index=['TAC Count', 'Major Outlier Count', 'Major Threshold',
                                                  'Minor Outlier Count', 'Minor Threshold', 'Imputed Count'])
                cleaning_stats.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=col * 2 + 5)
            except:
                cleaning_stats = pd.DataFrame({'Cleaning Stats': [f'Unable to generate: {occasion.invalid_reason}']})
                cleaning_stats.to_excel(self.writer, sheet_name=sheet_name, startrow=row, startcol=col * 2 + 5)

            counter += 1

    def export_episode_plots(self, invalid_occasions=False):
        counter = 0
        workbook = self.writer.book
        worksheet = workbook.get_worksheet_by_name(
            'Summaries and Plots') if not invalid_occasions else workbook.get_worksheet_by_name('Invalid Summaries')
        occasions = self.skyn_cohort.occasions if not invalid_occasions else self.skyn_cohort.invalid_occasions

        for occasion in occasions:
            col_start = 10
            col_interval = 12
            #should be based on height of General Stats
            row = (counter * 20) + 2

            x_scale = 65 / 140
            y_scale = 90 / 182

            try:
                x = xlsxwriter.utility.xl_col_to_name(col_start + (col_interval * 0))
                image_start_cell = x + str(row)
                worksheet.insert_image(image_start_cell,
                                       occasion.official_curve_plot,
                                       {'x_scale': x_scale,
                                        'y_scale': y_scale})
            except:
                pass

            try:
                if occasion.crop_start_with_timestamps or occasion.crop_end_with_timestamps:
                    x = xlsxwriter.utility.xl_col_to_name(col_start + (col_interval * 1))
                    image_start_cell = x + str(row)
                    worksheet.insert_image(image_start_cell, occasion.cropping_plot,
                                           {'x_scale': x_scale,
                                            'y_scale': y_scale})
            except:
                pass

            try:
                x = xlsxwriter.utility.xl_col_to_name(
                    col_start + (col_interval * (1 + self.cropping_plot_available) + 2))
                image_start_cell = x + str(row)
                worksheet.insert_image(image_start_cell, occasion.cleaning_comparison_plot,
                                       {'x_scale': x_scale,
                                        'y_scale': y_scale})
            except:
                pass

            try:
                x = xlsxwriter.utility.xl_col_to_name(
                    col_start + (col_interval * (2 + self.cropping_plot_available) + 2))
                image_start_cell = x + str(row)
                worksheet.insert_image(image_start_cell, occasion.device_removal_plot,
                                       {'x_scale': x_scale,
                                        'y_scale': y_scale})
            except:
                pass

            counter += 1

    def export_model_performance(self):
        results_col_idx = 0
        prob_col_idx = 0
        for model in self.skyn_cohort.models:
            model.cv_results['CV_Results_Dataframe'].to_excel(self.writer, sheet_name='Model_Results',
                                                              index=(results_col_idx == 0), startcol=results_col_idx)
            results_col_idx += 1 if results_col_idx != 0 else 2

            df_header = pd.DataFrame({'': [f'model: {model.model_name}']})
            probabilities = pd.DataFrame(model.cv_results['Probabilities'], columns=['-', '+'])
            probabilities = probabilities.round(2)
            id_indices = self.skyn_cohort.features[
                self.skyn_cohort.features[f'{model.model_name}_prediction'] != 'excluded'].index.tolist()
            id_column = (self.skyn_cohort.features.loc[id_indices, 'subid'].astype(str) + '_' +
                         self.skyn_cohort.features.loc[id_indices, 'dataset_identifier'].astype(str) +
                         self.skyn_cohort.features.loc[id_indices, 'episode_identifier'])
            id_column = id_column.to_frame().reset_index(drop=True)
            probabilities.insert(0, 'ID', id_column)

            df_header.to_excel(self.writer, sheet_name='Probabilities', startrow=1, header=False, index=False,
                               startcol=prob_col_idx)
            probabilities.to_excel(self.writer, sheet_name='Probabilities', startrow=3, startcol=prob_col_idx,
                                   index=False)
            prob_col_idx += len(probabilities.columns) + 1

    def export_features(self):
        features_to_export = self.skyn_cohort.features.copy()
        features_to_export.drop('TAC_N_Raw', axis=1, inplace=True, errors='ignore')

        columns_to_round = []

        #reordering columns
        matching_columns = [col for col in self.ordered_feature_columns if col in features_to_export.columns.tolist()]
        non_matching_columns = [col for col in features_to_export.columns if col not in self.ordered_feature_columns]
        all_columns_ordered = matching_columns + non_matching_columns
        features_to_export = features_to_export[all_columns_ordered]

        #round numeric columns
        columns_to_round = [col for i, col in enumerate(features_to_export.columns) if
                            pd.api.types.is_numeric_dtype(features_to_export[col]) and i >= 15 and col[
                                                                                                   -2:] != '_N' and col != 'removal_detect_method']
        features_to_export[columns_to_round] = features_to_export[columns_to_round].round(2)

        features_to_export = merge_using_subid(features_to_export, self.skyn_cohort.merge_variables)
        features_to_export.to_excel(self.writer, index=False, sheet_name="Features")
        self.skyn_cohort.invalid_features.to_excel(self.writer, index=False, sheet_name="Invalid")

    def export_feature_importance(self):
        row_idx = 1
        col_idx = 1
        for model in self.skyn_cohort.models:
            if len(model.feature_importance) > 0:
                df_header = pd.DataFrame({'': [f'model: {model.model_name}']})
                df_header.to_excel(self.writer, sheet_name='Feature Importance', startrow=row_idx, header=False,
                                   index=False, startcol=col_idx)
                model.feature_importance.to_excel(self.writer, sheet_name='Feature Importance', startrow=row_idx + 2,
                                                  startcol=col_idx)
                col_idx += len(model.feature_importance.columns) + 2

    def export_master_dataset(self):
        all_data = self.skyn_cohort.master_dataset
        matching_columns = [col for col in self.ordered_dataset_columns if col in all_data.columns.tolist()]
        all_data = all_data[matching_columns]
        all_data.to_excel(self.writer, sheet_name='All Data', index=False)
