from SDM.Configuration.file_management import merge_using_subid
import pandas as pd
import xlsxwriter.utility

from SDM.Skyn_Processors.skyn_dataset import SkynDataset


# TODO: what is this supposed to do
def export_variable_key(writer, new_model_development):
    pass


class SkynCohortTester:
    def __init__(self, cohort_processor, models: list = None):
        if models is None:
            models = []
        self.models = models
        self.data_folder = cohort_processor.data_folder
        self.cohort_name = cohort_processor.cohort_name
        self.data_out_folder = cohort_processor.data_out_folder
        self.graphs_out_folder = cohort_processor.graphs_out_folder
        self.analyses_out_folder = cohort_processor.analyses_out_folder
        self.merge_variables = cohort_processor.merge_variables
        self.occasions = cohort_processor.occasions
        self.stats = cohort_processor.stats
        self.predictions = {}

    # TODO: what is this supposed to do
    def make_binary_predictions(self, predictors):
        pass

    def create_feature_and_predictions_report(self):
        writer = pd.ExcelWriter(f'{self.analyses_out_folder}/SDM_predictions_{self.cohort_name}.xlsx',
                                engine='xlsxwriter')

        for version in ['CLN', 'RAW']:
            to_export = self.stats[version]
            cols = to_export.columns.tolist()
            for _ in self.models:
                cols.insert(3, cols.pop(len(cols) - 1))
                cols.insert(3, cols.pop(len(cols) - 1))
            to_export = to_export[cols]
            to_export = merge_using_subid(to_export, self.merge_variables)
            to_export.to_excel(writer, index=False, sheet_name=f'Features - {version}')

        # TODO: this method doesnt exist??
        export_variable_key(writer, new_model_development=False)

        counter = 0

        for occasion in self.occasions:
            row = (counter * 20) + 2
            col = 15
            stats = occasion.stats['CLN']
            signal_processing_info = occasion.signal_cleaning_results
            sheet_name = 'Report by Drinking Episode'

            occasion_data = [occasion.subid, occasion.condition, occasion.dataset_identifier, occasion.drinks,
                             'Yes' if occasion.crop_start_with_timestamps else 'No',
                             'Yes' if occasion.crop_end_with_timestamps else 'No']
            index_labels = ['SubID', 'Condition', 'Dataset_Identifier', 'Drink Total', 'Start Cropped', 'End Cropped']
            for model in self.models:
                # Assuming model_name can be cast to string from model - cannot find any reference to the usage here
                model_name = str(model)
                results = self.stats['CLN']
                prediction_clean = get_results_column(results, occasion, model_name)
                correct_clean = get_results_column(results, occasion, model_name, '_correct')

                results = self.stats['RAW']
                prediction_raw = get_results_column(results, occasion, model_name)
                correct_raw = get_results_column(results, occasion, model_name, '_correct')

                occasion_data.extend([prediction_clean, correct_clean])
                index_labels.extend(
                    [model.model_name + ' prediction (cleaned)', model.model_name + ' correct (cleaned)'])
                occasion_data.extend([prediction_raw, correct_raw])
                index_labels.extend([model.model_name + ' prediction (raw)', model.model_name + ' correct (raw)'])

            basic_info = pd.Series(
                name='Occasion Info / Model Predictions',
                data=occasion_data,
                index=index_labels
            )
            basic_info.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=1)

            general_stats = pd.Series(
                name='TAC Dataset: General Stats',
                data=[stats['mean'], stats['stdev'], stats['sem'], stats['TAC_N'], stats['baseline_mean'],
                      stats['baseline_stdev'], stats['avg_tac_diff'], stats['tac_alt_perc'], stats['not_worn_percent'],
                      stats['worn_duration'], stats['worn_duration_percent'], stats['valid_occasion']],
                index=['TAC Mean', 'TAC SD', 'TAC SEM', 'TAC Count', 'Baseline Mean', 'Baseline SD',
                       'Avg TAC Difference', 'TAC Alteration %', 'Not Worn %', 'Valid Duration', 'Valid %',
                       'Valid Occasion'])
            general_stats.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=4)

            curve_features = pd.Series(
                name='Curve Features',
                data=[stats['peak'], stats['auc_total'], stats['auc_per_hour'], stats['curve_duration'],
                      stats['rise_duration'], stats['fall_duration'], stats['rise_rate'], stats['fall_rate']],
                index=['Peak', 'AUC Total', 'AUC / Hr', 'Curve Duration', 'Rise Duration', 'Fall Duration', 'Rise Rate',
                       'Fall Rate'])
            curve_features.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=7)

            cleaning_results = pd.Series(
                name='Cleaning Results',
                data=[stats['TAC_N'], stats['major_outlier_N'], signal_processing_info['major_threshold'],
                      stats['minor_outlier_N'], signal_processing_info['minor_threshold'], stats['imputed_N'], ],
                index=['TAC Count', 'Major Outlier Count', 'Major Threshold', 'Minor Outlier Count', 'Minor Threshold',
                       'Imputed Count'])
            cleaning_results.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col * 2 + 5)

            counter += 1

        counter = 0
        workbook = writer.book
        worksheet = workbook.get_worksheet_by_name('Report by Drinking Episode')
        crop_plot_placement_adjustment = 1 if any([occasion.crop_with_timestamps for occasion in self.occasions]) else 0
        for occasion in self.occasions:
            col_start = 10
            col_interval = 13
            row = (counter * 20) + 2

            x_scale = 65 / 140
            y_scale = 90 / 182

            plot_folder = f'{occasion.plot_folder}/{occasion.subid}/{occasion.condition}'

            x = xlsxwriter.utility.xl_col_to_name(col_start + (col_interval * 0))
            image_start_cell = x + str(row)
            worksheet.insert_image(image_start_cell,
                                   f'{plot_folder}/{occasion.subid}_{occasion.condition}'
                                   f'{occasion.dataset_identifier}_smoothed_curve.png',
                                   {'x_scale': x_scale,
                                    'y_scale': y_scale})

            if occasion.crop_with_timestamps:
                x = xlsxwriter.utility.xl_col_to_name(col_start + (col_interval * 1))
                image_start_cell = x + str(row)
                worksheet.insert_image(image_start_cell,
                                       f'{plot_folder}/{occasion.subid}_{occasion.condition}'
                                       f'{occasion.dataset_identifier}_cropping.png',
                                       {'x_scale': x_scale,
                                        'y_scale': y_scale})

            x = xlsxwriter.utility.xl_col_to_name(col_start + (col_interval * (1 + crop_plot_placement_adjustment) + 2))
            image_start_cell = x + str(row)
            worksheet.insert_image(image_start_cell,
                                   f'{plot_folder}/cleaning/cleaning - {occasion.subid} - '
                                   f'{occasion.condition}{occasion.dataset_identifier}.png',
                                   {'x_scale': x_scale,
                                    'y_scale': y_scale})

            x = xlsxwriter.utility.xl_col_to_name(col_start + (col_interval * (2 + crop_plot_placement_adjustment) + 2))
            image_start_cell = x + str(row)
            worksheet.insert_image(image_start_cell,
                                   f'{plot_folder}/{occasion.subid}_{occasion.condition}{occasion.dataset_identifier}'
                                   f'_temp_cleaning.png',
                                   {'x_scale': x_scale,
                                    'y_scale': y_scale})

            counter += 1
        writer.close()


def get_results_column(results: pd.DataFrame,
                       occasion: SkynDataset, model_name: str,
                       suffix: str = '_prediction') -> list:
    column_name = f"{model_name}{suffix}"
    return results.loc[
        (results['subid'] == occasion.subid) &
        (results['condition'] == occasion.condition) &
        (results['dataset_identifier'] == occasion.dataset_identifier),
        column_name
    ].tolist()[0]
