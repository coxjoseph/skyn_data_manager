import pandas as pd


def get_sdm_run_settings(self, data_format, program, data_out, graphs_out, analyses_out, cohort_name):

    program_descriptions = {
        'P': 'Signal Processing',
        'PP': 'Signal Processing and Model Testing' if data_format == 'Processor' else 'Model Testing',
        'PTP': 'Signal Processing, Model Development & Testing' if data_format == 'Processor' else 'Model Development '
                                                                                                   '& Testing'
    }
    if data_format == 'Single':
        return pd.DataFrame(dict(zip(
            ['program', 'cohort name', 'data file', 'processed data folder', 'graphs folder', 'Timestamps to crop '
                                                                                              'pre-episode data',
             'Timezone of data download', 'Max dataset duration'],
            [[program_descriptions[self.program]], 
             [cohort_name], 
             [self.selected_data], 
             [data_out], 
             [graphs_out],
             [self.disable_timestamp_cropping],
             [self.skyn_upload_timezone],
             [self.max_dataset_duration]]
          ))).transpose()
    elif program == 'P':
        return pd.DataFrame(dict(zip(
            ['program', 'cohort name', 'raw data folder', 'processed data folder', 'graphs folder', 'Timestamps to '
                                                                                                    'crop pre-episode'
                                                                                                    ' data',
             'Timezone of data download', 'Max dataset duration'],
            [[program_descriptions[self.program]], 
             [cohort_name], 
             [self.selected_data], 
             [data_out], 
             [graphs_out],
             [self.disable_timestamp_cropping],
             [self.skyn_upload_timezone],
             [self.max_dataset_duration]]
          ))).transpose()
    elif program == 'PP':
        return pd.DataFrame(dict(zip(
            ['program', 'prediction models', 'cohort name', 'raw data folder', 'processed data folder', 'graphs folder',
             'ML results folder', 'Loaded already-processed data', 'files used in merging', 'chosen merge variables',
             'Subject ID column key', 'Timestamps to crop pre-episode data', 'Timezone of data download',
             'Max dataset duration'],
            [[program_descriptions[self.program]], 
             [", ".join([model.model_name for model in self.models])], 
             [cohort_name], 
             [self.selected_data], 
             [data_out], 
             [graphs_out],
             [analyses_out],
             ["Yes" if self.previous_processor else "No"], 
             [' | '.join([filename for filename, info in self.files_to_merge.items()])], 
             [' | '.join([', '.join(info['variables']) for variable, info in self.files_to_merge.items()])], 
             [' | '.join([info['subid_column'] for filename, info in self.files_to_merge.items()])],
             [self.disable_timestamp_cropping],
             [self.skyn_upload_timezone],
             [self.max_dataset_duration]]
          ))).transpose()
    elif program == 'PTP':
        return pd.DataFrame(dict(zip(
            ['program', 'cohort name', 'raw data folder', 'processed data folder', 'graphs folder',
             'model results folder', 'Loaded already-processed data', 'files used in merging', 'chosen merge variables',
             'Subject ID column key', 'Timestamps to crop pre-episode data', 'Timezone of data download',
             'Max dataset duration'],
            [[program_descriptions[self.program]], 
             [cohort_name], 
             [self.selected_data], 
             [data_out], 
             [graphs_out],
             [analyses_out],
             ["Yes" if self.previous_processor else "No"], 
             [' | '.join([filename for filename, info in self.files_to_merge.items()])], 
             [' | '.join([', '.join(info['variables']) for variable, info in self.files_to_merge.items()])], 
             [' | '.join([info['subid_column'] for filename, info in self.files_to_merge.items()])],
             [self.disable_timestamp_cropping],
             [self.skyn_upload_timezone],
             [self.max_dataset_duration]]
          ))).transpose()
    
    else:
        return pd.DataFrame()
