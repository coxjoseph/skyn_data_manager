"""
this file contains utilities for loading the date, standardizing formats, retrieving unique identifiers, etc.
"""
from typing import Optional, Any
import warnings
import pandas as pd
from pandas import NaT
import numpy as np
import os
from App.SDM.User_Interface.Utils.filename_tools import stringify_dataset_id
from App.SDM.Skyn_Processors.skyn_dataset import skynDataset

starting_standard_columns = ['datetime', 'device_id', 'Firmware Version', 'TAC ug/L(air)', 'Temperature_C', 'Motion']


def update_column_names(df):
    df.rename(columns={
        'device timestamp': 'datetime',
        'Timestamp': 'datetime',
        'Device Serial Number': 'device_id',
        'firmware version': 'Firmware Version',
        'Firmware version': 'Firmware Version',
        'tac (ug/L)': 'TAC ug/L(air)',
        'temperature (C)': 'Temperature_C',
        'Temperature (C)': 'Temperature_C',
        'Temperature C': 'Temperature_C',
        'Temperature LSB': 'Temperature_C',
        'motion (g)': 'Motion',
        'Motion LSB': 'Motion',
        'device id': 'device_id',
        'device.id': 'device_id'
    },
        inplace=True,
        errors='ignore'
    )
    return df


def rename_tac_column(df):
    df.rename(columns={'TAC ug/L(air)': 'TAC'}, inplace=True)
    df.drop('TAC LSB', axis=1, inplace=True, errors='ignore')
    return df


def includes_multiple_device_ids(df):
    return len(df['device_id'].unique()) > 1


def normalize_column(series):
    mean = series.mean()
    stdev = series.std()
    norm = (series - mean) / stdev
    return norm


def configure_timestamp_column(df):
    df['datetime_with_timezone'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce')
    df['datetime_without_timezone'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Check which format was successful
    if not df['datetime_with_timezone'].isna().all():
        df['datetime'] = df['datetime_with_timezone'].dt.tz_localize(None)
    elif not df['datetime_without_timezone'].isna().all():
        df['datetime'] = df['datetime_without_timezone']
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Drop the intermediate columns
    df.drop(['datetime_with_timezone', 'datetime_without_timezone'], axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df['datetime']


def get_sampling_rate(df, timestamp_column):
    time_diff = df[timestamp_column].diff()

    # Calculate the average time difference
    average_sampling_rate = time_diff.mean().total_seconds()  # Get the average in seconds
    average_sampling_rate_per_minute = round(60 / average_sampling_rate)  # Calculate samples per minute

    return average_sampling_rate_per_minute


def get_time_elapsed(df, timestamp_column):
    try:
        start_time = df[timestamp_column].iloc[0]
        df['Duration_Hrs'] = (df[timestamp_column] - start_time).dt.total_seconds() / 3600  # Time elapsed (h)
        return df
    except (KeyError, AttributeError, TypeError) as e:
        print(f"Error calculating elapsed time: {e}.")
        df['Duration_Hrs'] = np.nan
        return df


def remove_junk_columns(df):
    for col in df.columns:
        if col not in starting_standard_columns:
            df.drop('level_0', axis=1, inplace=True, errors='ignore')
    return df


def get_metadata_index(dataset: skynDataset):
    print(dataset.subid)
    print(dataset.episode_identifier)
    print(dataset.dataset_identifier)
    try:
        filtered_metadata = dataset.metadata[(dataset.metadata['SubID'] == dataset.subid) & (
                dataset.metadata['Episode_Identifier'] == int(dataset.episode_identifier[1:])) &
                                             (dataset.metadata['Dataset_Identifier'] == int(
                                                 dataset.dataset_identifier))]
        return filtered_metadata.index.tolist()[0]
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error getting metadata index: {e}")
        return None


def load_metadata(dataset: skynDataset, column: str = 'TotalDrks') -> Optional[Any]:
    try:
        return dataset.metadata.loc[dataset.metadata_index, column]
    except (KeyError, IndexError):
        return None


def is_binge(dataset: skynDataset) -> str:
    if dataset.drinks is None or dataset.sex is None:
        return "Unk"
    if dataset.drinks == 0:
        return "None"
    if dataset.sex == 1 and dataset.drinks >= 5:
        return "Heavy"
    if dataset.sex == 2 and dataset.drinks >= 4:
        return "Heavy"
    if dataset.drinks > 0:
        return "Light"
    return "Unk"


def correct_baseline_tac(tac_column: pd.Series) -> pd.Series:
    min_value = tac_column.min()
    if min_value < 0:
        return tac_column - min_value
    return tac_column


def nearest_odd(number):
    # TODO: why is this here?
    # Constant time bitwise or does the job here
    return number | 1


def get_full_identifier(subid, dataset_identifier, episode_identifier):
    return str(subid) + '_' + stringify_dataset_id(dataset_identifier) + episode_identifier


def get_full_identifier_from_metadata(metadata_row):
    dataset_identifier = '' if str(metadata_row['Dataset_Identifier']) == 'nan' else stringify_dataset_id(
        metadata_row['Dataset_Identifier'])
    episode_identifier = 'e1' if str(metadata_row['Episode_Identifier']) == 'nan' else 'e' + str(
        metadata_row['Episode_Identifier'])
    return str(metadata_row['SubID']) + '_' + dataset_identifier + episode_identifier


def get_cohort_full_identifiers(metadata):
    """provide metadata, function returns a list containing ..?"""
    return [get_full_identifier_from_metadata(row) for i, row in metadata.iterrows() if row['Use_Data'] == 'Y']


def is_dmy_format(dates):
    # TODO: this is jank and will fail for a *lot* of dates. Look into updating timeseries format
    filtered_dates = dates.dropna()
    any_dmy = pd.to_datetime(filtered_dates, errors='coerce', format='%d/%m/%Y').notnull().any()
    all_mdy = pd.to_datetime(filtered_dates, errors='coerce', format='%m/%d/%Y').notnull().all()
    return any_dmy and not all_mdy


# TODO: standardize the column into something better (ISO format)
def standardize_date_column_ymd(timestamps, column_name='Crop Begin Date', new_column_name="Crop Begin Date"):
    """
    provide structured dataframe (see Resource folder for example),
    ensures standard datetime type for the date column
    """

    warnings.filterwarnings("ignore", message="Parsing '.*' in DD/MM/YYYY format.*", category=UserWarning)

    # standardize column format
    if is_dmy_format(timestamps[column_name]):
        mask = pd.to_datetime(timestamps[column_name], errors='coerce', format='%d/%m/%Y').notnull()
        timestamps.loc[~mask, column_name] = pd.to_datetime(timestamps.loc[~mask, column_name],
                                                            errors='coerce').dt.strftime('%m/%d/%Y')
        timestamps[column_name] = pd.to_datetime(timestamps[column_name], errors='coerce', format='%d/%m/%Y')
    else:
        timestamps[column_name] = pd.to_datetime(timestamps[column_name], errors='coerce', format='%m/%d/%Y')

    # remove time component of datetime variable to only have date
    timestamps[new_column_name] = timestamps[new_column_name].apply(lambda x: x.date())
    return timestamps


def configure_timestamps(metadata):
    metadata.reset_index(inplace=True, drop=True)
    metadata = standardize_date_column_ymd(metadata, column_name='Crop Begin Date', new_column_name='Crop Begin Date')
    metadata = standardize_date_column_ymd(metadata, column_name='Crop End Date', new_column_name='Crop End Date')
    return metadata


# TODO: go through and use Paths not strings
def create_output_folders(dataset: skynDataset):
    if not os.path.exists(dataset.data_out_folder):
        os.mkdir(dataset.data_out_folder)
    if not os.path.exists(dataset.plot_folder):
        os.mkdir(dataset.plot_folder)
    subid_plot_folder = f'{dataset.plot_folder}/{dataset.subid}/'
    if not os.path.exists(subid_plot_folder):
        os.mkdir(subid_plot_folder)
    full_plot_folder = (f'{dataset.plot_folder}/{dataset.subid}/{dataset.dataset_identifier}'
                        f'{dataset.condition if dataset.condition else ""}/')
    if not os.path.exists(full_plot_folder):
        os.mkdir(full_plot_folder)
    dataset.plot_folder = full_plot_folder


# TODO: save as paths and use .extensions and stuff!!
def load_dataset(dataset: skynDataset):
    if dataset.path[-3:] == 'csv':
        return pd.read_csv(dataset.path, index_col=False)
    else:
        return pd.read_excel(dataset.path, index_col=False)


def configure_raw_data(dataset: skynDataset):
    print(f'initializing \n{dataset.subid} \n{dataset.condition} {dataset.dataset_identifier}')
    print(dataset.path)
    dataset.unprocessed_dataset['SubID'] = dataset.subid
    dataset.unprocessed_dataset['Dataset_Identifier'] = dataset.dataset_identifier
    dataset.unprocessed_dataset['Episode_Identifier'] = dataset.episode_identifier
    dataset.unprocessed_dataset['Full_Identifier'] = get_full_identifier(dataset.subid, dataset.dataset_identifier,
                                                                         dataset.episode_identifier)

    df_raw = update_column_names(dataset.unprocessed_dataset)
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
def timestamp_available(dataset: skynDataset, begin_or_end='Begin'):
    filter_ = (dataset.metadata['SubID'] == dataset.subid) & (
            (dataset.metadata['Dataset_Identifier'] == str(dataset.dataset_identifier)) | (
            dataset.metadata['Dataset_Identifier'] == int(dataset.dataset_identifier))) & (
                      dataset.metadata['Episode_Identifier'] == int(dataset.episode_identifier[1:]))
    return (dataset.metadata.loc[filter_, f'Crop {begin_or_end} Date'].notnull().any() and
            dataset.metadata.loc[filter_, f'Crop {begin_or_end} Time'].notnull().any())


def get_event_timestamps(dataset: skynDataset, metadata_path: str) -> dict[str, Any]:
    try:
        timestamps = pd.read_excel(metadata_path, sheet_name="Additional_Timestamps")
        # TODO: why > 5? why not all? rewrite
        timestamp_columns = [col for i, col in enumerate(timestamps.columns) if i > 5]

        timestamps[timestamp_columns] = timestamps[timestamp_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M',
                                                                            errors='coerce')

        filtered_timestamps = timestamps[
            (timestamps['SubID'] == dataset.subid) &
            (timestamps['Dataset_Identifier'] == dataset.dataset_identifier) &
            (timestamps['Episode_Identifier'] == int(dataset.episode_identifier[1:]))
            ]

        event_timestamps = {
            col: filtered_timestamps.loc[filtered_timestamps.index[0], col]
            for col in filtered_timestamps.select_dtypes(include='datetime64').columns
        }

        return {key: value for key, value in event_timestamps.items() if value is not NaT}

    except (FileNotFoundError, ValueError, KeyError, IndexError, TypeError) as e:
        print(f"An error occurred while processing the timestamps: {e}")
        return {}


def get_closest_index_with_timestamp(data: pd.DataFrame,
                                     timestamp: pd.Timestamp,
                                     datetime_column: str) -> Optional[int]:
    try:
        closest_index = (data[datetime_column] - timestamp).abs().idxmin()
        return closest_index
    except (KeyError, TypeError, ValueError) as e:
        print(f"An error occurred while finding the closest index: {e}")
        return None


def determine_initial_validity(dataset: skynDataset) -> tuple[int, Optional[str]]:
    """Checks metadata and device to see if event should be checked
    """
    # TODO: does this need to return anything? It seems like it's only setting class attributes which should be read
    if dataset.valid_occasion and dataset.metadata_index is None:
        dataset.valid_occasion = 0
        dataset.invalid_reason = 'Not in metadata'

    use_data = load_metadata(dataset, 'Use_Data')
    if use_data == 'N':
        dataset.valid_occasion = 0
        dataset.invalid_reason = f'Excluded by metadata: {dataset.metadata_note}'
        return dataset.valid_occasion, dataset.invalid_reason
    if dataset.disabled_by_multiple_device_ids:
        dataset.valid_occasion = 0
        dataset.invalid_reason = 'Multiple devices detected within dataset'
        return dataset.valid_occasion, dataset.invalid_reason

    dataset.valid_occasion = 1
    dataset.invalid_reason = None
    return dataset.valid_occasion, dataset.invalid_reason


def calculate_device_inactivity_stats(dataset: skynDataset) -> None:
    device_off_or_removed = get_device_off_or_removed(dataset)
    dataset.stats['device_inactive_perc'] = (device_off_or_removed.sum() / len(dataset.dataset)) * 100
    dataset.stats['device_inactive_duration'] = (device_off_or_removed.sum() * dataset.sampling_rate) / 60
    dataset.stats['device_active_perc'] = 100 - dataset.stats['device_inactive_perc']
    dataset.stats['device_active_duration'] = (
                                                      dataset.dataset['Duration_Hrs'].max() -
                                                      dataset.dataset['Duration_Hrs'].min()
                                              ) - (device_off_or_removed.sum() * dataset.sampling_rate) / 60


def calculate_imputation_stats(dataset: skynDataset, all_imputations: pd.Series) -> None:
    dataset.stats['imputed_N'] = all_imputations.sum()
    dataset.stats['tac_imputed_duration'] = (all_imputations.sum() * dataset.sampling_rate) / 60
    dataset.stats['tac_imputed_perc'] = (all_imputations.sum() / len(dataset.dataset)) * 100


def get_device_off_or_removed(dataset: skynDataset) -> pd.Series:
    return (dataset.dataset['gap_imputed'] == 1) | (dataset.dataset['TAC_device_off_imputed'] == 1)


def get_all_imputations(dataset: skynDataset) -> pd.Series:
    device_off_or_removed = get_device_off_or_removed(dataset)
    return (
            device_off_or_removed |
            (dataset.dataset['major_outlier'] == 1) |
            (dataset.dataset['minor_outlier'] == 1) |
            (dataset.dataset['sloped_start'] == 1) |
            (dataset.dataset['extreme_outlier'] == 1)
    )


def validate_device_usage(dataset: skynDataset) -> tuple[int, Optional[str]]:
    device_off_or_removed = get_device_off_or_removed(dataset)
    device_active_duration = ((len(dataset.dataset) - device_off_or_removed.sum()) * dataset.sampling_rate) / 60
    enough_device_on = (
            dataset.stats['device_inactive_perc'] < dataset.max_percentage_inactive
            and device_active_duration > dataset.min_duration_active
    )

    if not enough_device_on:
        dataset.valid_occasion = 0
        dataset.invalid_reason = (
            f'Duration of device inactivity (device non-wear, device off) is too great. Max allowed percentage of '
            f'inactivity ({dataset.max_percentage_inactive}%) was exceeded or the minimum required duration of '
            f'{dataset.min_duration_active} hours was not met. Device inactive for '
            f'{round(dataset.stats["device_inactive_perc"], 1)}% and there is '
            f'{round(device_active_duration, 1)} hours of valid data.'
        )
        return dataset.valid_occasion, dataset.invalid_reason

    return 1, None


def validate_imputation(dataset: skynDataset) -> tuple[int, Optional[str]]:
    """Determine if the device data is valid based on the imputation percentage."""
    too_many_imputations = dataset.stats['tac_imputed_perc'] > dataset.max_percentage_imputed

    if too_many_imputations:
        dataset.valid_occasion = 0
        dataset.invalid_reason = (
            f'Device contains too many artifacts or noisy data. Specifically, '
            f'{round(dataset.stats["tac_imputed_perc"], 1)}% of the data was imputed, exceeding the limit of '
            f'{dataset.max_percentage_imputed}%.'
        )
        return dataset.valid_occasion, dataset.invalid_reason

    return 1, None


def determine_post_cleaning_validity(dataset: skynDataset) -> tuple[int, Optional[str]]:
    all_imputations = get_all_imputations(dataset)

    calculate_device_inactivity_stats(dataset)
    calculate_imputation_stats(dataset, all_imputations)

    valid_occasion, invalid_reason = validate_device_usage(dataset)
    if not valid_occasion:
        return valid_occasion, invalid_reason

    valid_occasion, invalid_reason = validate_imputation(dataset)
    return valid_occasion, invalid_reason


def reduce_sampling_rate(raw_data, timestamp_column, cutoff_sec=59):
    last_index = 0
    indices_to_keep = [last_index]

    for i, row in raw_data.iterrows():
        if i > 0:
            duration_diff = (
                    raw_data.loc[i, timestamp_column] - raw_data.loc[last_index, timestamp_column]).total_seconds()
            if duration_diff >= cutoff_sec:
                indices_to_keep.append(i)
                last_index = i

    reduced_data = raw_data.loc[indices_to_keep].reset_index(drop=True)

    return reduced_data
