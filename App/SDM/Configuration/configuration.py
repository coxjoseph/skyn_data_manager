"""
this file contains utilities for loading the date, standardizing formats, retrieving unique identifiers, etc.
"""
from typing import Optional
import warnings
import pandas as pd
import numpy as np
from SDM.User_Interface.Utils.filename_tools import stringify_dataset_id

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


def get_closest_index_with_timestamp(data: pd.DataFrame,
                                     timestamp: pd.Timestamp,
                                     datetime_column: str) -> Optional[int]:
    try:
        closest_index = (data[datetime_column] - timestamp).abs().idxmin()
        return closest_index
    except (KeyError, TypeError, ValueError) as e:
        print(f"An error occurred while finding the closest index: {e}")
        return None


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
