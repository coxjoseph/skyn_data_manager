import pandas as pd
from datetime import datetime, timedelta
from SDM.Visualization.plotting import plot_cropping
import pytz

"""
FOR REVISIONS

if baseline mean is very high and drinking began prior to captured baseline, 
then impute values prior to go back in time
"""


def get_start_date_time(dataset):
    """provide dataframe, function returns date and time, separately. Assumes dataframe is already sorted by datetime
    column"""
    datetime_start = dataset.loc[0, 'datetime']
    datetime_start = datetime.strptime(str(datetime_start), '%Y-%m-%d %H:%M:%S')

    start_date = datetime_start.date()
    return start_date, datetime_start


def get_session_start_and_end(self, session_timestamp, crop_end_adjustment=0, crop_begin_adjustment=0,
                              start_date_column='Crop Begin Date', end_date_column="Crop End Date"):
    """provide metadata/timestamp row and max duration (integer, in hours), and the function will provide the
    timestamps to crop at beginning and end"""

    date_format_str = '%Y-%m-%d %H:%M:%S'

    if self.crop_start_with_timestamps:
        time_begin_drinking = str(session_timestamp.loc[0, 'Crop Begin Time'])
        date_begin_drinking = str(session_timestamp.loc[0, start_date_column])
        datetime_string_begin_drinking = date_begin_drinking + ' ' + time_begin_drinking
        datetime_begin = datetime.strptime(datetime_string_begin_drinking, date_format_str) + pd.Timedelta(
            hours=crop_begin_adjustment)
    else:
        datetime_begin = self.begin

    if self.crop_end_with_timestamps:
        time_end = str(session_timestamp.loc[0, 'Crop End Time'])
        date_end = str(session_timestamp.loc[0, end_date_column])
        datetime_string_end_drinking = date_end + ' ' + time_end
        datetime_end = datetime.strptime(datetime_string_end_drinking, date_format_str) + pd.Timedelta(
            hours=crop_end_adjustment)
        self.max_duration = (datetime_end - datetime_begin).total_seconds() / 3600
        self.crop_method = 'start and end timestamps'
    else:
        datetime_end = datetime_begin + timedelta(hours=self.max_duration)
        self.crop_method = 'start timestamp + max duration' if self.crop_start_with_timestamps else 'max duration'

    return datetime_begin, datetime_end


def crop_using_timestamps(self, dataset):
    """crop dataset using timestamps file (see Resource folder for example). If Crop End Timestamps are also provided,
    data after this timestamp will be removed. Otherwise, data after the default max duration will be removed."""
    filter = (self.metadata['SubID'] == self.subid) & (
            self.metadata['Episode_Identifier'] == int(self.episode_identifier[1:])) & (
                     self.metadata['Dataset_Identifier'] == self.dataset_identifier)

    session_timestamp = self.metadata[filter]
    session_timestamp.reset_index(inplace=True, drop=True)

    if (self.skyn_upload_timezone == 999) or (self.skyn_upload_timezone is None) or (
            self.skyn_upload_timezone == '999'):
        # 999 is used when timezone is assumed to be correct, therefore no timezone conversion
        dataset.loc[:, 'datetime'] = dataset.loc[:, 'datetime'] + pd.Timedelta(hours=0)
    else:
        utc_offset = get_utc_offset(session_timestamp.loc[0, 'Crop Begin Date'], self.skyn_upload_timezone)
        device_user_timezone_utc = int(session_timestamp.loc[0, 'Time Zone'].split(':')[0]) if session_timestamp.loc[
            0, 'Time Zone'] else utc_offset
        hour_adjustment = device_user_timezone_utc - utc_offset
        dataset.loc[:, 'datetime'] = dataset.loc[:, 'datetime'] + pd.Timedelta(hours=hour_adjustment)

    datetime_begin, datetime_end = get_session_start_and_end(self, session_timestamp, self.crop_end_adjustment,
                                                             self.crop_begin_adjustment)
    cropped_plot_path = plot_cropping(dataset, datetime_begin, datetime_end, self)
    self.crop_begin = datetime_begin
    self.crop_end = datetime_end
    cropped_dataset = dataset[(dataset['datetime'] > datetime_begin) & (dataset['datetime'] < datetime_end)]
    cropped_dataset.sort_values(by='Duration_Hrs', inplace=True)
    cropped_dataset.reset_index(drop=True, inplace=True)

    self.begin = cropped_dataset['datetime'].min()
    self.end = cropped_dataset['datetime'].max()

    if len(cropped_dataset) > 0:
        time_correction = cropped_dataset.loc[0, 'Duration_Hrs']
        cropped_dataset.loc[:, 'Duration_Hrs'] = cropped_dataset['Duration_Hrs'] - time_correction
    return cropped_dataset, cropped_plot_path


def crop_device_off_end(df):
    total_count = len(df['Temperature_C'])
    below_threshold_count = sum(1 for temp in df['Temperature_C'] if temp < 27)
    below_threshold_majority = below_threshold_count / total_count > 0.9
    below_threshold_end = any(
        [True for temp in df.loc[len(df) - 6: len(df) - 1, 'Temperature_C'].tolist() if temp <= 27])

    if below_threshold_end and not below_threshold_majority:
        temp_below_threshold = True
        counter = 6
        while temp_below_threshold:
            index = len(df) - counter
            if (df.loc[index, 'Temperature_C'] > 27) or (index == len(df) - 1):
                temp_below_threshold = False
            if counter == len(df) - 6:
                temp_below_threshold = False
            counter += 1
        return df.loc[:index], counter
    else:
        return df, 0


def crop_device_off_start(df):
    total_count = len(df['Temperature_C'])
    below_threshold_count = sum(1 for temp in df['Temperature_C'] if temp < 27)
    below_threshold_majority = below_threshold_count / total_count > 0.9
    below_threshold_start = any([True for temp in df.loc[:6, 'Temperature_C'].tolist() if temp <= 27])

    if below_threshold_start and not below_threshold_majority:
        temp_below_threshold = True
        counter = 6
        while temp_below_threshold:
            index = counter
            if (df.loc[index, 'Temperature_C'] > 27) or (index == len(df) - 1):
                temp_below_threshold = False
            counter += 1
        df = df.loc[index:]
        df.reset_index(inplace=True)
        return df, counter
    else:
        return df, 0


def get_utc_offset(date, timezone):
    city = {'CST': 'America/Chicago',
            'EST': 'America/New_York',
            'MST': 'America/Denver',
            'PST': 'America/Los_Angeles'}[timezone]

    tz = pytz.timezone(city)

    date_time_input = datetime.combine(date, datetime.min.time())
    # Convert the input date to the specified timezone
    date_in_tz = tz.localize(date_time_input)

    # Get the UTC offset in hours
    utc_offset = date_in_tz.utcoffset().total_seconds() // 3600

    return utc_offset
