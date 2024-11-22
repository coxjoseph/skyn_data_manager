from datetime import datetime
import pytz

"""
FOR REVISIONS

if baseline mean is very high and drinking began prior to captured baseline, 
then impute values prior to go back in time
"""


# TODO: regularize time
def get_start_date_time(dataset):
    """provide dataframe, function returns date and time, separately. Assumes dataframe is already sorted by datetime
    column"""
    datetime_start = dataset.loc[0, 'datetime']
    datetime_start = datetime.strptime(str(datetime_start), '%Y-%m-%d %H:%M:%S')

    start_date = datetime_start.date()
    return start_date, datetime_start


def crop_device_off_end(df):
    total_count = len(df['Temperature_C'])
    below_threshold_count = sum(1 for temp in df['Temperature_C'] if temp < 27)
    below_threshold_majority = below_threshold_count / total_count > 0.9
    below_threshold_end = any(
        [True for temp in df.loc[len(df) - 6: len(df) - 1, 'Temperature_C'].tolist() if temp <= 27])

    index = -1
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

    index = 0
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
