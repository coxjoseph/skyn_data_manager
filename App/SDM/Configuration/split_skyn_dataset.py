from datetime import datetime, timedelta
from time import strptime

import pandas as pd
from SDM.Configuration.configuration import configure_timestamp_column


# TODO: this looks like where you would implement social days
def split_skyn_dataset(data_to_split: pd.DataFrame, split_time: str) -> dict[str, pd.DataFrame]:
    data_to_split['datetime'] = configure_timestamp_column(data_to_split)
    data_to_split.sort_values(by='datetime', inplace=True)

    unique_days = data_to_split['datetime'].dt.date.unique().tolist()
    day_before = unique_days[0] - timedelta(days=1)
    unique_days = [day_before] + unique_days

    datasets = {}
    for unique_day in unique_days:
        start, end = calculate_day_split_range(unique_day, split_time)
        data = data_to_split[(data_to_split['datetime'] > start) & (data_to_split['datetime'] < end)]
        if len(data) > 0:
            datasets[f'{min(data['datetime'])} - {max(data['datetime'])}'] = data
    return datasets


def calculate_day_split_range(unique_day: datetime.date, split_time: str) -> (datetime, datetime):
    time = datetime.strptime(split_time, '%H:%M')
    start = datetime.combine(unique_day, time)
    end = start + timedelta(days=1)
    return start, end


def split_skyn_dataset_by_email(data_to_split: pd.DataFrame, split_time: str):
    data_to_split['datetime'] = configure_timestamp_column(data_to_split)
    data_to_split.sort_values(by='datetime', inplace=True)

    all_datasets = {}

    email_groups = data_to_split.groupby('email')

    for email, group in email_groups:
        datasets = {}

        unique_days = group['datetime'].dt.date.unique().tolist()
        day_before = unique_days[0] - timedelta(days=1)
        unique_days = [day_before] + unique_days

        for unique_day in unique_days:
            start, end = calculate_day_split_range(unique_day, split_time)
            data = group[(group['datetime'] > start) & (group['datetime'] < end)]
            if len(data) > 0:
                key = f"{email}: {min(data['datetime'])} - {max(data['datetime'])}"
                datasets[key] = data

        all_datasets[email] = datasets
    return all_datasets
