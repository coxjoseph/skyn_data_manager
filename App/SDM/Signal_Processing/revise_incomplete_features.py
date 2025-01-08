import pandas as pd
from SDM.Configuration.file_management import load_default_model


def revise_fall_features(occasion, fall_revision_threshold=0.5):
    if occasion.stats['fall_completion'] < fall_revision_threshold:
        logistic_regression = load_default_model(name='fall_duration')
        predictor_values = pd.DataFrame(
            [[occasion.stats[p] for p in logistic_regression.predictors]],
            columns=logistic_regression.predictors  # Set the predictors as column names
        )
        occasion.stats[logistic_regression.outcome] = logistic_regression.predict(predictor_values)
        occasion.stats['fall_rate_CLN'] = occasion.stats['relative_peak_CLN'] / occasion.stats[
            logistic_regression.outcome]
    elif occasion.stats['fall_completion'] < 0.9:
        occasion.stats[f'fall_duration_CLN'] = (((1 - occasion.stats['fall_completion']) / (
            occasion.stats['fall_completion'])) * occasion.stats['fall_duration_CLN']) + occasion.stats[
                                                   'fall_duration_CLN']


def revise_rise_features(occasion, rise_revision_threshold=0.5):
    if occasion.rise_completion < rise_revision_threshold:
        logistic_regression = load_default_model(name='rise_duration')
        predictor_values = pd.DataFrame(
            [[occasion.stats[p] for p in logistic_regression.predictors]],
            columns=logistic_regression.predictors  # Set the predictors as column names
        )
        occasion.stats[logistic_regression.outcome] = logistic_regression.predict(predictor_values)
        occasion.stats['rise_rate_CLN'] = (occasion.stats['relative_peak_CLN'] /
                                           occasion.stats[logistic_regression.outcome])
    elif occasion.rise_completion < 0.9:
        occasion.stats[f'rise_duration_CLN'] = (((1 - occasion.stats['fall_completion']) / (
            occasion.stats['fall_completion'])) * occasion.stats['rise_duration_CLN']) + occasion.stats[
                                                   'rise_duration_CLN']
