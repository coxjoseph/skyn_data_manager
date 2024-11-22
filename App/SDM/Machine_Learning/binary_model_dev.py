from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
from SDM.Machine_Learning.model import Model


def cv_group_k_fold(model_design, model_name, cohort_processor):
    ground_truth_column, grouping_column_name, ground_truth_labels, cv_filter, additional_columns = get_cv_specs(
        model_design)

    # columns that are required for cross validation or filtering but not used as model predictors
    non_predictor_columns = list(
        set([grouping_column_name, ground_truth_column] + list(cv_filter.keys()) + additional_columns))
    features = cohort_processor.features[non_predictor_columns + cohort_processor.training_features].reset_index(
        drop=True)

    # removing unwanted subjects/datasets
    features, excluded = filter_features(features, cv_filter)

    # relabeling the ground truth labels to integers (for training compatibility)
    if len(ground_truth_labels) > 0:
        features.loc[:, ground_truth_column] = [ground_truth_labels[c] for c in features[ground_truth_column].tolist()]

    k_groups = features[grouping_column_name].tolist()
    n_splits = len(features[grouping_column_name].unique())
    x = features[cohort_processor.training_features]
    y = features[ground_truth_column]

    model = Model(model=None, name=model_name, predictors=cohort_processor.training_features,
                  outcome=ground_truth_column, n_splits=n_splits, group_k_fold=k_groups)
    model.optimize()

    # rain model using those settings
    model.fit(x, y)

    # use model to make predictions and probabilities
    predictions = cross_val_predict(model.best, x, y, cv=GroupKFold(n_splits=n_splits), groups=k_groups,
                                    method='predict')
    probabilities = cross_val_predict(model.best, x, y, cv=GroupKFold(n_splits=n_splits), groups=k_groups,
                                      method='predict_proba')

    features[f'pred_{model_name}'] = predictions
    correct, incorrect, features = split_predictions(features, ground_truth_column,
                                                     prediction_column=f'pred_{model_name}', model_name=model_name)

    excluded[f'pred_{model_name}'] = 'excluded'

    cv_results = {
        'Prediction_Features': cohort_processor.training_features,
        'Features': features,
        'Incorrect_Occasions': incorrect,
        'Correct_Occasions': correct,
        'Predictions': predictions,
        'Probabilities': probabilities,
        'Splits': n_splits,
        'Prediction_N': len(correct) + len(incorrect),
        'Correct_N': len(correct),
        'TP': len(correct[correct[ground_truth_column] == 1]),
        'TN': len(correct[correct[ground_truth_column] == 0]),
        'FP': len(incorrect[incorrect[ground_truth_column] == 0]),
        'FN': len(incorrect[incorrect[ground_truth_column] == 1]),
        'Accuracy': len(correct) / (len(correct) + len(incorrect)),
        'AUC_ROC': roc_auc_score(y, probabilities[:, 1]),
    }

    model.cv_results = calculate_results(cv_results, model_name=model_name)
    cohort_processor.features[f'{model_name}_prediction'] = ['' for _ in range(0, len(cohort_processor.occasions))]

    save_tac_feature_predictions_to_sdm(cohort_processor, model_name, ground_truth_column, ground_truth_labels,
                                        incorrect, correct)

    cohort_processor.models.append(model)


def get_cv_specs(model_design: str):
    specs = {
        'Alc_vs_Non': {
            'cv_filter': {'valid_occasion': [0]},
            'ground_truth_column': 'condition',
            'grouping_column_name': 'subid',
            'ground_truth_labels': {'Alc': 1, 'Non': 0},
            'additional_columns': [],
        },
        'Light_vs_Heavy': {
            'cv_filter': {'valid_occasion': [0], 'condition': ['Non'], 'binge': ['Unk', 'None', None]},
            'ground_truth_column': 'binge',
            'grouping_column_name': 'subid',
            'ground_truth_labels': {'Heavy': 1, 'Light': 0, 'None': -9, 'Unk': 999},
            'additional_columns': [],
        },
        'AUD_vs_Not': {
            'cv_filter': {'valid_occasion': [0], 'condition': ['Non']},
            'ground_truth_column': 'AUD',
            'grouping_column_name': 'subid',
            'ground_truth_labels': {},
            'additional_columns': [],
        },
    }

    if 'worn_vs_removed' in model_design:
        return (
            'device_on',
            'Full_Identifier',
            {1: 1, 0: 0},
            {'device_on': 'unk'},
            ['Row_ID', 'Full_Identifier'],
        )

    if model_design not in specs:
        raise ValueError(f"Unsupported model design: {model_design}")

    model_spec = specs[model_design]
    return (
        model_spec['ground_truth_column'],
        model_spec['grouping_column_name'],
        model_spec['ground_truth_labels'],
        model_spec['cv_filter'],
        model_spec['additional_columns'],
    )


# TODO: This seems excessive?
def split_predictions(features, ground_truth_column, prediction_column, model_name):
    features[f'{model_name}_result'] = np.where(
        (features[ground_truth_column] == 1) & (features[prediction_column] == 1), 'True Positive',
        np.where((features[ground_truth_column] == 0) & (features[prediction_column] == 1), 'False Positive',
                 np.where((features[ground_truth_column] == 0) & (features[prediction_column] == 0), 'True Negative',
                          np.where((features[ground_truth_column] == 1) & (features[prediction_column] == 0),
                                   'False Negative', 'Unknown'))))

    features[f'{model_name}_correct'] = np.where(features[ground_truth_column] == features[prediction_column],
                                                 'correct', 'incorrect')
    correct = features[features[f'{model_name}_correct'] == 'correct']
    incorrect = features[features[f'{model_name}_correct'] == 'incorrect']

    return correct, incorrect, features


def save_tac_feature_predictions_to_sdm(cohort_processor, model_name, ground_truth_column, group_labels, incorrect,
                                        correct):
    for i, occasion in enumerate(cohort_processor.occasions):
        condition = occasion.condition if ground_truth_column != 'condition' else group_labels[
            getattr(occasion, ground_truth_column.lower())]
        ground_truth = group_labels[getattr(occasion, ground_truth_column.lower())] if len(group_labels) else getattr(
            occasion, ground_truth_column.lower())

        if len(incorrect[(incorrect['subid'] == occasion.subid) & (incorrect[ground_truth_column] == ground_truth) & (
                incorrect['condition'] == condition)]):
            occasion.predictions[model_name] = 'incorrect'
            cohort_processor.features.loc[i, f'{model_name}_prediction'] = 'incorrect'
        elif len(correct[(correct['subid'] == occasion.subid) & (correct[ground_truth_column] == ground_truth) & (
                correct['condition'] == condition)]):
            occasion.predictions[model_name] = 'correct'
            cohort_processor.features.loc[i, f'{model_name}_prediction'] = 'correct'
        else:
            occasion.predictions[model_name] = 'excluded'
            cohort_processor.features.loc[i, f'{model_name}_prediction'] = 'excluded'


def filter_features(features, feature_filter):
    excluded = pd.DataFrame(columns=features.columns)
    for column, values_to_exclude in feature_filter.items():
        excluded_rows = features[features[column].isin(values_to_exclude)]
        excluded = pd.concat([excluded, excluded_rows])
    excluded = excluded.drop_duplicates()

    features = features[~features.isin(excluded)].dropna(how='all')
    features = features.drop_duplicates()

    return features, excluded


def train_and_test_model_with_holdout(features, predictors, model_name='worn_vs_removed_LR', k=3, holdout=0.3):
    ground_truth_column, grouping_column_name, ground_truth_labels, feature_filter, additional_columns = get_cv_specs(
        model_name)

    features, excluded = filter_features(features, feature_filter)

    non_predictor_columns = list(
        set([grouping_column_name, ground_truth_column] + list(feature_filter.keys()) + additional_columns))
    features = features[non_predictor_columns + predictors].reset_index(drop=True)
    if len(ground_truth_labels) > 0:
        features.loc[:, ground_truth_column] = [ground_truth_labels[c] for c in features[ground_truth_column].tolist()]

    if holdout > 0:
        holdout_n = round(len(features[grouping_column_name].unique()) * holdout)
        holdout_group = features[grouping_column_name].unique().tolist()[:holdout_n]
        training_group = features[grouping_column_name].unique().tolist()[holdout_n:]
        holdout_features = features[features[grouping_column_name].isin(holdout_group)]
        training_features = features[features[grouping_column_name].isin(training_group)]
    else:
        training_features = features
        holdout_features = None

    predictors = [col for col in training_features.columns if
                  col not in additional_columns and col not in [ground_truth_column]]
    x = training_features[predictors]
    y = training_features[ground_truth_column]

    model = Model(model=None, name=model_name, predictors=predictors, outcome=ground_truth_column, n_splits=k)
    model.optimize()
    model.fit(x, y)

    if holdout > 0:
        x_test = holdout_features[predictors]
        pred = model.predict(x_test)
        probabilities = model.model.predict_proba(x_test)
        holdout_features[f'{ground_truth_column}_pred'] = pred

        correct, incorrect, holdout_features = split_predictions(holdout_features, ground_truth_column,
                                                                 prediction_column=f'{ground_truth_column}_pred',
                                                                 model_name=model_name)

        training_features.loc[:, f'{ground_truth_column}_pred'] = 'training'
        training_features.loc[:, f'{model_name}_result'] = 'training'
        training_features.loc[:, f'{model_name}_correct'] = 'training'
        all_features = pd.concat([holdout_features, training_features])
        cv_results = {
            'Prediction_Features': predictors,
            'Training_Features': training_features,
            'Holdout_Features': holdout_features,
            'All_Features': all_features,
            'Incorrect_Rows': incorrect,
            'Correct_Rows': correct,
            'Predictions': pred,
            'Probabilities': probabilities,
            'Splits': 2,
            'Holdout_Proportion': holdout,
            'Prediction_N': len(correct) + len(incorrect),
            'Correct_N': len(correct),
            'TP': len(correct[correct[ground_truth_column] == 1]),
            'TN': len(correct[correct[ground_truth_column] == 0]),
            'FP': len(incorrect[incorrect[ground_truth_column] == 0]),
            'FN': len(incorrect[incorrect[ground_truth_column] == 1]),
            'Accuracy': len(correct) / (len(correct) + len(incorrect)),
            'AUC_ROC': roc_auc_score(holdout_features[ground_truth_column], probabilities[:, 1]),
        }

        model.cv_results = calculate_results(cv_results, model_name=model_name)

    return model


def calculate_results(cv_results: dict, model_name: str) -> dict:
    cv_results['Sensitivity'] = cv_results['TP'] / (cv_results['TP'] + cv_results['FN'])
    cv_results['Specificity'] = cv_results['TN'] / (cv_results['TN'] + cv_results['FP'])
    cv_results['CV_Results_Dataframe'] = pd.DataFrame(
        index=['Features', 'Prediction_N', 'Split Method', 'TP', 'TN', 'FP', 'FN', 'Correct', 'Sensitivity',
               'Specificity', 'Accuracy', 'AUC_ROC'], data={
            f'{model_name}_Result': [', '.join(cv_results['Prediction_Features']), cv_results['Prediction_N'],
                                     cv_results['Splits'], cv_results['TP'], cv_results['TN'], cv_results['FP'],
                                     cv_results['FN'], cv_results['Correct_N'], cv_results['Sensitivity'],
                                     cv_results['Specificity'], cv_results['Accuracy'], cv_results['AUC_ROC']]})

    return cv_results
