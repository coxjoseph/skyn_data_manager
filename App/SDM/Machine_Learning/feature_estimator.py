from sklearn.linear_model import LinearRegression
from SDM.Machine_Learning.binary_model_dev import filter_features
from SDM.Machine_Learning.model import Model
from SDM.Configuration.file_management import save_to_computer


def train_feature_estimator(cohort_processor, predictors, outcome, training_filter=None):
    if training_filter is None:
        training_filter = {}
    features = cohort_processor.features

    features, excluded = filter_features(features, training_filter)

    x_train = features[predictors]
    y_train = features[outcome]

    lr = Model(LinearRegression(), outcome + '_LinearReg', predictors, outcome, n_splits=3, group_k_fold=[])
    lr.optimize()
    lr.fit(x_train, y_train)

    # TODO: sdmtm is here
    save_to_computer(lr, lr.model_name, cohort_processor.python_object_folder, extension='sdmtm')
