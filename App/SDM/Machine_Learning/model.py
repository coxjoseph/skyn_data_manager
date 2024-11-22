from SDM.Machine_Learning.model_optimization import create_search_grid
import pandas as pd


class Model:
    def __init__(self, model, name, predictors, outcome, n_splits, group_k_fold=None):
        if group_k_fold is None:
            group_k_fold = []
        self.model = model
        self.model_name = name
        self.predictors = predictors
        self.outcome = outcome
        self.n_splits = n_splits
        self.best = None
        self.use_group_k_fold = True if len(group_k_fold) else False
        self.group_k_fold = group_k_fold
        self.optimized = False
        self.feature_importance = pd.DataFrame()
        self.cv_results = {}

    def optimize(self):
        model_map = {
            "LR": "logistic_reg",
            "RF": "random_forest",
            "LinearReg": "linear_reg"
        }

        # FIXME: theoretically could find multiple models, don't think it does but if it do it will use the last
        for key, model_type in model_map.items():
            if key in self.model_name:
                self.model = create_search_grid(
                    model_type,
                    self.n_splits,
                    group_kfold=self.use_group_k_fold
                )
                self.optimized = True

    def fit(self, x, y):
        if self.use_group_k_fold:
            self.model.fit(x, y)
            self.best = self.model.best_estimator_
        else:
            self.model.fit(x, y)
            self.best = self.model.best_estimator_

    def predict(self, x):
        predictions = self.model.predict(x)
        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions
