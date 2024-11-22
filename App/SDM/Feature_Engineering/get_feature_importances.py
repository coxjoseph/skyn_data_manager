import pandas as pd


def get_feature_importances(model, features):
    forest_importance = pd.DataFrame(model.feature_importances_, index=features,
                                     columns=['Mean Decrease Impurity']).sort_values('Mean Decrease Impurity',
                                                                                     ascending=False).rename_axis(
        'Feature')
    return forest_importance
