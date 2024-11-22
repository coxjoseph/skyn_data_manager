from sklearn.decomposition import PCA


def pca_with_features(features, selected_features):
    features = features[features['valid_occasion'] == 1]
    x = features[selected_features]
    pca = PCA(n_components=len(selected_features))
    pca.fit(x)
    return pca.explained_variance_ratio_
