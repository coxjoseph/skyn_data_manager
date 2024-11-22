from sklearn.cluster import KMeans
from kneed import KneeLocator


def get_distortions(data, k_max):
    distortions = []
    k_values_to_test = range(1, k_max)
    for k in k_values_to_test:
        tac = data['TAC'].to_numpy().reshape(-1, 1)
        k_means_model = KMeans(n_clusters=k)
        k_means_model.fit(tac)
        distortions.append(k_means_model.inertia_)
    return distortions, k_values_to_test


def get_knee(distortions, k_values_to_test):
    kn = KneeLocator(k_values_to_test, distortions, curve='convex', direction='decreasing')
    return kn.knee


def label_clusters(data, optimal_k):
    tac = data['TAC'].to_numpy().reshape(-1, 1)
    kmeans = KMeans(n_clusters=optimal_k)
    prediction = kmeans.fit_predict(tac)
    data['cluster'] = prediction
    return data


def get_tac_clusters(data):
    distortions, k_values_to_test = get_distortions(data, 10)
    optimal_k = get_knee(distortions, k_values_to_test)
    data = label_clusters(data, optimal_k)
    return data
