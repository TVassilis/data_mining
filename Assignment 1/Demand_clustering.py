import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import zscore
import matplotlib.pyplot as plt

if __name__ == '__main__':
    demand_file = 'Demand_Merged.csv'
    demand_df = pd.read_csv(demand_file)

    df = pd.DataFrame(list(demand_df.iloc[:, 3]))
    m = df.mean()
    df = df.fillna(m)
    time_series = zscore(np.array(df))

    # plt.plot(time_series)
    # plt.show()

    time_series = time_series.reshape(-1, 1)
    # print(time_series)

    cluster_num = 8
    kmeans_model = KMeans(n_clusters=cluster_num)
    # res = kmeans_model.fit_transform(time_series)
    res = kmeans_model.fit_predict(time_series)

    centroids = kmeans_model.cluster_centers_
    # print(centroids)

    distances = []
    for index, item in enumerate(time_series):
        distances.append((index, abs(centroids[res[index]] - item)[0]))

    distances.sort(key=lambda y: y[1])
    # print(distances)
    outliers = distances[-3:]
    print([c[0] for c in outliers])
    print([time_series[i][0] for i in [c[0] for c in outliers]])

    plt.plot(time_series)
    plt.show()
