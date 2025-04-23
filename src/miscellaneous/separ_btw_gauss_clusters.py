from src.read_data import read_subset_features
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def find_optimum_nb_clusters(features):
    # Elbow method to determine optimal K
    wcss = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS

    # Plot the Elbow Method Graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.show()



def calc_separability_of_kmeans(features_ut,features_nonut, ut_):


    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters_ut = kmeans.fit_predict(features_ut)
    clusters_nonut = kmeans.fit_predict(features_nonut)
    
if __name__ == '__main__':
    main_dir = "D:\Verona_msXL"
    folio_name = r"msXL_335v_b"
    modality = "M"
    folio_names = ["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                   "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b",
                   "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", r"msXL_315r_b"]
    all_features_ut = []
    all_features_nonut = []
    for folio_name in folio_names:
        features_ut, _, _ = read_subset_features(main_dir, folio_name, "undertext", modality, box=None)
        features_nonut, _, _ = read_subset_features(main_dir, folio_name, "undertext", modality, box=None)
        all_features_ut.append(features_ut)
        all_features_nonut.append(features_nonut)
    all_features_ut = np.concatenate(all_features_ut, axis=0)
    all_features_nonut = np.concatenate(all_features_nonut, axis=0)
    find_optimum_nb_clusters(all_features_ut)
    find_optimum_nb_clusters(all_features_nonut)