from sklearn.feature_extraction.text import TfidfVectorizer
from vectorizing import X
from clean_title import df
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




# def optimal_clusters(X):
        
#         scores = []

#         for k in range(2, len(df)):
#             k_means = KMeans(n_clusters= 5, init= 10, random_state= 42)
#             print(k_means.fit(X))
#             score = silhouette_score(X, KMeans.labels_)
#             scores = scores.append(score)
#             print(scores)
#         return scores

# function who is searching for optimal number of cluster

def find_optimal_clusters(X, max_k = 50):

#the maximum number of cluster should not exceed the half of the documents length
    max_possible_k = min(max_k, len(df) // 2)
    if max_possible_k < 2:
         return 2

# we want to test for each number of cluster which gives the best score
 
    k_range = range(2, max_possible_k)

# we will stock each cluster its corresponding score inside the "scores" table 
    scores = []
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            scores.append(score)
            print(f"cluster : {k}, score:{score}")
        except:
            scores.append(-1)  #  Clustering errors

    if not scores or all(s == -1 for s in scores):
        return 2

    best_k = k_range [scores.index(max(scores))]
    print(f"best_cluster: {best_k} score {score}")
    return best_k
    

# calling the function to process the matrix
# if len(df) >=10 : 
    # n_clusters = find_optimal_clusters(X)
# else :
#     n_clusters = min(3, len(df)) 

n_clusters = find_optimal_clusters(X) 

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

df['cluster'] = kmeans.fit_predict(X)


print(df['cluster'])