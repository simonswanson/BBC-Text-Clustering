
# coding: utf-8

# In[1]:

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as met


# # Part 1 - Clustering

# <h2> Question 1 - Import and Verify Data

# In[2]:

# import 3 CSV Files as X, trueLabels, terms

X = np.array(pd.read_csv('bbcsport_mtx.csv',header=None))
trueLabels = np.array(pd.read_csv('bbcsport_classes.csv', header=None))
terms = np.array(pd.read_csv('bbcsport_terms.csv', header=None))

# Verify shape of arrays

print X.shape
print trueLabels.shape
print terms.shape


# <h2> Question 2 - K-Means Clustering using Euclidean Distance</h2>

# In[219]:

# Part 1 - compute KMeans for X with 5 clusters

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=50, random_state=2)
kmeans.fit_predict(X)
predLabels = np.array(kmeans.labels_)
centroids = kmeans.cluster_centers_


# In[63]:

# Part 2 - Calculate Adjusted Rand Score and MI Score

true_Labels = trueLabels.flatten()
print("Adjusted Rand Score: " + str(met.adjusted_rand_score(true_Labels, predLabels)))
print("\nAdjusted Mutual Information Score: " + str(met.adjusted_mutual_info_score(true_Labels, predLabels)))


# Both the adjusted Rand and MI scores are very low, indicating poor quality of clustering.  Adjusting the k-means parameters (number of initializations and iterations) had minimal effect, and attempting to run PCA prior to conducting clustering resulted in even worse scores (see Appendix - Clustering Results after PCA).

# In[213]:

# Part 3 - Run 50 random initializations of K-means and report average results

# Run 50 random initializations

rand_scores = np.zeros((50))
mi_scores = np.zeros((50))
centroids_euc = {}
for i in range(0,50):
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=50)
    kmeans.fit_predict(X)
    predLabels = np.array(kmeans.labels_)
    rand_scores[i] = (met.adjusted_rand_score(true_Labels, predLabels))
    mi_scores[i] = (met.adjusted_mutual_info_score(true_Labels, predLabels))
    centroids_euc[i] = kmeans.cluster_centers_


# In[197]:

# Print mean and st. dev. over 50 iterations

print ("\nAdjusted Rand Score Averaged over 50 initializations: " + str(rand_scores.mean()))
print ("Adjusted Rand Score Std Dev over 50 initializations: " + str(rand_scores.std()))
print ("\nAdjusted MI Score Averaged over 50 initializations: " + str(mi_scores.mean()))
print ("Adjusted MI Score St Dev over 50 initializations: " + str(mi_scores.std()))

# Find lowest & Highest 10 scores on Rand Index

low = np.argpartition(rand_scores, 10)[:10]
high = np.argpartition(rand_scores, -10)[-10:]
print ("\nLowest 10 scores - Rand Index: \n" + str(np.sort((rand_scores[low]))))
print ("\nHighest 10 scores - Rand Index: \n" + str(np.sort((rand_scores[high]))))


# In[210]:

# Plot histogram for each measure

plt.subplot(1,2,1)
plt.hist(rand_scores)
plt.title("Adjusted Rand Index Scores")

plt.subplot(1,2,2)
plt.hist(mi_scores)
plt.title("Adjusted Mutual Information Scores")
plt.tight_layout()
plt.show()


# Averaging over 50 random initializations returns similar poor clustering results on both Rand and MI measures, with five runs returning negative adjusted Rand scores, indicating that the clustering has performed worse than random chance on those runs.  Average results were significantly worse than the original single run. Both measures seem to have a somewhat symmetric distribution.

# <h2> Question 3 - K-Means Clustering using Cosine Distance</h2>

# In[64]:

# Part 1 - Compute Kmeans using Cosine distance with 5 clusters

# Use KMeans Clusterer from Natural Language Toolkit Package as this allows Cosine distance to be used as distance measure

# Import packages from NLTK

from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance



# In[65]:

# Perform clustering using Cosine distance
km_cos = KMeansClusterer(5, distance=cosine_distance, avoid_empty_clusters=True)
km_cos_cl = km_cos.cluster(X, assign_clusters=True)


# In[103]:

# Part 2 - Calculate Adjusted Rand Score and MI Score
print("Adjusted Rand Score (Cosine Distance): " + str(met.adjusted_rand_score(true_Labels, km_cos_cl)))
print("\nAdjusted Mutual Information Score (Cosine Distance): " + str(met.adjusted_mutual_info_score(true_Labels, km_cos_cl)))


# Clustering using Cosine distance has resulted in far better results on both the Adjusted Rand and Adjusted MI scores.  As with Euclidean distance, attempting clustering after performing PCA resulted in reduced scores on both indices (see Appendix).

# In[222]:

# Part 3 - Run 50 random initializations 

cos_rand_scores = np.zeros((50))
cos_mi_scores = np.zeros((50))
centroids_cos = {}
for i in range(0,50):
    km_cos = KMeansClusterer(5, distance=cosine_distance, avoid_empty_clusters=True)
    km_cos_cl = km_cos.cluster(X, assign_clusters=True)
    cos_rand_scores[i] = (met.adjusted_rand_score(true_Labels, km_cos_cl))
    cos_mi_scores[i] = (met.adjusted_mutual_info_score(true_Labels, km_cos_cl))
    centroids_cos[i] = km_cos.means()
    


# In[199]:

# Report average Adjusted Rand and Mutual Information scores

print ("Adjusted Rand Score Averaged over 50 initializations (Cosine Distance): " + str(cos_rand_scores.mean()))
print ("Adjusted Rand Score St Dev over 50 initializations (Cosine Distance): " + str(cos_rand_scores.std()))
print ("\nAdjusted Mutual Information Score Averaged over 50 initializations (Cosine Distance): " + str(cos_mi_scores.mean()))
print ("Adjusted Mutual Information Score St Dev over 50 initializations (Cosine Distance): " + str(cos_mi_scores.std()))

# Find 10 lowest & Highest scores 

lowcos = np.argpartition(cos_rand_scores, 10)[:10]
highcos = np.argpartition(cos_rand_scores, -10)[-10:]
print ("\nLowest 10 scores - Rand Index: \n" + str(np.sort((cos_rand_scores[lowcos]))))
print ("\nHighest 10 scores - Rand Index: \n" + str(np.sort((cos_rand_scores[highcos]))))




# In[211]:

# Plot histogram for each measure

plt.subplot(1,2,1)
plt.hist(cos_rand_scores)
plt.title("Adjusted Rand Index Scores")

plt.subplot(1,2,2)
plt.hist(cos_mi_scores)
plt.title("Adjusted Mutual Information Scores")
plt.tight_layout()
plt.show()


# Again, running 50 random initializations using Cosine distance has resulted in far better scores than what was achieved using Euclidean distance.  Both measures (ARI and AMI) using Cosine distance produced non-symmetric distributions of the mean values.  In addition, there was a very large difference in the scores achieved between the highest and lowest 10 scores, indicating high variability and the need to conduct multiple initializations of the clustering in order to achieve the best results.  Finally, as with the Euclidean distance clusters, the means achieved with the full set of features were far higher than the average mean achieved across 20 initializations with post-PCA data (per Appendix).

# <h2> Question 4 - Visualise Cluster Centres with WordCloud</h2>

# In[221]:

# Part 1- Euclidean Distance Clusters - Visualise Cluster Centres with WordCloud

#Import WordCloud package

from wordcloud import WordCloud

#  Find centroids of 'Best' run by highest ARI score

cent_max = centroids_euc[np.argmax(rand_scores)]

# Get indices of top terms per centroid

cent_sorted = cent_max.argsort()[:, ::-1] 

# Get 20 words closest to each centroid

centroid_terms = {}
for i in range(5):
    jlist = []
    for j in cent_sorted[i,:20]:
        jlist.append((str(terms[j])).replace("'",""))
    centroid_terms[i] = jlist

# Print Word Clouds for each cluster    

for i in centroid_terms:
    wordcloud = WordCloud().generate(str(centroid_terms[i]))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Euclidean Distance Cluster " + str(i) + " WordCloud")
    plt.show()


# TagClouds were produced using the top 20 terms closest to the centroids of each cluster, from the run with the highest ARI scores from the 50 random initializations.  Three of the Cluster topics are clearly represented in the Tag Clouds (4: cricket, 3: football and 1: tennis), while Clusters 0 and 2 seem to be more of a mix of more generic terms that could apply to most of the sports.

# In[228]:

# Part 2 - Visualise Cluster Centres with WordCloud - Cosine Distances

# Get indices of top terms per centroid from 'Best' run by ARI


cos_cent_max = np.array(centroids_cos[np.argmax(cos_rand_scores)])
cos_cent_sorted = cos_cent_max.argsort()[:, ::-1] 

# Get 20 words closest to each centroid

cos_centroid_terms = {}
for i in range(5):
    jlist = []
    for j in cos_cent_sorted[i,:20]:
        jlist.append((str(terms[j])).replace("'",""))
    cos_centroid_terms[i] = jlist
    
# Print Word Clouds for each cluster     

for i in cos_centroid_terms:
    wordcloud = WordCloud().generate(str(cos_centroid_terms[i]))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Cosine Distance Cluster " + str(i) + " Word Cloud")
    plt.show()


# Again TagClouds were produced using the top 20 terms closest to the centroids of each cluster, from the run with the highest ARI scores from the 50 random initializations.  As opposed to the Euclidean distance Clouds, in the case of the Cosine clouds it is clear which topic is related to which cluster (0: Athletics, 1: Rugby, 2: Cricket, 3: Tennis, 4: Football), with almost all words in all clusters being clearly related to the relevant topic.  This difference between the distance measures makes sense given the much higher ARI and AMI scores achieved using Cosine distance.

# # Part-2 (Dimensionality Reduction using PCA/SVD)

# In[4]:

# Import Packages

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# Normalize X
Xnorm = scale(X)



# In[ ]:

# Part 1 - Perform PCA with all components

Xpca = PCA()
Xpca.fit(Xnorm)


# In[105]:

# Plot Explained Variance

Xvar= Xpca.explained_variance_ratio_
X_cumvar=np.cumsum(np.round(Xpca.explained_variance_ratio_, decimals=4)*100)
plt.plot(X_cumvar)
plt.xlabel("Principal components")
plt.ylabel("Variance captured")
plt.title("Explained Variance by Number of Principal Components")
plt.show()


# In[123]:

# Print number of dimensions required for 95% and 98% variance capture

print ("Minimum Dimensions Required to Capture at least 95% of Variance: " + str(np.where(X_cumvar >= 95)[0][0]+1))
print ("\nMinimum Dimensions Required to Capture at least 98% of Variance: " + str(np.where(X_cumvar >= 98)[0][0]+1))


# The chart above shows that there is not a dramatic difference in the amount of variability captured by different principal components.  This is also evident in the fact that 555 out of 737 possible components are required to capture 95% of the variance.

# # Appendix - Clustering results after performing PCA

# In[181]:

# Check clustering with PCA n = 555

# Perform PCA with 554 Components

Xpca95 = PCA(n_components = 555)
x_95 = Xpca95.fit_transform(X)



# In[182]:

# Attempt k-means clustering using PCA dimensions, Euclidean distance and 5 clusters

kmeans95 = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=50, random_state=133)
kmeans95.fit_predict(x_95)
pred95 = np.array(kmeans95.labels_)
true_Labels = trueLabels.flatten()


# In[195]:

# Get averages over 20 random iterations
rand_scores95 = np.zeros((20))
mi_scores95 = np.zeros((20))
for i in range(0,20):
    kmeans95 = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=10)
    kmeans95.fit_predict(x_95)
    pred95 = np.array(kmeans95.labels_)
    rand_scores95[i] = (met.adjusted_rand_score(true_Labels, pred95))
    mi_scores95[i] = (met.adjusted_mutual_info_score(true_Labels, pred95))

#print("Adjusted Rand Index: " + str((met.adjusted_rand_score(true_Labels, pred95))))
#print("Adjusted MI Score: " + str((met.adjusted_mutual_info_score(true_Labels, pred95))))
print("Average Adjusted Rand Index: " + str(rand_scores95.mean()))
print("Adjusted Rand Index St Dev: " + str(rand_scores95.std()))
print("Average Adjusted MI Score: " + str(mi_scores95.mean()))


# In[187]:

# Attempt k-means clustering using PCA dimensions, Cosine distance and 5 clusters
cos_pca = KMeansClusterer(5, distance=cosine_distance, avoid_empty_clusters=True)
km_cos_pca = cos_pca.cluster(x_95, assign_clusters=True)


# In[188]:

# Calculate Adjusted Rand Score and MI Score
print("Adjusted Rand Index: " + str(met.adjusted_rand_score(true_Labels, km_cos_pca)))
print("Adjusted MI Score: " + str(met.adjusted_mutual_info_score(true_Labels, km_cos_pca)))


# In[191]:

# Get Averages over 20 random initializations
cos_rand95 = np.zeros((20))
cos_mi95 = np.zeros((20))
for i in range(0,20):
    km_cos95 = KMeansClusterer(5, distance=cosine_distance, avoid_empty_clusters=True)
    km_cos_95 = km_cos95.cluster(x_95, assign_clusters=True)
    cos_rand95[i] = (met.adjusted_rand_score(true_Labels, km_cos_95))
    cos_mi95[i] = (met.adjusted_mutual_info_score(true_Labels, km_cos_95))


# In[194]:

print("Average Adjusted Rand Index: " + str(cos_rand95.mean()))
print("Adjusted Rand Index St Dev: " + str(cos_rand95.std()))
print("Average Adjusted MI Score: " + str(cos_mi95.mean()))

