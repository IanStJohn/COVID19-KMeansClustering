# Written By:   Ian St. John
# Last Edited:  December, 13th, 2020
# Class:        CS-484 Data Mining
# Institute:    George Mason University

# K-means Clustering
# 1.    Select K points as the initial centroids.
# 2.    repeat
# 3.        Form K clusters by assigning all points to the closest centroid.
# 4.        Recompute the centroid of each cluster.
# 5.    until The centroids don't change.

import csv
import os
from random import randint
from datetime import datetime

# Helper function to read in .csv files.
def read_file(inFilePath):
    data = []
    with open(inFilePath, 'r') as inFile:
        reader = csv.reader(inFile, delimiter=',')
        for row in reader: data.append([x for x in row])
    return data

# Feature picking helper function.
def pick_features(data, features):
    if any(isinstance(i, list) for i in data): return [[row[i] for i in features] for row in data]
    else: return [data[i] for i in features]

# Euclidean Distance Calculation
def distance(X,Y):
    if(len(X) == len(Y)): return (sum([(float(x)-float(y))*(float(x)-float(y)) for x,y in zip(X,Y)]))**.5
    else: return -1

# Calculate closest centroid given a data-point.
def closest_centroid(d, centroids, data):
    closest = 0
    for i,centroid in enumerate(centroids[1:]): 
        if(distance(data[centroid],d) < distance(data[centroids[closest]],d)): closest = i+1
    return closest

# Calculate the closest data-point given a centroid.
def closest_data(mean, data):
    closest = 0
    for i,d in enumerate(data[1:]):
        if(distance(d, mean) < distance(data[closest], mean)): closest = i+1
    return closest

# Helper function to computer the centroids for k-means clustering.
def compute_centroids(clusters, centroids, data):
    k = len(centroids)
    means = [[0 for j in range(len(data[0]))] for i in range(k)]
    cnt = [0 for i in range(k)]

    # Accumulate sums and counters.
    for i,d in enumerate(data):
        cnt[clusters[i]] += 1
        for j,num in enumerate(d): 
            means[clusters[i]][j] += float(num)

    # Produce average by dividing the sums by the counters.
    for i,centroid in enumerate(means):
        for j,num in enumerate(centroid):
            if(cnt[i] != 0): means[i][j] = num / cnt[i]

    # Match averages to closest data points.
    for i in range(k):
        centroids[i] = closest_data(means[i], data)
    
    return centroids

# Main k-means clustering function.
def k_means_clustering(k, data):
    centroids = [randint(0,len(data)-1)for n in range(k)]
    while(len(set(centroids)) != k): centroids = [randint(0,len(data)-1)for n in range(k)]

    # Repeat until the centroids don't change.
    changed = True
    while(changed):
        old_centroids = centroids.copy()

        # Form K clusters by assigning all points to the closest centroid.
        clusters = [closest_centroid(d, centroids, data) for d in data]

        # Recompute the centroid of each cluster.
        centroids = compute_centroids(clusters, centroids, data)

        # Check if the centroids changed.
        changed = len([x for x in set(centroids) if x not in set(old_centroids)]) != 0
    
    return clusters

# Break main data-set down day by day and save out.
def preprocess(inFilePath):
    file = read_file(inFilePath)

    headers = file[0]
    data = file[1:]

    include_features = {x for x in range(0, 50)}
    exclude_features = {0, 1, 3, 33}
    features = [feature for feature in include_features if feature not in exclude_features]
    headers = [headers[i] for i in features]

    exclude_features = {0, 1, 33}
    features = [feature for feature in include_features if feature not in exclude_features]
    data = pick_features(data, features)

    days = [date[0] for date in read_file("src/data/_days.csv")[1:]]
    dayDict = {day:[[data[i][j] for j in range(len(features)) if j not in {1}] for i in range(len(data)) if data[i][1] == day] for day in days}
    for day in days:
        outFilePath = os.path.join("src", "data", day.replace("/","_") + ".csv")
        outData = dayDict.get(day)
        with open(outFilePath, 'w', newline='') as outFile:
            csvWriter = csv.writer(outFile, delimiter=',')
            csvWriter.writerow(headers)
            csvWriter.writerows(outData)

# Helper function to get k-means cluster given specified day.
def clusterDay(day, features, k):
    inFilePath = os.path.join("src", "data", day.replace("/","_") + ".csv")
    file = read_file(inFilePath)

    headers = [file[0][i] for i in range(len(file[0])) if i not in {0}]
    classes = [row[0] for row in file[1:]]
    data = [[row[i] for i in range(len(row)) if i not in {0}] for row in file[1:]]

    headers = pick_features(headers, features)
    data = pick_features(data, features)

    return k_means_clustering(k, data), classes, data
    

# MAIN PROGRAM ----------------------------------------------------------------------------------
# preprocess("src/data/_owid-covid-data.csv")

k_set = [i for i in range(3,11)]
features_set = [{7,10,12,14,16,24,27,28,29}, {7,10,24}, {7,10,14}, {7,14,16}, {24,28,29}, {12,27,29}, {7,10,14,16}, {7,14,12,24,29}]

feature_names_set = [feature for feature in read_file("src/data/_feature_names.csv")][0]
days = [date[0] for date in read_file("src/data/_days.csv")[1:]]

# Ensure directory exists.
runDir = datetime.now().strftime("%d.%m.%Y_%H.%M")
newpath = os.path.join(os.getcwd(), "src", "out", runDir)
if not os.path.exists(newpath):
    os.makedirs(newpath)

outFilePath = os.path.join("src", "out", runDir, "_metrics" + ".csv")
with open(outFilePath, 'w', newline='') as outFile:
    csvWriter = csv.writer(outFile, delimiter=',')
    csvWriter.writerow(["k_set",k_set])
    for f,features in enumerate(features_set):
        row = [name for i,name in enumerate(feature_names_set) if i in features]
        row.insert(0,"f"+str(f))   # Add headers.
        csvWriter.writerow(row)

for k in k_set:
    for f,features in enumerate(features_set):
        feature_names = [name for i,name in enumerate(feature_names_set) if i in features]
        
        # Ensure directory exists.
        newpath = os.path.join(os.getcwd(), "src", "out", runDir, "k" + str(k), "f" + str(f))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        nationDict = {}

        for day in days: 
            # Run helper function to perform k-means clustering on specified day.
            clusters, classes, fileData = clusterDay(day, features, k)

            fileData.insert(0, feature_names)   # Add headers.
            classes.insert(0, "country")        # Add headers.
            clusters.insert(0, "cluster")       # Add headers.
            data = [[country, cluster] for (country,cluster) in zip(classes, clusters)]
            data = [[cc[0], cc[1], fd[0] ,fd[1], fd[2]] for (cc,fd) in zip(data, fileData)]

            # Ensure directory exists.
            outFilePath = os.path.join("src", "out", runDir, "k" + str(k), "f" + str(f), "days", day.replace("/","_") + ".csv")
            newpath = os.path.join(os.getcwd(), "src", "out", runDir, "k" + str(k), "f" + str(f), "days")
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            
            # Write out country name, cluster, and the first 3 features scores.
            # The 3 features were originally used for graphing, now they are vestigial.
            with open(outFilePath, 'w', newline='') as outFile:
                csvWriter = csv.writer(outFile, delimiter=',')
                csvWriter.writerows(data)

            # Tally up nations clustered together.
            for i,nation in enumerate(classes[1:]):
                if nation not in nationDict: nationDict[nation] = {}
                clusteredNations = [country for country,cluster in zip(classes[1:],clusters[1:]) if cluster == clusters[i+1] and country != nation]
                for cNation in clusteredNations:
                    if cNation not in nationDict[nation]: nationDict[nation][cNation] = 1
                    else: nationDict[nation][cNation] += 1

        # Print out different .csv files for each nation to show similarity scores.
        for nation in nationDict:
            data = list(nationDict[nation].items())
            data.insert(0, ("Nation", "Similarity Score"))  # Add headers.

            # Ensure directory exists.
            newpath = os.path.join(os.getcwd(), "src", "out", runDir, "k" + str(k), "f" + str(f), "nations")
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            outFilePath = os.path.join("src", "out", runDir, "k" + str(k), "f" + str(f), "nations", nation + ".csv")
            with open(outFilePath, 'w', newline='') as outFile:
                    csvWriter = csv.writer(outFile, delimiter=',')
                    csvWriter.writerows(data)