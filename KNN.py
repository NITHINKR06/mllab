import numpy as np
from math import sqrt
from collections import Counter

X = np.array([
    [2, 150],
    [3, 200],
    [5, 250],
    [6, 300],
    [7, 350]
])

y = np.array([0, 0, 1, 1, 1])
k = 3 

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)

def knn_classify(train_data, train_labels, test_point, k):
    distances = []
    
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_point)
        distances.append((dist, train_labels[i]))
        
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    class_labels = [label for (_, label) in k_nearest]
    majority_class = Counter(class_labels).most_common(1)[0][0]
    
    return majority_class

new_traffic = [
    [3, 180],
    [4, 230],
    [6, 320]
]

print("KNN Classification Results:\n")

for traffic in new_traffic:
    result = knn_classify(X, y, traffic, k)
    
    if result == 0:
        print("Traffic", traffic, "NORMAL")
    else:
        print("Traffic", traffic, "MALICIOUS")
