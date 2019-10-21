import os
import cv2
import numpy as np
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from timeit import default_timer as timer
import sys

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        if not os.path.isdir(test_path+label+'/'):
            continue
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels

def extract_histogram(arr):
    return arr.flatten()
    
def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    train_hists = [extract_histogram(train_feature) for train_feature in train_features]
    test_hists = [extract_histogram(test_feature) for test_feature in test_features]
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_hists, train_labels)
    predicted_categories = neigh.predict(test_hists)
    for i in range(len(predicted_categories)):
        predicted_categories[i] = predicted_categories[i]+1
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    resized = cv2.resize(input_image, (target_size, target_size))#.flatten()
    output_image = np.zeros((target_size, target_size))
    output_image = cv2.normalize(resized, output_image, -1, 1, norm_type=cv2.NORM_INF, dtype=cv2.CV_32F)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    total, total_right = 0, 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            total_right += 1
        total += 1
    accuracy = total_right / total
    return accuracy * 100


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    #vocabulary = []
    n = 10
    if feature_type == "sift":
        detector = cv2.xfeatures2d.SIFT_create()
    if feature_type == "surf":
        detector = cv2.xfeatures2d.SURF_create()
    if feature_type == "orb":
        detector = cv2.ORB_create(nfeatures=n)

    descriptors = []
    for i in range(len(train_images)):
        _, des = detector.detectAndCompute(train_images[i],None)
        if not des is None:
            descriptors.extend(des[:min(len(des), 5)])

    if clustering_type == "kmeans":
        clusterer = KMeans(n_clusters=dict_size, random_state=0)
    if clustering_type == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=dict_size)

    clusterer.fit(descriptors)
    
    if clustering_type == "kmeans":
        print(clusterer.cluster_centers_.shape)
        return clusterer.cluster_centers_
    if clustering_type == "hierarchical":
        centers = np.zeros((dict_size, len(descriptors[0])))  # dimensions of the feature array
        for i in range(0,dict_size):
            cluster_points = []
            for j in range(0, len(clusterer.labels_)):
                if clusterer.labels_[j] == i:
                    cluster_points.append(descriptors[j])
            #cluster_points = descriptors[ clusterer.labels_ == i ]
            cluster_mean = np.mean(cluster_points, axis=0)
            centers[i, :] = cluster_mean
        return centers

    
def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    if feature_type == "sift":
        detector = cv2.xfeatures2d.SIFT_create()
    if feature_type == "surf":
        detector = cv2.xfeatures2d.SURF_create()
    if feature_type == "orb":
        detector = cv2.ORB_create()
        
    _, des = detector.detectAndCompute(image,None)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(vocabulary, [i for i in range(len(vocabulary))])

    print("v", vocabulary.shape)
    print("d", des.shape)
    feature_labels = neigh.predict(des)

    Bow = np.zeros(len(vocabulary))
    hist = np.zeros(len(vocabulary))
    for i in feature_labels:
        hist[i] += 1

    s = np.sum(hist)
    Bow = [ i / s for i in hist ]
    #print(Bow)
    return Bow


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    classResult = []
    print("\t\t Acc, Time")
    for neighbors in [1, 3, 6]:
        for size in [8, 16, 32]:
            start = timer()
            test_predicted_labels = KNN_classifier([imresize(train_feature, size) for train_feature in train_features],
                                                   train_labels, [imresize(test_feature, size) for test_feature in test_features], neighbors)
            end = timer()
            accuracy = reportAccuracy(test_labels, test_predicted_labels)
            print(str(neighbors)+"x"+str(neighbors)+" "+str(size) +" neighbors:", accuracy, end-start)
            classResult.append(accuracy)
            classResult.append(end-start)
    return classResult
    
