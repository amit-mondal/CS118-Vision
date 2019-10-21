import cv2
import numpy
#import timeit
import time
import classifiers
from sklearn import neighbors, svm, cluster
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import os

feature_detectors = {'sift' : cv2.xfeatures2d.SIFT_create(), 'orb' : cv2.ORB_create(), 'surf' : cv2.xfeatures2d.SURF_create(400)}

def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
    # to be zero-mean, and in the [-1, 1] range.
    output_image = cv2.normalize(cv2.resize(input_image, (target_size, target_size)),
                                 None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return output_image

def reportAccuracy(true_labels, predicted_labels, label_dict):
    # generates and returns the accuracy of a model
    # true_labels is a n x 1 cell array, where each entry is an integer
    # and n is the size of the testing set.
    # predicted_labels is a n x 1 cell array, where each entry is an 
    # integer, and n is the size of the testing set. these labels 
    # were produced by your system
    # label_dict is a 15x1 cell array where each entry is a string
    # containing the name of that category
    # accuracy is a scalar, defined in the spec (in %)

    num_correct = 0
    for true, pred in zip(true_labels, predicted_labels):
        if (true == pred):
            num_correct += 1
        
    return float(num_correct/predicted_labels.size)

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a n x 1 array of images
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be dict_size x d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.
    
    feature_detector = feature_detectors[feature_type]
    clustering = cluster.AgglomerativeClustering(dict_size) if clustering_type != 'kmeans' else cluster.KMeans(dict_size)

    descriptors = None
    for img in train_images:
        _, des = feature_detector.detectAndCompute(img, None)
        if descriptors is not None:
            np.append(descriptors, des, axis=0)
        else:
            descriptors = des

    clustering.fit(descriptors)
    return clustering.cluster_centers_

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    feature_detector = feature_detectors[feature_type]
    clf = KNeighborsClassifier(n_neighbors=1)
    labels = list(range(vocabulary.shape[0]))
    histogram = np.zeros(vocabulary.shape[0])
    
    clf.fit(vocabulary, labels)
    _, des = feature_detector.detectAndCompute(image, None)
    pred = clf.predict(des)

    for p in pred:
        histogram[p] += 1
    
    return histogram / np.sum(histogram)

def tinyImages(train_features, test_features, train_labels, test_labels, label_dict):
    # train_features is a nx1 array of images
    # test_features is a nx1 array of images
    # train_labels is a nx1 array of integers, containing the label values
    # test_labels is a nx1 array of integers, containing the label values
    # label_dict is a 15x1 array of strings, containing the names of the labels
    # classResult is a 18x1 array, containing accuracies and runtimes
    sizes = [8, 16, 32]
    
    classResult = []
    for neighbors in [1, 3, 6]:
        for size in sizes:
            train = np.array([imresize(img, size) for img in train_features]).reshape((len(train_features), size*size))
            test = np.array([imresize(img, size) for img in test_features]).reshape((len(test_features), size*size))
            start = time.clock()
            pred = classifiers.KNN_classifier(train, train_labels, test, neighbors)
            end = time.clock()
            classResult.append(reportAccuracy(test_labels, pred, label_dict))
            classResult.append(float(end - start))
    
    return classResult

def bowknn(train_features, test_features, train_labels, test_labels, label_dict):

    classResult = []    
    for neighbors in [1, 3, 6]:
        for dict_size in [50]:            
            for feature_type in ['sift', 'surf', 'orb']:
                saved_vocab = f'bow_vocab_{feature_type}_{dict_size}.obj'
                vocab = None
                if saved_vocab not in os.listdir():
                    vocab = buildDict(train_features, dict_size, feature_type, 'kmeans')
                    with open(saved_vocab, 'wb') as f:
                        pickle.dump(vocab, f)
                else:
                    with open(saved_vocab, 'rb') as f:
                        vocab = pickle.load(f)

                train = [computeBow(img, vocab, feature_type) for img in train_features]
                test = [computeBow(img, vocab, feature_type) for img in test_features]
                start = time.clock()
                pred = classifiers.KNN_classifier(train, train_labels, test, neighbors)
                end = time.clock()
                classResult.append(reportAccuracy(test_labels, pred, label_dict))
                classResult.append(float(end - start))
                print(f'accuracy: {classResult[-2]}, time: {classResult[-1]}')
    return classResult
