import os
import cv2
import numpy as np
import timeit, time
import random
import gc
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.neighbors import KNeighborsClassifier


MAX_FEATURES = 40

NUM_FEATURES = {
    'hierarchical' : 25,
    'kmeans' : MAX_FEATURES
}

fd_cache = {
    'sift' : {},
    'surf' : {},
    'orb' : {},
}


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')], key=lambda s: s.upper())
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
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


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
    clf = KNeighborsClassifier(n_neighbors=num_neighbors)
    clf.fit(train_features, train_labels)
    predicted_categories = clf.predict(test_features)
    
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
    output_image = cv2.normalize(cv2.resize(input_image, (target_size, target_size)),
                                 None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    num_correct = 0
    for true, pred in zip(true_labels, predicted_labels):
        if (true == pred):
            num_correct += 1
    accuracy = float(num_correct/predicted_labels.size) * 100
    return accuracy

def getDescriptors(feature_type, img, nfeatures):
    feature_detectors = {'sift' : cv2.xfeatures2d.SIFT_create(nfeatures=MAX_FEATURES), 'orb' : cv2.ORB_create(nfeatures=MAX_FEATURES), 'surf' : cv2.xfeatures2d.SURF_create()}
    feature_detector = feature_detectors[feature_type]    
    _, des = feature_detector.detectAndCompute(img, None)
    if des is not None and des.shape[0] > nfeatures:
        #return des[np.random.choice(des.shape[0], nfeatures, replace=False)]
        return des[:nfeatures]
    return des

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
    clustering = cluster.AgglomerativeClustering(n_clusters=dict_size, memory='./cache', linkage='single') if clustering_type != 'kmeans' else cluster.KMeans(dict_size)

    print(f'building dict for {dict_size} {feature_type} {clustering_type}')

    if clustering_type not in fd_cache[feature_type]:
        descriptors = []
        for img in train_images:
            des = getDescriptors(feature_type, img, NUM_FEATURES[clustering_type])
            if des is None:
                continue
            descriptors.extend(des)
        fd_cache[feature_type][clustering_type] = descriptors
    else:
        descriptors = fd_cache[feature_type][clustering_type]
        
    clustering.fit(descriptors)

    if clustering_type != 'kmeans':
        print('start')
        descriptors = np.array(descriptors)
        centers = []
        labels = np.unique(clustering.labels_)
        for label in labels:
            indices = [i for i, v in enumerate(clustering.labels_) if v == label]
            centers.append(np.mean(descriptors[indices], axis=0))
        return np.array(centers)
    
    return clustering.cluster_centers_


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    clf = KNeighborsClassifier(n_neighbors=1)
    labels = list(range(vocabulary.shape[0]))
    histogram = np.zeros(vocabulary.shape[0])

    
    clf.fit(vocabulary, labels)
    des = getDescriptors(feature_type, image, NUM_FEATURES['kmeans'])
    
    if des is None:
        return histogram
    
    pred = clf.predict(des)        
    for p in pred:
        histogram[p] += 1
    
    return histogram / np.sum(histogram)


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
    sizes = [8, 16, 32]
    
    classResult = []
    for neighbors in [1, 3, 6]:
        for size in sizes:
            train = np.array([imresize(img, size) for img in train_features]).reshape((len(train_features), size*size))
            test = np.array([imresize(img, size) for img in test_features]).reshape((len(test_features), size*size))
            start = time.clock()
            pred = KNN_classifier(train, train_labels, test, neighbors)
            end = time.clock()
            classResult.append(reportAccuracy(test_labels, pred))
            classResult.append(float(end - start))
        
    return classResult
    
