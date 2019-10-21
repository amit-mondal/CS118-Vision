from classifiers import *
import utils
import os

# interpreting your performance with 100 training examples per category:
# accuracy  =   0 ->  your code is broken (probably not the classifier's
#                     fault! a classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .07 -> your performance is chance.
#  accuracy ~= .20 -> rough performance with tiny images and nearest
#                     neighbor classifier.
#  accuracy ~= .20 -> rough performance with tiny images and linear svm
#                     classifier. the linear classifiers will have a lot of
#                     trouble trying to separate the classes and may be
#                     unstable (e.g. everything classified to one category)
#  accuracy ~= .50 -> rough performance with bag of sift and nearest
#                     neighbor classifier.
#  accuracy ~= .60 -> you've gotten things roughly correct with bag of
#                     sift and a linear svm classifier.
#  accuracy >= .70 -> you've also tuned your parameters well. e.g. number
#                     of clusters, svm regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense sift features.
#  accuracy >= .80 -> you've added in spatial information somehow or you've
#                     added additional, complementary image features. this
#                     represents state of the art in lazebnik et al 2006.
#  accuracy >= .85 -> you've done extremely well. this is the state of the
#                     art in the 2010 sun database paper from fusing many 
#                     features. don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> you get to teach the class next year.
#  accuracy >= .96 -> you can beat a human at this task. this isn't a
#                     realistic number. some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), '../../data')
    test_path = os.path.join(data_path, 'test')
    train_path = os.path.join(data_path, 'train')

    test_features, train_features, train_labels, test_labels = [], [], [], []

    labels = [d for d in os.listdir(train_path) if not d.startswith('.')]
    label_dict = {}
    for i, label in enumerate(labels):
        test_img_path = os.path.join(test_path, label.lower())
        for f in [os.path.join(test_img_path, img) for img in os.listdir(test_img_path) if not img.startswith('.')]:
            test_features.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
            test_labels.append(i)

        train_img_path = os.path.join(train_path, label)
        for f in [os.path.join(train_img_path, img) for img in os.listdir(train_img_path) if not img.startswith('.')]:
            train_features.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
            train_labels.append(i)

        label_dict[i] = label

    
    utils.bowknn(train_features, test_features, train_labels, test_labels, label_dict)
    
    
