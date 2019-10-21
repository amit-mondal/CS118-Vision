from utils import *
import argparse
import time

parser = argparse.ArgumentParser(description='CS188.2 - Fall 19 - Homework 1')
parser.add_argument("--tiny", "-t", type=bool, default=False, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
parser.add_argument("--make-vocab", "-mv", type=bool, default=False, help='generate vocab files')
parser.add_argument("--make-bow", "-mb", type=bool, default=False, help='generate bow files')
args = parser.parse_args()

print(args)

# The argument is included as an idea for debugging, with a few examples in the main. Feel free to modify it or add arguments.
# You are also welcome to disregard this entirely

#############################################################################################################################
# This file contains the main. All the functions you need to write are included in utils. You also need to edit the main.
# The main just gets you started with the data and highlights the high level structure of the project.
# You are free to modify it as you wish - the modifications you are required to make have been marked but you are free to make
# others.
# What you cannot modify is the number of files you have to save or their names. All the save calls are done for you, you
# just need to specify the right data.
#############################################################################################################################


if __name__ == "__main__":
    
    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results') 
        SAVEPATH = 'Results/'
    
    # Load data, the function is written for you in utils
    train_images, test_images, train_labels, test_labels = load_data()
    
    if args.tiny:
        # You have to write the tinyImages function
        tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
    
        # Split accuracies and runtimes for saving  
        for element in tinyRes[::2]:
            # Check that every second element is an accuracy in reasonable bounds
            pass
            #assert (7 < element and element < 20)            
        acc = np.asarray(tinyRes[::2])
        runtime = np.asarray(tinyRes[1::2])
    
        # Save results
        np.save(SAVEPATH + 'tiny_acc.npy', acc)
        np.save(SAVEPATH + 'tiny_time.npy', runtime)

    # Create vocabularies, and save them in the result directory
    # You need to write the buildDict function
    vocabularies = []
    vocab_idx = [] # If you have doubts on which index is mapped to which vocabulary, this is referenced here
    # e.g vocab_idx[i] will tell you which algorithms/neighbors were used to compute vocabulary i
    # This isn't used in the rest of the code so you can feel free to ignore it

    #TODO: GET HIERARCHICAL AND ORB WORKING
    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                if args.make_vocab:
                    vocabulary = buildDict(train_images, dict_size, feature, algo)
                    np.save(SAVEPATH + filename, np.asarray(vocabulary))
                else:
                    vocabulary = np.load(SAVEPATH + filename)
                vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary
                
    # Compute the Bow representation for the training and testing sets
    test_rep = [] # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = [] # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4 # Order in which features were used 
    # for vocabulary generation
    
    # You need to write ComputeBow()
    if args.make_bow:
        for i, vocab in enumerate(vocabularies):
            print(f'working on {vocab_idx[i]}, feature should be {features[i]}')
            for image in train_images: # Compute the BOW representation of the training set
                rep = computeBow(image, vocab, features[i]) # Rep is a list of descriptors for a given image
                train_rep.append(rep)
            np.save(SAVEPATH + 'bow_train_' + str(i) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i
            train_rep = [] # reset the list to save the following vocabulary
            for image in test_images: # Compute the BOW representation of the testing set
                rep = computeBow(image, vocab, features[i])
                test_rep.append(rep)
            np.save(SAVEPATH + 'bow_test_' + str(i) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i
            test_rep = [] # reset the list to save the following vocabulary

    test_reps, train_reps = {}, {}
    for i in range(len(vocabularies)):
        test_reps[i] = np.load(f'{SAVEPATH}bow_test_{i}.npy')
        train_reps[i] = np.load(f'{SAVEPATH}bow_train_{i}.npy')
    
    # Use BOW features to classify the images with a KNN classifier
    # A list to store the accuracies and one for runtimes
    knn_accuracies = []
    knn_runtimes = []    

    # Your code below, eg:
    # for i, vocab in enumerate(vocabularies):
    # ...
    
    for i, vocab in enumerate(vocabularies):
        test_rep, train_rep = test_reps[i], train_reps[i]
        start = time.process_time()
        print(f'train {train_rep.shape}, labels {np.array(train_labels).shape}, test {test_rep.shape}')
        pred = KNN_classifier(train_rep, train_labels, test_rep, 9)
        end = time.process_time()
        knn_accuracies.append(reportAccuracy(test_labels, pred))
        knn_runtimes.append(float(end - start))
        print(f'dict size {dict_size}, feature {features[i]}, vocab {vocab_idx[i]}, accuracy {knn_accuracies[-1]}, time {knn_runtimes[-1]}')
    
    np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory
    
    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    
    # Your code below
    #...

    np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory
    
    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []
    
    # Your code below
    # ...
    
    np.save(SAVEPATH +'rbf_accuracies.npy', np.asarray(rbf_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH +'rbf_runtimes.npy', np.asarray(rbf_runtimes)) # Save the runtimes in the Results/ directory
            
    
