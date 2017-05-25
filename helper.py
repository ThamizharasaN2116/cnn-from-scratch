import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from random import shuffle
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from PIL import Image
import sys
import shutil


def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def _load_label_names_model():
    """
    Load the label names from file
    """
    return ['apple', 'banana', 'broccoli', 'carrot', 'onion', 'pineapple', 'pumpkin']


def _load_label_name_100():
    """
    Load the label names from file
    """
    return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # picked from 
    # import pickle
    # with open(cifar100_dataset_folder_path+"/meta", 'rb') as fo:
    #     dict = pickle.load(fo, encoding='latin')
    # print(dict['fine_label_names'])
    #return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']



def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def load_model_data():
    """
    Load a batch of the dataset
    """
    with open('data.pkl', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['train']
    labels = batch['train_labels']

    return features, labels

def load_cfar100_batch(cifar10_dataset_folder_path):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/train', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['fine_labels']
    
    return features, labels

def display_stats(sample_id):
    """
    Display Stats of the the dataset
    """
    # label_names = None
    # if cifar10_dataset_folder_path == 'cifar-100-python':
    #     print('x')
    #     features, labels = load_cfar100_batch(cifar10_dataset_folder_path)
    #     label_names = _load_label_name_100()
    # else:
    #     batch_ids = list(range(1, 6))
    #     if batch_id not in batch_ids:
    #         print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
    #         return None
    #     features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
    #     label_names = _load_label_names()
    features, labels = load_model_data()
    label_names = _load_label_names_model()
    if not (0 <= sample_id < len(features)):
        print('{}   {} is out of range.'.format(len(features), sample_id))
        return None

    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    #print(labels)
    print('sample label',sample_label)

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.axis('off')
    plt.imshow(sample_image)


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'), protocol=4)


def preprocess_and_save_data_10(cifar10_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test.p')

def preprocess_and_save_data_100(cifar100_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    features, labels = load_cfar100_batch(cifar100_dataset_folder_path)
    validation_count = int(len(features) * 0.1)

    # Prprocess and save a batch of training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        features[:-validation_count],
        labels[:-validation_count],
        'preprocess_train_100.p')

    # Use a portion of training batch for validation
    valid_features.extend(features[-validation_count:])
    valid_labels.extend(labels[-validation_count:])
        

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation_100.p')

    with open(cifar100_dataset_folder_path + '/test', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['fine_labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test_100.p')

def preprocess_and_save_data_model(cifar100_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """

    valid_features = []
    valid_labels = []
    
    features, labels = load_model_data()
    validation_count = int(len(features) * 0.1)

    Features =  features[:-validation_count]
    Labels = labels[:-validation_count]

    splitCount = int(len(tFeatures)/2)

    # Prprocess and save a batch of training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        Features[:-splitCount],
        Labels[:-splitCount],
        'preprocess_train_model_20_1.p')
    
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        tFeatures[-splitCount:],
        tLabels[-splitCount:],
        'preprocess_train_model_20_2.p')

    # Use a portion of training batch for validation
    valid_features.extend(features[-validation_count:])
    valid_labels.extend(labels[-validation_count:])
        

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation_model_20.p')

    with open('data.pkl', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['test']
    test_labels = batch['test_labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test_model_20.p')

def preprocess_and_save_data(folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    if folder_path == 'cifar-10-batches-py':
        preprocess_and_save_data_10(folder_path, normalize, one_hot_encode)
    else:
        preprocess_and_save_data_model(folder_path, normalize, one_hot_encode)

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def preprocess_validation_batch(batch_size):
    """
    Load the Preprocessed validation data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_validation_model_20.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    return batch_features_labels(features, labels, batch_size)

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

def load_preprocess_training_model(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_train_model_20_'+ str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

def load_preprocess_training_100(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_train_100.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

def display_image_predictions(features, labels, predictions):
    n_classes = 1000
    label_names = _load_label_names_model()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(nrows=10, ncols=2, figsize=(15,15))
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions
    print('features: ',len(features), ' lab: ', len(labels), ' pred ', len(predictions))
    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]
        print('values: ',pred_values[::-1])
        print('names: ',pred_names[::-1])
        axies[image_i][0].imshow(feature)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])

def resize_images(img_path, conv_path, data_aug=False):
    """
    Resize images with 224*224*3
    """

    if not os.path.exists(conv_path):
        os.makedirs(conv_path)
        print("\nNew Directory created")
    else:
        shutil.rmtree(conv_path)
        os.makedirs(conv_path)
        print("\nNew Directory created")
    onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    print("Total files in input dir: {}".format(len(onlyfiles)))
    for i, f in enumerate(onlyfiles):
        #print(i)
        #print(f)
        img = Image.open(img_path+'/'+f)
        img = img.resize((69,69), Image.ANTIALIAS)
        newPath = conv_path+'/'+str(i)+'.jpeg'
        sys.stdout.write("\r {}".format(newPath))
        sys.stdout.flush()
        img.save(newPath)
        im = cv2.imread(newPath)
        if data_aug:
        # copy image to display all 4 variations
            horizontal_img = im.copy()
            vertical_img = im.copy()
            both_img = im.copy()

            # flip img horizontally, vertically,
            # and both axes with flip()
            horizontal_img = cv2.flip( im, 0 )
            vertical_img = cv2.flip( im, 1 )
            both_img = cv2.flip( im, -1 )
            
            cv2.imwrite(conv_path+'/'+str(i)+'_1.jpeg',horizontal_img) 
            cv2.imwrite(conv_path+'/'+str(i)+'_2.jpeg',vertical_img) 
            cv2.imwrite(conv_path+'/'+str(i)+'_3.jpeg',both_img) 
        if(im.shape[2] != 3):
            print(im.shape)

def convert_image_to_numpy(img_path):
    """
    Convert Images to numpy array
    """
    img_path = './converted_test'
    onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path,f))]
    print(len(onlyfiles))
    data = []
    label = []
    rgb = []
    for i, f in enumerate(onlyfiles):
        img = cv2.imread(img_path+'/'+f)
        b,g,r = cv2.split(img)
        img2 = cv2.merge([r,g,b])
        rgb.append(img2)
        data.append(img)
        label.append(0)
    data = rgb
    return data, label

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    #x = np.divide(x,np.max(x, axis=0))
    return x/np.max(x, axis=0)
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    #print(x)
    oneHot = None
    oneHot = np.zeros((len(x), 7))
    for idx, v in enumerate(x):
        oneHot[idx][v] = 1
    return oneHot
    
def preprocess_and_save_test_data(f, l):
    """
    Preprocess Training and Validation Data
    """

    valid_features = []
    valid_labels = []
    
    
    # load the test data
    test_features = f
    test_labels = l

    # Preprocess and Save all test data
    # test_features = normalize(f)
    # test_labels = one_hot_encode(l)
    # print(len(test_features))
    # print(len(test_labels))
    # pickle.dump((test_features, test_labels), open('preprocess_test_model_20.p', 'wb'), protocol=4)


    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test_model_20.p')


def preprocess_and_save_test_data():
    with open('new_data.pkl', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['train']
    test_labels = batch['train_labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test_model_new.p')

