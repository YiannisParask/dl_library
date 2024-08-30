import tensorflow as tf
import fiftyone as fo
import json
import os
from tensorflow.keras.utils import to_categorical
import numpy as np

# TODO: Handle case where labels are already in integer format
class LoadFiftyoneDataset:
    ''' Class to load and preprocess Fiftyone data for training and inference. '''
    def __init__(self, dataset_dir, labels_dir):
        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.class_to_index = {}

    def load_fo_dataset(self, dataset_name):
        """Imports annotations into a FiftyOne dataset and returns it.
        Args:
            dataset_dir: The directory containing the dataset.
            dataset_name: The name of the dataset.
        Returns:
            dataset: The FiftyOne dataset.
        Notes: Directory structure should be as follows:
            dataset_dir
            ├── data
            │   ├── image_1.png
            │   ├── image_2.png
            │   └── ...
        """
        if dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(dataset_name)
        else:
            dataset = fo.Dataset.from_dir(
                dataset_dir=self.dataset_dir,
                dataset_type=fo.types.ImageClassificationDirectoryTree,
                name=dataset_name,
            )
        return dataset
    
    def load_labels_from_json(self):
        '''Load labels from a JSON file'''
        with open(self.labels_dir, "r") as file:
            labels_data = json.load(file)
        label_mappings = labels_data["labels"] 
        return label_mappings
    
    def load_labels_from_fiftyone(self, file_paths, label_mappings):
        '''Load labels from a FiftyOne dataset'''
        labels = []
        missing_labels = []
        for fp in file_paths:
            filename = os.path.splitext(os.path.basename(fp))[0]
            if filename in label_mappings:
                labels.append(label_mappings[filename][0])
            else:
                missing_labels.append(filename)
                labels.append("unknown")

        if missing_labels:
            print(f"Missing labels for the following files: {missing_labels}")

        # Convert categorical labels to indices
        unique_labels = set(labels)
        self.class_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        # Convert labels to integer indices
        labels = [self.class_to_index[label] for label in labels]
        
        # Filter out samples with missing labels
        valid_indices = [i for i, label in enumerate(labels) if label != -1]
        file_paths = [file_paths[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        return file_paths, labels

    def load_and_preprocess_image(self, file_path, image_size=(224, 224), is_rgb=False):
        """Loads and preprocesses an image for inference using TensorFlow."""
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size)
        if not is_rgb:
            image = tf.image.rgb_to_grayscale(image)
        image = image / 255.0
        return image

    def fiftyone_to_tf_dataset(self, fo_dataset, image_size=(224, 224), batch_size=32, is_rgb=False):
        file_paths = fo_dataset.values("filepath")
        label_mappings = self.load_labels_from_json()
        file_paths, labels = self.load_labels_from_fiftyone(file_paths, label_mappings) 
        num_classes = len(self.class_to_index)
        labels = to_categorical(labels, num_classes=num_classes)

        def load_image_and_label(file_path, label):
            image = self.load_and_preprocess_image(file_path, image_size, is_rgb)
            return image, label

        tf_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        tf_dataset = tf_dataset.map(
            lambda file_path, label: load_image_and_label(file_path, label),  
            num_parallel_calls=tf.data.AUTOTUNE
        )
        tf_dataset = tf_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return tf_dataset
    
    def get_samples_labels(self, train, test):
        # extract the images and labels for each dataset
        train_images = np.concatenate(list(train.map(lambda x, y: x)))
        train_labels = np.concatenate(list(train.map(lambda x, y: y)))
        
        test_images = np.concatenate(list(test.map(lambda x, y: x)))
        test_labels = np.concatenate(list(test.map(lambda x, y: y)))
        
        # join them together
        inputs = np.concatenate((train_images, test_images), axis = 0)
        targets = np.concatenate((train_labels, test_labels), axis = 0)
        
        return inputs, targets

    def splits(dataset, TRAIN_RATIO=0.7, VAL_RATIO=0.2, TEST_RATIO=0.1):
        '''Split the dataset into train, validation, and test sets'''
        DATASET_SIZE = len(dataset)

        train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

        val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
        val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))  

        test_dataset = val_test_dataset.skip(int(TEST_RATIO*DATASET_SIZE))
        return train_dataset, val_dataset, test_dataset