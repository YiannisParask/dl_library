import tensorflow as tf
import fiftyone as fo
import json
import os
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# TODO: Handle case where labels are already in integer format
class LoadFiftyoneDataset:
    ''' Class to load and preprocess Fiftyone data for training and inference. '''
    def __init__(self, dataset_dir, labels_dir):
        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir


    def load_fo_dataset(self, dataset_name):
        """Imports annotations into a FiftyOne dataset and returns it.
        Args:
            dataset_dir: The directory containing the dataset.
            dataset_name: The name of the dataset.
        Returns:
            dataset: The FiftyOne dataset.
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
        '''
        Load labels from Fiftyone dataset
        Args:
            file_paths: List of file paths
            label_mappings: Dictionary containing label mappings
        Returns:
            file_paths: List of file paths
            labels: List of labels
        Note:
            Labels are expected to be numeric!
        '''
        labels = []
        missing_labels = []
        for fp in file_paths:
            filename = os.path.splitext(os.path.basename(fp))[0]
            if filename in label_mappings:
                labels.append(label_mappings[filename])
            else:
                missing_labels.append(filename)
                labels.append([])

        if missing_labels:
            print(f"Missing labels for the following files: {missing_labels}")

        # Filter out samples with missing labels (if exist)
        valid_indices = [i for i, label in enumerate(labels) if len(label) > 0]
        file_paths = [file_paths[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        return file_paths, labels

    def load_image_and_label(self, file_path, label, image_size=(224, 224), is_rgb=False):
        '''
        Load and preprocess image.
        Args:
            file_path: String
            image_size: Tuple of image dimensions
            is_rgb: Boolean
        Returns:
            image: Preprocessed image
        '''
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size)
        if not is_rgb:
            image = tf.image.rgb_to_grayscale(image)
        image = image / 255.0
        return image, label


    def fiftyone_to_tf_dataset(self, fo_dataset, image_size=(224, 224), batch_size=32, is_rgb=False):
        '''
        Convert Fiftyone dataset to TensorFlow dataset
        Args:
            fo_dataset: Fiftyone dataset
            image_size: Tuple of image dimensions
            batch_size: Integer
            is_rgb: Boolean
        Returns:
            tf_dataset: TensorFlow dataset
        '''
        file_paths = fo_dataset.values("filepath")
        label_mappings = self.load_labels_from_json()
        file_paths, labels = self.load_labels_from_fiftyone(file_paths, label_mappings)

        labels = to_categorical(labels)
        # mlb = MultiLabelBinarizer()
        # labels = mlb.fit_transform(labels)

        tf_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        tf_dataset = tf_dataset.map(
            lambda file_path, label: self.load_image_and_label(file_path, label, image_size, is_rgb),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        tf_dataset = tf_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return tf_dataset


    def get_samples_labels(self, train, test, valid=None):
        '''
        Get samples and labels from TensorFlow datasets
        Args:
            train: TensorFlow dataset
            test: TensorFlow dataset
        Returns:
            inputs: Numpy array of images
            targets: Numpy array of labels
        '''
        # extract the images and labels for each dataset
        train_images = np.concatenate(list(train.map(lambda x, y: x)))
        train_labels = np.concatenate(list(train.map(lambda x, y: y)))

        test_images = np.concatenate(list(test.map(lambda x, y: x)))
        test_labels = np.concatenate(list(test.map(lambda x, y: y)))

        if valid is not None:
            valid_images = np.concatenate(list(valid.map(lambda x, y: x)))
            valid_labels = np.concatenate(list(valid.map(lambda x, y: y)))
            inputs = np.concatenate((train_images, valid_images, test_images), axis = 0)
            targets = np.concatenate((train_labels, valid_labels, test_labels), axis = 0)
            return inputs, targets

        # join them together
        inputs = np.concatenate((train_images, test_images), axis = 0)
        targets = np.concatenate((train_labels, test_labels), axis = 0)

        return inputs, targets


    def splits(self, dataset, TRAIN_RATIO=0.7, VAL_RATIO=0.2, TEST_RATIO=0.1):
        '''
        Split dataset into training, validation, and test sets.
        Args:
            dataset: TensorFlow dataset
            TRAIN_RATIO: Float
            VAL_RATIO: Float
            TEST_RATIO: Float
        Returns:
            train_dataset: TensorFlow dataset
            val_dataset: TensorFlow dataset
            test_dataset: TensorFlow dataset
        '''
        DATASET_SIZE = len(dataset)

        train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

        val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
        val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))

        test_dataset = val_test_dataset.skip(int(TEST_RATIO*DATASET_SIZE))
        return train_dataset, val_dataset, test_dataset
