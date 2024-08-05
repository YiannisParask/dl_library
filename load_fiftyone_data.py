import tensorflow as tf
import fiftyone as fo


class LoadFiftyoneDataset:
    ''' Class to load and preprocess Fiftyone data for training and inference. '''
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

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
        # Check if the dataset name exists, load it if it does
        if dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(dataset_name)
        else:
            # Create a new FiftyOne dataset if it doesn't exist
            dataset = fo.Dataset.from_dir(
                dataset_dir=self.dataset_dir,
                dataset_type=fo.types.ImageClassificationDirectoryTree,
                name=dataset_name,
            )
        return dataset

    def load_and_preprocess_image(self, file_path, image_size=(224, 224), is_rgb=False):
        """Loads and preprocesses an image for inference using TensorFlow."""
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size)
        if not is_rgb:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def fiftyone_to_tf_dataset(self, fo_dataset_view, image_size=(224, 224), batch_size=32, is_rgb=False):
        """Converts a FiftyOne dataset view to a TensorFlow Dataset (tf.data.Dataset)."""
        file_paths = fo_dataset_view.values("filepath")
        labels = fo_dataset_view.values("label")
        file_paths = [str(fp) for fp in file_paths]

        def load_image_and_label(file_path, label):
            image = self.load_and_preprocess_image(file_path.numpy().decode(), image_size, is_rgb)
            return image, label

        tf_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        tf_dataset = tf_dataset.map(
            lambda file_path, label: tf.py_function(
                func=load_image_and_label,
                inp=[file_path, label],
                Tout=[tf.float32, tf.string],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        tf_dataset = tf_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return tf_dataset

    def splits(dataset, TRAIN_RATIO=0.7, VAL_RATIO=0.2, TEST_RATIO=0.1):
        '''Split the dataset into train, validation, and test sets'''
        DATASET_SIZE = len(dataset)

        train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

        val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
        val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))  

        test_dataset = val_test_dataset.skip(int(TEST_RATIO*DATASET_SIZE))
        return train_dataset, val_dataset, test_dataset