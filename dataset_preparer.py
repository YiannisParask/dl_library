import tensorflow as tf


class DatasetPreparer:
    def __init__(self):
        self.data_augmentation = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        '''Create the data augmentation pipeline'''
        return tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ])

    def preprocess_image(self, image):
        '''Convert image to float32 and normalize'''
        return tf.image.convert_image_dtype(image, tf.float32) / 255.0

    def augment_and_preprocess(self, image, augment):
        '''Apply data augmentation and preprocess the image'''
        if augment:
            image = self.data_augmentation(image)
        return self.preprocess_image(image)

    def prepare_dataset(self, dataset, augment=False):
        '''Prepare a single dataset split for training'''
        dataset = dataset.map(lambda x, y: (self.augment_and_preprocess(x, augment), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def splits(dataset, TRAIN_RATIO=0.7, VAL_RATIO=0.2, TEST_RATIO=0.1):
        '''Split the dataset into train, validation, and test sets'''
        DATASET_SIZE = len(dataset)

        train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

        val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
        val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))  

        test_dataset = val_test_dataset.skip(int(TEST_RATIO*DATASET_SIZE))
        return train_dataset, val_dataset, test_dataset