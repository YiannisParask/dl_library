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
    