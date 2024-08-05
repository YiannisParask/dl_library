import os
import numpy as np
from keras.models import load_model
import json
import fiftyone as fo
import tensorflow as tf


class FoDatasetInference:
    '''
    Class to perform inference on a FiftyOne dataset using a Keras model.
    
    Usage:
    model_path = "best_custom_model.keras"
    dataset_dir = "Datasets/mlt-gbl-640-dataset"
    dataset_name = "mlt-gbl-640-dataset"
    class_names = ["Lathe", "Milling Machine", "Spring", "Sheet Metal"]

    inference = FoDatasetInference(model_path, dataset_dir, dataset_name, class_names)
    inference.execute()
    '''
    def __init__(self, model_path, dataset_dir, dataset_name, class_names, image_size=(224, 224), batch_size=32, is_rgb=False):
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.image_size = image_size
        self.batch_size = batch_size
        self.is_rgb = is_rgb
        self.model = self.load_keras_model(model_path)
        self.dataset = self.load_fo_dataset(dataset_dir, dataset_name)
        self.tf_dataset = self.fiftyone_to_tf_dataset()

    def load_keras_model(self, model_path):
        """Loads a Keras model from a file path."""
        model = load_model(model_path)
        return model

    def load_pytorch_model(self, model_path):
        """Loads a PyTorch model from a file path."""
        # model = torch.load(model_path)
        # return model
        pass

    def load_and_preprocess_image(self, file_path):
        """Loads and preprocesses an image for inference using TensorFlow."""
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        if not self.is_rgb:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def fiftyone_to_tf_dataset(self):
        """Converts a FiftyOne dataset view to a TensorFlow Dataset (tf.data.Dataset)."""
        file_paths = self.dataset.values("filepath")
        file_paths = [str(fp) for fp in file_paths]
        # Create a TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        tf_dataset = tf_dataset.map(
            lambda file_path: tf.py_function(
                func=lambda fp: self.load_and_preprocess_image(fp),
                inp=[file_path],
                Tout=[tf.float32, tf.string],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        tf_dataset = tf_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return tf_dataset

    def load_fo_dataset(self, dataset_dir, dataset_name):
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
                dataset_dir=dataset_dir,
                dataset_type=fo.types.ImageDirectory,
                name=dataset_name,
            )
        return dataset

    def run_inference(self):
        """Runs inference on images in a directory and saves the predictions to a JSON file."""
        # A dictionary to hold the class labels
        labels = {}

        # Iterate over the images in the directory
        for elements in self.tf_dataset:
            images, file_paths = elements
            predictions = self.model.predict(images)
            predicted_class_indices = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            for i in range(len(images)):
                file_path = file_paths[i].numpy().decode("utf-8")
                predicted_class_idx = int(predicted_class_indices[i])
                filename = os.path.basename(file_path)
                filename = os.path.splitext(filename)[0]
                confidence = float(confidences[i])

                labels[str(filename)] = {
                    "label": predicted_class_idx,
                    "confidence": confidence,
                    "attributes": {},
                }

        # Create the final output dictionary
        output = {"classes": self.class_names, "labels": labels}

        # Save the predictions list to a JSON file
        with open("labels.json", "w") as json_file:
            json.dump(output, json_file, indent=4)

        print("Predictions saved to labels.json")

    def execute(self):
        self.run_inference()
    
