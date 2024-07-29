from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ModelBuilder:
    def __init__(self):
        self.num_classes = 4
        self.input_shape = (224, 224, 3)
        self.model = None

    def build_model(self, filters):
        """Builds a simple CNN model for image classification."""
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        for filter in filters:
            x = layers.Conv2D(32 * 2**filter, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(32)(x)
        x = layers.Activation("relu")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model

    def compile_model(self, optimizer="adam", loss="categorical_crossentropy", metric=["accuracy"]):
        """Compiles the model."""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric,
        )
        return self.model

    def train_model(self, train_dataset, val_dataset, epochs=10, callbacks=None):
        """Trains the model."""
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )
        return history

    def evaluate_model(self, test_dataset):
        """Evaluates the model."""
        loss, accuracy = self.model.evaluate(test_dataset)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

    def get_class_names(self, dataset):
        ''' Extract the class names '''
        return dataset.class_names
     
    def metrics(self, model, test_ds):
        '''Function to calculate the metrics of the model'''
        class_names = self.get_class_names(test_ds)
        # Make predictions
        predictions = model.predict(test_ds)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get the true labels
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_true = np.argmax(y_true, axis=1)
        
        # Print classification report
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred))
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    