import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class DataVisualization:
    def __init__(self):
        pass

    def plot_training_history(self, history):
        '''Function to plot the training and validation history'''
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Training and Validation Loss")

        plt.show()
        
    def plot_class_distribution(dataset_dir, subset_name):
        '''Plot the class distribution of the dataset
        
           Usage: plot_class_distribution("dataset/train", "Train")        
        '''
        class_count = {}
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            num_images = len(os.listdir(class_dir))
            class_count[class_name] = num_images
        
        plt.figure(figsize=(10, 5))
        plt.bar(class_count.keys(), class_count.values())
        plt.xlabel("Class")
        plt.ylabel("Number of images")
        plt.title("Class distribution" + " " + subset_name) 
        plt.show()
        
    def plot_images_from_dataset(dataset, class_names, num_images=6):
        '''Plot images from the dataset
        
           Usage: plot_images_from_dataset(train_ds, ["class1", "class2"], 9)
        '''
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            labels = tf.argmax(labels, axis=1)  
            num_images = min(num_images, len(images))
            for i in range(num_images):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i].numpy()])
                plt.axis("off")
                
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
        print(classification_report(y_true, y_pred))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()