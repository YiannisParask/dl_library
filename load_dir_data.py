import keras


class LoadDirData:
    ''' Load data from directory '''
    def __init__(self, train_path, val_path, test_path):
        self.train_dir = train_path
        self.val_dir = val_path
        self.test_dir = test_path

    def load_data(self, img_size=(224, 224), batch_size=32, seed=0, split=0.2):
        train_dataset = keras.utils.image_dataset_from_directory(
            self.train_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=["fire", "non fire", "Smoke"],
            validation_split=split,
            subset="training",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size
        )

        validation_dataset = keras.utils.image_dataset_from_directory(
            self.train_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=["fire", "non fire", "Smoke"],
            validation_split=split,
            subset="validation",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size
        )

        # Load test dataset
        test_dataset = keras.utils.image_dataset_from_directory(
            self.test_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=["fire", "non fire", "Smoke"],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        return train_dataset, validation_dataset, test_dataset