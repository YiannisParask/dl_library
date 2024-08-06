from tensorflow.keras import layers
from tensorflow import keras


class ResidualBlock(layers.Layer):
    def __init__(self, filters, pooling=False):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.pooling = pooling
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation('relu')
        self.max_pool = layers.MaxPooling2D(2, padding='same')
        self.conv3 = layers.Conv2D(filters, 1, strides=2, padding='same')
        self.conv4 = layers.Conv2D(filters, 1, padding='same')

    def call(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)

        if self.pooling:
            x = self.max_pool(x)
            residual = self.conv3(residual)
        elif self.filters != residual.shape[-1]:
            residual = self.conv4(residual)

        x = layers.add([x, residual])
        return x

# TODO: Parse config dictionary instead of passing arguments
class CnnModelSubClassing(keras.Model):
    '''
    Usage:
    num_classes = 10
    input_shape = (224, 224, 3)
    filters = [0, 1, 2]
    model = CnnModelSubClassing(num_classes, input_shape, output_function="sigmoid")
    model.build_model(filters, add_residual=True)
    '''
    def __init__(self, num_classes, input_shape, output_function="softmax"):
        super(CnnModelSubClassing, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(32)
        self.activation = layers.Activation("relu")
        self.output_dense = layers.Dense(num_classes, activation=output_function)
        self.conv_blocks = []

    def build_model(self, filters, add_residual=False):
        '''Function to build the model'''
        for filter in filters:
            conv_block = [
                layers.Conv2D(32 * 2**filter, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D()
            ]
            if add_residual:
                conv_block.append(ResidualBlock(32 * 2**filter))
            self.conv_blocks.append(conv_block)
        self.build((None, *self.input_shape))

    def call(self, inputs, training=True):
        '''Function to call the model'''
        x = inputs
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation(x)
        return self.output_dense(x)

    