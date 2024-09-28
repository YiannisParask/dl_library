import tf2onnx
import onnx
import tensorflow as tf

class KerasToOnnx:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path

    def convert_func(self):
        '''
        Convert a functional or sequential model to ONNX format.
        '''
        model = tf.keras.models.load_model(self.model_path)
        spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save(onnx_model, self.output_path)
        print(f"Model has been successfully converted to {self.output_path}")
        
    def convert_subc(self, sample_input):
        '''
        Convert the subclassing model to ONNX format
        
        input:
        sample_input: sample input to the model (example: sample_input = tf.random.normal([1, 224, 224, 3]))
        '''
        model = tf.keras.models.load_model(self.model_path)
        _ = model(sample_input)
        spec = (tf.TensorSpec(sample_input.shape, tf.float32),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save(onnx_model, self.output_path)
        print(f"Model has been successfully converted to {self.output_path}")