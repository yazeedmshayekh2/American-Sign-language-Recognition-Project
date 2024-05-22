import tensorflow as tf
from PreProcessing import Preprocess, train_df, decoder
import numpy as np
import pandas as pd
from Model_Architecture import models
import os

class TFLiteModel(tf.Module):

    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model
    """

    def __init__(self, islr_models):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = Preprocess()
        self.islr_models = islr_models

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}

ROWS_PER_FRAME = 543  # number of landmarks per frame

def add_padding(number, target_divisor):

    padding = (target_divisor - (number % target_divisor)) % target_divisor
    return number + padding

def load_relevant_data_subset(pq_path):

    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    print("data.shape: ", data.shape)
    n_rows, n_cols = data.shape
    total_elements = n_rows * n_cols
    print('total_elements: ', total_elements)
    padding = add_padding(total_elements, ROWS_PER_FRAME) // 3
    # padding = (total_elements % ROWS_PER_FRAME) // 3
    print('data: ', data)
    # padding_rows = pd.DataFrame(np.nan, index=range(ROWS_PER_FRAME - (padding - n_rows)), columns=data.columns)
    # padded_df = pd.concat([data, padding_rows], ignore_index=True)
    padded_df = data.iloc[:-(ROWS_PER_FRAME - (padding - n_rows))]
    print(padded_df, padded_df.shape)
    n_frames = int(len(padded_df) / ROWS_PER_FRAME)
    padded_df = padded_df.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return padded_df.astype(np.float32), padded_df

tflite_keras_model = TFLiteModel(islr_models=models)
relevant_data, data = load_relevant_data_subset('test_case.parquet')
demo_output = tflite_keras_model(relevant_data)["outputs"]
print(data.shape)
print(decoder(np.argmax(demo_output.numpy(), axis=-1)))

