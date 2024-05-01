import tensorflow as tf
import json
import pandas as pd
import numpy as np

def model_run():

    # Opening JSON file
    f = open('inference_args.json')
    data = json.load(f)
    COLUMNS0 = data['selected_columns']

    with open('character_to_prediction_index.json') as json_file:
        CHAR2ORD = json.load(json_file)

    ORD2CHAR = {j:i for i,j in CHAR2ORD.items()}

    # Number of Frames to resize recording to
    N_TARGET_FRAMES = 128

    example_parquet_df = pd.read_parquet('test_case.parquet')

    # Each parquet file contains 1000 recordings
    print(f'# Unique Recording: {example_parquet_df.index.nunique()}')
    # Display DataFrame layout
    print(example_parquet_df.head())

    # Output Predictions to string
    def outputs2phrase(outputs):
        if outputs.ndim == 2:
            outputs = np.argmax(outputs, axis=1)

        return ''.join([ORD2CHAR.get(s, '') for s in outputs])


    demo_sequence_id = 1
    demo_raw_data = example_parquet_df.loc[demo_sequence_id, COLUMNS0].values

    print(demo_raw_data)

    # Path to the TFLite model file
    tflite_model_path = 'model.tflite'

    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    # Resize input shape for dynamic shape model and allocate tensor
    interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], [128, 164])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    data0 = np.array(demo_raw_data, dtype=np.float32)

    # Fill NaN Values With 0
    data = tf.where(tf.math.is_nan(data0), 0.0, data0)

    # Hacky
    data = data[None]

    # Empty Hand Frame Filtering
    hands = tf.slice(data, [0, 0, 0], [-1, -1, 84])
    hands = tf.abs(hands)
    mask = tf.reduce_sum(hands, axis=2)
    mask = tf.not_equal(mask, 0)
    data = data[mask][None]

    # Pad Zeros
    N_FRAMES = len(data[0])
    if N_FRAMES < N_TARGET_FRAMES:
        data = tf.concat((
            data,
            tf.zeros([1, N_TARGET_FRAMES - N_FRAMES, len(COLUMNS0)], dtype=tf.float32)
        ), axis=1)

    # Downsample
    data = tf.image.resize(
        data,
        [1, N_TARGET_FRAMES],
        method=tf.image.ResizeMethod.BILINEAR,
    )

    # Squeeze Batch Dimension
    data = tf.squeeze(data, axis=[0])

    interpreter.set_tensor(input_details[0]['index'], data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return outputs2phrase(output_data)
