from deep_speaker.conv_models import DeepSpeakerModel
import argparse
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np

def main(model_path: str, output_path: str, device: str = 'cuda'):
    NUM_FRAMES = 99
    NUM_FBANKS = 64
    model = DeepSpeakerModel(
            batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1),
        )
    model.m.load_weights(model_path, by_name=True)

    input_signature = [tf.TensorSpec([None, NUM_FRAMES, NUM_FBANKS, 1], tf.float32, name='x')]
    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(model.m, input_signature, opset=14)
    onnx.save(onnx_model, output_path)

    input1 = np.zeros((10, NUM_FRAMES, NUM_FBANKS, 1), np.float32)
    sess = ort.InferenceSession(output_path, providers=["CUDAExecutionProvider"])
    results_ort = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: input1})
    print(results_ort)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='tensorflow model path')
    parser.add_argument('--output_path', type=str, required=True, help='output model path')
    args = parser.parse_args()
    main(args.model_path, args.output_path)