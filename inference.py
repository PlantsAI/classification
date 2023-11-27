import argparse
import numpy as np
import cv2
import onnxruntime


parser = argparse.ArgumentParser(description="Get inference from onnx model")
parser.add_argument("--input", default="io/input/image2.jpg", type=str)
parser.add_argument("--output", default="./io/output", type=str)
parser.add_argument("--weights", default="weights/best.onnx", type=str)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--thread", default=4, type=int)
args = parser.parse_args()


def load_model():
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = args.thread
    opts.inter_op_num_threads = args.thread
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(args.weights, opts, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def preprocess(image, size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    image = image.transpose(2, 0, 1).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


if __name__ == "__main__":
    session, input_name, output_name = load_model()    
    image = cv2.imread(args.input)
    image = preprocess(image, args.size)
    result = session.run([output_name], {input_name: image})
    print(result)
    print(np.argmax(result))
