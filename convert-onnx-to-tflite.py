import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare

onnx_file = "mobilenetv2-7.onnx"
graph_def_file = onnx_file.rsplit('.', 1)[0] + '.pb'
tflite_file = onnx_file.rsplit('.', 1)[0] + '.tflite'

onnx_model = onnx.load(onnx_file)  # load onnx model
tf_rep = prepare(onnx_model, strict=False)  # prepare tf representation
tf_rep.export_graph(graph_def_file)  # export the model
input_arrays = [n.name for n in onnx_model.graph.input]
output_arrays = [n.name for n in onnx_model.graph.output]

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
with open(tflite_file, "wb") as file:
    print(tflite_file)
    file.write(tflite_model)