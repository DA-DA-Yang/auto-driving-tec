import torch
import onnx
import onnxruntime
import numpy
import struct
import os
import argparse
import shutil
import sys
from collections import OrderedDict

print("\n")
print("------------darkSCNN------------")

model_path = "/auto-driving-tec/perception/lane_detector/model/denseline/denseline.onnx"
image_path = "/auto-driving-tec/data/test.png"

input = numpy.random.random(size=(1,3,480,640)).astype(numpy.float32)

model = onnx.load_model(model_path)
model.ir_version=7
onnx.save_model(model, model_path)
opset = model.opset_import[0].version
# 加载模型
device = ['CPUExecutionProvider']
model_run = onnxruntime.InferenceSession(model_path, providers=device)
output_name = []
for output in model_run.get_outputs():
    output_name.append(output.name)

pred_res = model_run.run(output_name,
                       {'data_input': input})

print("===================")
print("Inference success!")
print("===================")

