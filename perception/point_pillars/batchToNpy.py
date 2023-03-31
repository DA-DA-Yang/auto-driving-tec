import torch
import onnx
import onnxruntime
import numpy
import struct
import os
from collections import OrderedDict
import shutil

in_dir_path = '/home/yangda/my_project/ait_study/tmp/kitti-10/batch_image/'
out_dir_path = '/home/yangda/my_project/ait_study/tmp/kitti-10/npy/'
if os.access(out_dir_path, os.F_OK) == True:
    shutil.rmtree(out_dir_path)
os.mkdir(out_dir_path)

for file_name in os.listdir(in_dir_path):
    batch_image = numpy.fromfile(in_dir_path + file_name, dtype=numpy.float32)
    batch_image.shape=1, 10, 400, 352
    out_name = out_dir_path + os.path.splitext(file_name)[0] + '.npy'
    numpy.save(out_name, batch_image)
