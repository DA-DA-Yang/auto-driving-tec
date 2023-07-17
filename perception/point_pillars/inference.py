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
print(">>>------------rpn------------<<<")

parser = argparse.ArgumentParser()
parser.add_argument('--dirPath', type=str, default=None)
args = parser.parse_args()

in_dirPath = args.dirPath
model_path = "/auto-driving-tec/perception/point_pillars/model/pointPillar.onnx"

str_t = in_dirPath[::-1]
str_t = str_t.split("/", 2)[2]
str_t = str_t[::-1] + "/"

in_dirPath = str_t + "batch_image/"
if os.access(in_dirPath, os.F_OK) == False:
    print("Read batch_image_dir failed: %s" %(in_dirPath))
    sys.exit()
print("Read batch_image_dir success: %s" %(in_dirPath))

# 创建输出文件夹
out_dirPath = str_t + "rpn_data/"
if os.access(out_dirPath, os.F_OK) == True:
    shutil.rmtree(out_dirPath)
os.mkdir(out_dirPath)

# 加载模型
device = ['CPUExecutionProvider']
rpn_run = onnxruntime.InferenceSession(model_path, providers=device)
output_name_rpn = []
for output in rpn_run.get_outputs():
    output_name_rpn.append(output.name)

for file_name in os.listdir(in_dirPath):
    # 读取batch_image
    batch_image = numpy.fromfile(in_dirPath+file_name, dtype=numpy.float32)
    batch_image.shape=1, 10, 400, 352
    max_value = numpy.max(batch_image)
    print("max value: %f" %(max_value))

    input = batch_image
    pred_rpn = rpn_run.run(output_name_rpn,
                       {'batch_image': input})
    pred_rpn = OrderedDict(zip(output_name_rpn, pred_rpn))

    # print('pred_rpn: ')
    out_name = out_dirPath + os.path.splitext(file_name)[0] + '.bin'
    bin_out = open(out_name, 'wb')
    count=0
    for output_key in pred_rpn.keys():
        # print(pred_rpn[output_key].shape)
        for i in range(pred_rpn[output_key].shape[0]):
            for j in range(pred_rpn[output_key].shape[1]):
                for k in range(pred_rpn[output_key].shape[2]):
                    for m in range(pred_rpn[output_key].shape[3]):
                        tvalue=pred_rpn[output_key][i][j][k][m]
                        tdata = struct.pack('f', tvalue)
                        count = count +1
                        bin_out.write(tdata)
    print("success: %s" % (file_name))
    bin_out.close()

print("==========================\n"
      " Rpn done!\n"
      "==========================")
