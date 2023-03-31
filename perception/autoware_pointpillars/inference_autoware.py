import torch
import onnx
import onnxruntime
import numpy
import struct
import os
from collections import OrderedDict

dir_path = '/ait_study/perception/autoware_pointpillars/model/'
device = ['CPUExecutionProvider']
res_dir_path = '/home/yangda/my_project/perception_lidar/tmp/temp/'

# 读取数据
pillar_x = numpy.fromfile(
    res_dir_path+'pillar_x.bin', dtype=numpy.float32)
pillar_x.shape = 1, 1, 12000, 100


pillar_y = numpy.fromfile(
    res_dir_path+'pillar_y.bin', dtype=numpy.float32)
pillar_y.shape = 1, 1, 12000, 100

pillar_z = numpy.fromfile(
    res_dir_path+'pillar_z.bin', dtype=numpy.float32)
pillar_z.shape = 1, 1, 12000, 100

pillar_i = numpy.fromfile(
    res_dir_path+'pillar_i.bin', dtype=numpy.float32)
pillar_i.shape = 1, 1, 12000, 100

x_coors_for_sub_shaped = numpy.fromfile(
    res_dir_path+'x_coors_for_sub_shaped.bin', dtype=numpy.float32)
x_coors_for_sub_shaped.shape = 1, 1, 12000, 100

y_coors_for_sub_shaped = numpy.fromfile(
    res_dir_path+'y_coors_for_sub_shaped.bin', dtype=numpy.float32)
y_coors_for_sub_shaped.shape = 1, 1, 12000, 100

pillar_feature_mask = numpy.fromfile(
    res_dir_path+'pillar_feature_mask.bin', dtype=numpy.float32)
pillar_feature_mask.shape = 1, 1, 12000, 100

num_points_per_pillar = numpy.fromfile(
    res_dir_path+'num_points_per_pillar.bin', dtype=numpy.float32)
num_points_per_pillar.shape = 1, 12000

x_coors = numpy.fromfile(
    res_dir_path+'x_coors.bin', dtype=numpy.int32)
x_coors.shape = 1, 12000

y_coors = numpy.fromfile(
    res_dir_path+'y_coors.bin', dtype=numpy.int32)
y_coors.shape = 1, 12000

# 加载模型
#pfe
pfe_run = onnxruntime.InferenceSession(
    dir_path + 'pfe.onnx', providers=device)

output_name_pfe = [pfe_run.get_outputs()[0].name]
pred_pfe = pfe_run.run(output_name_pfe, 
                        {'pillar_x': pillar_x, 
                        'pillar_y': pillar_y, 
                        'pillar_z': pillar_z, 
                        'pillar_i': pillar_i, 
                        'x_sub_shaped': x_coors_for_sub_shaped, 
                        'y_sub_shaped': y_coors_for_sub_shaped, 
                        'mask': pillar_feature_mask, 
                        'num_points_per_pillar': num_points_per_pillar})
pred_pfe = OrderedDict(zip(output_name_pfe, pred_pfe))

print('pred_pfe: ')
for output_key in pred_pfe.keys():
    print(pred_pfe[output_key].shape)

#rpn
rpn_input = numpy.zeros([1, 64, 496, 432],dtype=numpy.float32)
pfe_174 = pred_pfe['174']
count=0
# scatter
for i in range(12000):
    y = y_coors[0][i]
    x = x_coors[0][i]
    if x == 0 and y == 0:
        count = count + 1
    for f in range(64): 
        rpn_input[0][f][y][x] = pred_pfe['174'][0][f][i][0]

#rpn_input = torch.rand(1, 64, 496, 432).to('cpu').numpy()

rpn_run = onnxruntime.InferenceSession(
    dir_path + 'rpn.onnx', providers=device)

output_name_rpn = []
for output in rpn_run.get_outputs():
    output_name_rpn.append(output.name)
pred_rpn = rpn_run.run(output_name_rpn,
                       {'input.1': rpn_input})
pred_rpn = OrderedDict(zip(output_name_rpn, pred_rpn))
#dayang:rpn输出三种结果
# 184：box，1*248*216*14， 参数7个
# 185：cls, 1*248*216*2，参数1个
# 187: dir, 1*248*216*4，参数2个
# 248*216*2为anchor数目

print('pred_rpn: ')
bin_out = open(res_dir_path+'rpn_data.bin', 'wb')
count = 0
valid_count=0
for output_key in pred_rpn.keys():
    print(pred_rpn[output_key].shape)
    for i in range(pred_rpn[output_key].shape[0]):
        for j in range(pred_rpn[output_key].shape[1]):
            for k in range(pred_rpn[output_key].shape[2]):
                for m in range(pred_rpn[output_key].shape[3]):
                    tvalue = pred_rpn[output_key][i][j][k][m]
                    tdata = struct.pack('f', tvalue)
                    count = count + 1
                    bin_out.write(tdata)
bin_out.close()
first_value = pred_rpn['184'][0][0][0][0]
print("first value: ")
print(first_value)
last_value = pred_rpn['187'][0][247][215][3]
print("last value: ")
print(last_value)
print('total size:')
print(count)

print("==========================\n"
      "Done!\n"
      "==========================")
