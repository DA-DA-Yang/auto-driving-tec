# apollo 编译库的移植工具
# 遍历.cache文件，根据已有目录，创建多级目录
# 将目录下包含的.so文件进行拷贝

import os,string
from shutil import copyfile
cyber_tools_list = ['mainboard', 
                    'cyber_recorder',
                    'cyber_launch',
                    'cyber_monitor',
                    'cyber_visualizer']
def traverse_dir(dir_path_in, dir_path_out):
    # 遍历文件夹，获得文件夹下的根路径、子目录、子文件
    for root, dirs, files in os.walk(dir_path_in):
        # 在新的路径下创建文件夹
        new_path = root.replace(dir_path_in, dir_path_out)
        if(os.path.exists(new_path) == False):
            os.mkdir(new_path)
        # 遍历当前文件夹下的文件列表
        for file in files:
            # 将cyber及mainboard可执行程序进行拷贝
            if file in cyber_tools_list:
                copyfile(root+'/'+file, new_path + '/' + file, follow_symlinks=False)
            # 将so文件拷贝到新路径下
            suff = os.path.splitext(file)[-1]
            if suff == '.so':
                # 设置False，当源文件为软链接时，复制软链接
                copyfile(root+'/'+file, new_path + '/' + file,follow_symlinks=False)


dir_path_in = os.getcwd()+'/.cache'
dir_path_out = os.getcwd() + '/apollo_to_a1000/.cache'

if os.path.exists(dir_path_in) == False:
    print(f"目标文件夹不存在：{dir_path_in}")
    exit()

# 创建输出文件夹
if os.path.exists(dir_path_out):
    print(f'文件夹已存在:{dir_path_out}')
    exit()
else:
    os.makedirs(dir_path_out)

traverse_dir(dir_path_in, dir_path_out)

print("====================")
print("[OK] Done!")
print("====================\n")
