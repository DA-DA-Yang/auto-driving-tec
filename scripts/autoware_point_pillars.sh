
#需要指明所需文件的绝对路径
if [ $# -lt 4 ]; then
  echo "Usage: autoware_point_pillars.sh [pointcloud_file_path] [box_file_name]"
  exit 1
fi

#在shell中调用conda命令
source /home/yangda/anaconda3/bin/activate
conda activate py310_pt120

cd /auto-driving-tec/build/perception/autoware_point_pillars/
./Pfe_Autoware $1
cd /auto-driving-tec/perception/autoware_point_pillars/
python rpn_autoware.py
cd /auto-driving-tec/build/perception/autoware_point_pillars/
./Post_Autoware $2