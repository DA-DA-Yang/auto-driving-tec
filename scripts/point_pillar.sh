
#需要指明所需文件的绝对路径
if [ $# -lt 1 ]; then
  echo "Usage: point_pillar.sh [pointcloud_path]"
  exit 1
fi

#在shell中调用conda命令
source /home/yangda/anaconda3/bin/activate
conda activate py310_pt120

cd /auto-driving-tec/perception/point_pillars/build/
./Pfe $1
python ../inference.py --dirPath=$1
./Post $1