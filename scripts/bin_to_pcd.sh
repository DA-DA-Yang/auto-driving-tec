
#需要指明所需文件的绝对路径
if [ $# -lt 1 ]; then
  echo "Usage: show_boxes.sh [pointcloud_path]"
  exit 1
fi
POINTCLOUD_PATH = $1
/auto-driving-tec/build/tools/BinToPcd $1