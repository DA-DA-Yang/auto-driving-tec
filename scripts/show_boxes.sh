#需要指明所需文件的绝对路径
if [ $# -lt 2 ]; then
  echo "Usage: show_boxes.sh [pcd_path] [boxes_path]"
  exit 1
fi

PCD_PATH=$1
BOXES_PATH=$2

/auto-driving-tec/build/ShowBoxes $PCD_PATH $BOXES_PATH