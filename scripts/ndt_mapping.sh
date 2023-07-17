#需要指明所需文件的绝对路径
if [ $# -lt 2 ]; then
  echo "Usage: ndt_mapping.sh [workspace_dir==\"pcd_path\"] [output_file==\"output.pcd\"]"
  exit 1
fi

/auto-driving-tec/build/localization/ndt/ndt_mapping $1 $2