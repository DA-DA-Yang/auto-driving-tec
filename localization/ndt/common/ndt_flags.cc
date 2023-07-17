#include "localization/ndt/common/ndt_flags.h"

// filter
DEFINE_double(min_scan_range, 4.0, "the square of the min scan range");
DEFINE_double(max_scan_range, 40000.0, "the square of the max scan range");
DEFINE_double(min_add_scan_shift, 1.0, "the square of the min add scan length");
DEFINE_double(voxel_leaf_size, 0.5, "voxel leaf size");
DEFINE_double(sample_interval_time, 0, "cloud map sample interval time");
DEFINE_int32(align_map_count, 30, "cloud count in map for align");
// ndt
DEFINE_double(trans_eps, 0.001, "transformation epsilon");
DEFINE_double(step_size, 0.1, "step size");
DEFINE_double(ndt_res, 1, "ndt resolution");
DEFINE_int32(max_iter, 30, "maximum iterations times");
// map
DEFINE_string(output_file, "/auto-driving-tec/data/ndt_mapping/output.pcd", "map save file path");
DEFINE_string(output_dir, "/auto-driving-tec/data/ndt_mapping/", "map save dir");
DEFINE_string(workspace_dir, "/auto-driving-tec/data/ndt_mapping/pcd", "work dir");
DEFINE_string(trans_pcd_dir, "/auto-driving-tec/data/ndt_mapping/transform_pcd", "work dir");