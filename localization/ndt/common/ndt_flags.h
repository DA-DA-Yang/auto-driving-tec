#include <gflags/gflags.h>

// filter
DECLARE_double(min_scan_range);
DECLARE_double(max_scan_range);
DECLARE_double(min_add_scan_shift);
DECLARE_double(voxel_leaf_size);
DECLARE_double(sample_interval_time);
DECLARE_int32(align_map_count);
// ndt
DECLARE_double(trans_eps);
DECLARE_double(step_size);
DECLARE_double(ndt_res);
DECLARE_int32(max_iter);
// map
DECLARE_string(output_file);
DECLARE_string(output_dir);
DECLARE_string(workspace_dir);
DECLARE_string(trans_pcd_dir);