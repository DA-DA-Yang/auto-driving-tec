
#ifndef FLOODFILL_H
#define FLOODFILL_H

#include "define.h"
#include <vector>

class FloodFill
{
public:
    FloodFill();
    FloodFill(float grid_radius, float cell_size);
   ~FloodFill();

    /** 
     *@brief 根据输入点云，获得点云中的连通域
     *@param pointcloud 输入点云
     *@param segments_indices
     *@param num_cells_per_segment 每个连通域包含格子的数量
    */
    void getSegments(const std::vector<PointXYZI>& pointcloud, 
                     std::vector<std::vector<int>>& point_index_in_segments, 
                     std::vector<int>& num_cells_per_segment);
    
    inline void setGridRadius(float grid_radius){ _grid_radius = grid_radius; }
    inline void setCellSize(float cell_size){ _cell_size = cell_size; }

    inline int getNumRows() { return _num_grid_row; }
    inline int getNumCols() { return _num_grid_col; }

    const std::vector<int>& getPointInGridIdx() const { return _point_in_grid_index; }

    void buildGrid(const std::vector<PointXYZI> &pointcloud);
    int getConnectComponents();
    void dfsColoring(int row, int col, int index_component);

    int getGridIndex(float x, float y);
    bool getGridIndex(float x, float y, int& row, int& col);
    bool isRowInGrid(int row) { return (row >= 0 && row < _num_grid_row);  }
    bool isColInGrid(int col) { return (col >= 0 && col < _num_grid_col); }

private:
    float _grid_radius{80.f};
    float _cell_size{1.f};
    float _num_points{};
    float _offset_x{};
    float _offset_y{};
    int _num_grid_row{};
    int _num_grid_col{};
    int _num_grid{};
    std::vector<int> _point_in_grid_index;
    std::vector<int> _grid_label;

    const int _empty = -2;
    const int _non_empty = -1;

    // 以一个格子为中心，向周围8个方向扩散
    const int _num_directions{8}; 
    const int _drow[8]{-1, -1, -1, 0, 0, +1, +1, +1};
    const int _dcol[8]{-1, 0, +1, -1, +1, -1, 0, +1};
};

#endif