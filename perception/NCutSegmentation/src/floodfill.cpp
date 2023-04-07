
#include "floodfill.h"

FloodFill::FloodFill(float grid_radius, float cell_size)
{
    _grid_radius = grid_radius;
    _cell_size = cell_size;
}

FloodFill::FloodFill()
{
}

FloodFill::~FloodFill()
{
}

void FloodFill::getSegments(const std::vector<PointXYZI> &pointcloud,
                            std::vector<std::vector<int>> &point_index_in_segments,
                            std::vector<int> &num_cells_per_segment)
{   
    /*principle：
    将输入点云划分为cell，根据cell邻域是否包含点，将网格进行分割，变成一个个的连通域；
    然后把连通域中对应的点的索引进行输出，同时输出每个连通域包含的cell的数目。
    */

    // 建立网格
    buildGrid(pointcloud);
    // 确定连通域
    int num_components = getConnectComponents();
    // 输出结果
    point_index_in_segments.clear();
    point_index_in_segments.resize(num_components);
    num_cells_per_segment.clear();
    num_cells_per_segment.resize(num_components, 0);

    for (int i = 0; i < _num_grid; ++i)
    {
        int label = _grid_label[i];
        if (label != _empty)
        {
            num_cells_per_segment[label]++;
        }
    }
    for (int i = 0; i < _num_points; ++i)
    {
        // 当前点所处的网格索引
        int idx = _point_in_grid_index[i];
        // 根据网格索引计算连通域索引，然后把点的索引放入vector中
        point_index_in_segments[_grid_label[idx]].push_back(i);
    }
}

void FloodFill::buildGrid(const std::vector<PointXYZI> &pointcloud)
{
    _num_points = static_cast<int>(pointcloud.size());
    // 确定网格的最大最小范围
    float min_grid_radius = _grid_radius / 10.f;
    float max_x = -min_grid_radius;
    float min_x = -min_grid_radius;
    float max_y = min_grid_radius;
    float min_y = min_grid_radius;
    for (int i = 0; i < _num_points; ++i)
    {
        auto point = pointcloud[i];
        max_x = std::max(max_x, point.x);
        min_x = std::min(min_x, point.x);
        max_y = std::max(max_y, point.y);
        min_y = std::min(min_y, point.y);
    }

    const float lower_x = std::max(min_x, -_grid_radius);
    const float lower_y = std::max(min_y, -_grid_radius);
    const float upper_x = std::min(max_x, _grid_radius);
    const float upper_y = std::min(max_y, _grid_radius);
    _offset_x = -lower_x;
    _offset_y = -lower_y;

    // 划分网格
    _num_grid_row = static_cast<int>(ceil((upper_y - lower_y) / _cell_size)) + 1;
    _num_grid_col = static_cast<int>(ceil((upper_x - lower_x) / _cell_size)) + 1;
    _num_grid = _num_grid_row * _num_grid_col;

    // 确定点云每个点在网格中的索引
    _point_in_grid_index.assign(_num_points, -1);//初始均设为-1
    _grid_label.assign(_num_grid, _empty);       // 初始均设为空
    for (int i = 0; i < _num_points; ++i)
    {
        const int idx = getGridIndex(pointcloud[i].x, pointcloud[i].y);
        //点云对应的网格索引
        _point_in_grid_index[i] = idx;
        //有点落在网格中，说明该网格不为空
        _grid_label[idx] = _non_empty;
    }
}

int FloodFill::getConnectComponents()
{
    int count_components{};
    // 对每一个格子进行处理，若某个格子不为空，即包含有点
    // 以该格子为起点，利用dfsColoring方法进行连通域的搜索
    // 同时，将该连通域包含的所有格子命名为连通域的序号。
    for (int i = 0; i < _num_grid; ++i)
    {
        // 判断当前格子是否为空
        auto &label = _grid_label[i];
        if (label == _non_empty)
        {   
            // 不为空，将当前格子命名为连通域的序号
            label = count_components;
            // 对连通域进行搜索
            dfsColoring(i / _num_grid_col, i % _num_grid_col, count_components);
            ++count_components;
        }
    }
    return count_components;//返回连通域的数量
}

void FloodFill::dfsColoring(int row, int col, int index_component)
{
    // 以一个格子为中心，向其8个邻点进行扩散
    for (int idx = 0; idx < _num_directions;++idx)
    {
        const int row2 = row + _drow[idx];
        const int col2 = col + _dcol[idx];
        //判断邻域格子是否存在
        if(isRowInGrid(row2)&&isColInGrid(col2))
        {
            //判断邻域格子是否为空
            auto &label = _grid_label[row2 * _num_grid_col + col2];
            if (label == _non_empty)
            {
                label = index_component;
                dfsColoring(row2, col2, index_component);
            }
        }
    }
}

int FloodFill::getGridIndex(float x, float y)
{
    const int row = static_cast<int>((y + _offset_y) / _cell_size);
    if (!isRowInGrid(row))
    {
        return -1;
    }
    const int col = static_cast<int>((x + _offset_x) / _cell_size);
    if (!isColInGrid(col))
    {
        return -1;
    }
    return row * _num_grid_col + col;
}

bool FloodFill::getGridIndex(float x, float y, int& row, int& col)
{
    row = static_cast<int>((y + _offset_y) / _cell_size);
    if (!isRowInGrid(row))
    {
        return false;
    }
    col = static_cast<int>((x + _offset_x) / _cell_size);
    if (!isColInGrid(col))
    {
        return false;
    }
    return true;
}
    