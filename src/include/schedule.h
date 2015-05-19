#ifndef SCHEDULE_H_
#define SCHEDULE_H_

#include <vector>

namespace pogs {


// For distributing linear algebra
struct MatrixMeta {
  size_t row_begin, row_end;
  size_t column_begin, column_end;

  size_t Rows() const { return row_end - row_begin; }
  size_t Cols() const { return column_end - column_begin; }
};

// For distributing optimization
struct BlockMeta {
  int row, column;
  size_t row_begin, row_end;
  size_t column_begin, column_end;

  size_t Rows() const { return row_end - row_begin; }
  size_t Cols() const { return column_end - column_begin; }
};

struct ProcessInfo {
  MatrixMeta matrix;
  BlockMeta block;
  std::vector<int> gpu_indicies;

};

class Schedule {
 private:
  // Number of blocks
  const int _m_blocks, _n_blocks;

  // Process ID (can use MPI rank) ->
  //   DLA:
  //     - master ID
  //     - begin range
  //     - end range
  //   DLO:
  //     - block row
  //     - block column
  //     - begin range
  //     - end range
  const std::vector<ProcessInfo> _info;
 public:
  Schedule(int m_blocks, int n_blocks,
           const std::vector<ProcessInfo> info)
    : _m_blocks(m_blocks), _n_blocks(n_blocks), _info(info) {}
  ~Schedule() {}

  int MBlocks() const { return _m_blocks; }
  int NBlocks() const { return _n_blocks; }
  const ProcessInfo& At(int id) const { return _info[id]; }
};
}
#endif
