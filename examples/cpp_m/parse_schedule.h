#ifndef PARSE_SCHEDULE_H_
#define PARSE_SCHEDULE_H_

#include "schedule.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using namespace rapidjson;

Schedule parse_schedule(const char* json_schedule,
                        const size_t m, const size_t n) {
  int m_blocks, n_blocks;
  size_t m_per, n_per;
  std::vector<ProcessInfo> info;
  Document d;
  Value s;

  d.Parse(json_schedule);
  m_blocks = d["m_blocks"].GetInt();
  n_blocks = d["n_blocks"].GetInt();

  m_per = m / m_blocks;
  n_per = n / n_blocks;

  const Value& a = d["info"];
  for (SizeType i = 0; i < a.Size(); ++i) {
    ProcessInfo proc;

    const Value& blockValue = a[i]["block"];

    proc.block.row = blockValue["row"].GetInt();
    proc.block.column = blockValue["column"].GetInt();

    proc.block.row_begin = m_per * proc.block.row;
    proc.block.row_end = m_per * (proc.block.row + 1);

    proc.block.column_begin = n_per * proc.block.column;
    proc.block.column_end = n_per * (proc.block.column + 1);

    const Value& gpus = a["gpu_indicies"];
    for (SizeType i = 0; i < gpus.Size(); ++i) {
      proc.gpu_indicies.push_back(gpus[i].GetInt());
    }

    info.push_back(proc);
  }

  return Schedule(m_blocks, n_blocks, info);
}
#endif//PARSE_SCHEDULE_H_
