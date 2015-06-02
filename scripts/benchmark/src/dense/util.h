#ifndef POGS_UTIL_H_
#define POGS_UTIL_H_

#include "pogs.h"
#include <stdio.h>

#define MASTER(kRank) if (kRank == 0)

#define NUM_RANDS 4

template <typename T>
void SaveMatrix(T *a,
                std::vector<FunctionObj<T> > &f,
                std::vector<FunctionObj<T> > &g,
                std::string filename) {
  FILE* pFile;
  pFile = fopen(filename.c_str(), "wb");
  // Write out f operators
  auto f_size = f.size();
  fwrite(&f_size,
         sizeof(typename std::vector<FunctionObj<T> >::size_type),
         1,
         pFile);
  for (int i = 0; i < f.size(); i++) {
    FunctionObj<T> &fo = f[i];
    fwrite(&fo.h, sizeof(fo.h), 1, pFile);
    fwrite(&fo.a, sizeof(fo.a), 1, pFile);
    fwrite(&fo.b, sizeof(fo.b), 1, pFile);
    fwrite(&fo.c, sizeof(fo.c), 1, pFile);
    fwrite(&fo.d, sizeof(fo.d), 1, pFile);
    fwrite(&fo.e, sizeof(fo.e), 1, pFile);
  }
  // Write out g operators
  auto g_size = g.size();
  fwrite(&g_size,
         sizeof(typename std::vector<FunctionObj<T> >::size_type),
         1,
         pFile);
  for (int i = 0; i < g.size(); i++) {
    FunctionObj<T> &go = g[i];
    fwrite(&go.h, sizeof(go.h), 1, pFile);
    fwrite(&go.a, sizeof(go.a), 1, pFile);
    fwrite(&go.b, sizeof(go.b), 1, pFile);
    fwrite(&go.c, sizeof(go.c), 1, pFile);
    fwrite(&go.d, sizeof(go.d), 1, pFile);
    fwrite(&go.e, sizeof(go.e), 1, pFile);
  }
  // Write out A
  fwrite(a, sizeof(T), f.size() * g.size(), pFile);
  fclose(pFile);
}

template <typename T>
void LoadMatrix(std::string filename,
                T **a,
                std::vector<FunctionObj<T> > &f,
                std::vector<FunctionObj<T> > &g) {
  FILE* pFile;
  pFile = fopen(filename.c_str(), "rb");
  // Write out f operators
  typename std::vector<FunctionObj<T> >::size_type f_size;
  fread(&f_size,
        sizeof(typename std::vector<FunctionObj<T> >::size_type),
        1,
        pFile);
  for (int i = 0; i < f_size; i++) {
    FunctionObj<T> fo;
    fread(&fo.h, sizeof(fo.h), 1, pFile);
    fread(&fo.a, sizeof(fo.a), 1, pFile);
    fread(&fo.b, sizeof(fo.b), 1, pFile);
    fread(&fo.c, sizeof(fo.c), 1, pFile);
    fread(&fo.d, sizeof(fo.d), 1, pFile);
    fread(&fo.e, sizeof(fo.e), 1, pFile);
    f.push_back(fo);
  }
  // Write out g operators
  typename std::vector<FunctionObj<T> >::size_type g_size;
  fread(&g_size,
        sizeof(typename std::vector<FunctionObj<T> >::size_type),
        1,
        pFile);
  for (int i = 0; i < g_size; i++) {
    FunctionObj<T> go;
    fread(&go.h, sizeof(go.h), 1, pFile);
    fread(&go.a, sizeof(go.a), 1, pFile);
    fread(&go.b, sizeof(go.b), 1, pFile);
    fread(&go.c, sizeof(go.c), 1, pFile);
    fread(&go.d, sizeof(go.d), 1, pFile);
    fread(&go.e, sizeof(go.e), 1, pFile);
    g.push_back(go);
  }
  // Write out A
  (*a) = new T[f_size * g_size];
  fread((*a), sizeof(T), g_size * f_size, pFile);
  fclose(pFile);
}

#endif
