#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <string>
#include <cstdlib>

#include "pogs.h"
#include "schedule.h"

template <typename T>
struct ExampleData {
  std::vector<T> A;
  std::vector<FunctionObj<T> > f, g;
};

template <typename T>
ExampleData<T> Lasso(size_t m, size_t n, int seed);

/* template <typename T>
double LassoPath(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double Logistic(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double LpEq(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double LpIneq(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double NonNegL2(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double Svm(pogs::Schedule &s, size_t m, size_t n, int seed);
*/

#endif  // EXAMPLES_H_
