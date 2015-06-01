#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <string>
#include <cstdlib>

#include "schedule.h"

template <typename T>
double Lasso(size_t m, size_t n, int seed, std::string file);

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
