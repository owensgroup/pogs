#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include <string>
#include <cstdlib>

#include "pogs.h"
#include "schedule.h"

typedef double real_t;

template <typename T>
struct ExampleData {
  std::vector<T> A;
  std::vector<FunctionObj<T> > f, g;
};

template <typename T>
ExampleData<T> Lasso(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
ExampleData<T> LpEq(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
ExampleData<T> LpEqM(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
ExampleData<T> LpCone(pogs::Schedule &s, size_t m, size_t n, int seed);

/* template <typename T>
double LassoPath(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double Logistic(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double LpIneq(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double NonNegL2(pogs::Schedule &s, size_t m, size_t n, int seed);

template <typename T>
double Svm(pogs::Schedule &s, size_t m, size_t n, int seed);
*/

template<typename T>
using GenFn = ExampleData<T> (*)(pogs::Schedule &s, size_t m, size_t n,
                                 int seed);

enum ProblemType {
  LASSO,
  LP_EQ,
  LP_EQ_M,
  LP_CONE
  // LASSO_PATH,
  // LOGISTIC,
  // LP_INEQ,
  // NON_NEG_L2,
  // SVM
};

extern const GenFn<real_t> ExampleFns[];

ProblemType GetProblemFn(std::string type);

#endif  // EXAMPLES_H_
