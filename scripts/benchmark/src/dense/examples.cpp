#include <iostream>

#include "examples.h"

template <typename T>
ExampleData<T> ErrorProblem(size_t, size_t, int) {
  std::cerr << "Problem type invalid" << std::endl;
  std::exit(EXIT_FAILURE);
  return {};
}


ProblemType GetProblemFn(std::string type){
  ProblemType pType;

  if (type == "lasso") {
    pType = LASSO;
    // } else if (typ == "lasso_path") {
    //   pType = LASSO_PATH;
    // } else if (typ == "logistic") {
    //   pType = LOGISTIC;
    // } else if (typ == "lp_eq") {
    //   pType = LP_EQ;
    // } else if (typ == "lp_ineq") {
    //   pType = LP_INEQ;
    // } else if (typ == "non_neg_l2") {
    //   pType = NON_NEG_L2;
    // } else if (typ == "svm") {
    //   pType = SVM;
  } else {
    std::cout << "No problem of that type\n" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return pType;
}
