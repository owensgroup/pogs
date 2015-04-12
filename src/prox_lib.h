#ifndef PROX_LIB_H_
#define PROX_LIB_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif

#include "_interface_defs.h"

// List of functions supported by the proximal operator library.
enum Function { kAbs,       // f(x) = |x|
                kExp,       // f(x) = e^x
                kHuber,     // f(x) = huber(x)
                kIdentity,  // f(x) = x
                kIndBox01,  // f(x) = I(0 <= x <= 1)
                kIndEq0,    // f(x) = I(x = 0)
                kIndGe0,    // f(x) = I(x >= 0)
                kIndLe0,    // f(x) = I(x <= 0)
                kLogistic,  // f(x) = log(1 + e^x)
                kMaxNeg0,   // f(x) = max(0, -x)
                kMaxPos0,   // f(x) = max(0, x)
                kNegEntr,   // f(x) = x log(x)
                kNegLog,    // f(x) = -log(x)
                kRecipr,    // f(x) = 1/x
                kSquare,    // f(x) = (1/2) x^2
                kZero };    // f(x) = 0

// Object associated with the generic function c * f(a * x - b) + d * x.
// Parameters a and c default to 1, while b and d default to 0.
template <typename T>
struct FunctionObj {
  Function h;
  T a, b, c, d, e;

  FunctionObj(Function h, T a, T b, T c, T d, T e)
      : h(h), a(a), b(b), c(c), d(d), e(e) { CheckConsts(); }
  FunctionObj(Function h, T a, T b, T c, T d)
      : h(h), a(a), b(b), c(c), d(d), e(0) { CheckConsts(); }
  FunctionObj(Function h, T a, T b, T c)
      : h(h), a(a), b(b), c(c), d(0), e(0) { CheckConsts(); }
  FunctionObj(Function h, T a, T b)
      : h(h), a(a), b(b), c(1), d(0), e(0) { }
  FunctionObj(Function h, T a)
      : h(h), a(a), b(0), c(1), d(0), e(0) { }
  explicit FunctionObj(Function h)
      : h(h), a(1), b(0), c(1), d(0), e(0) { }
  FunctionObj()
      : h(kZero), a(1), b(0), c(1), d(0), e(0) { }

  void CheckConsts() {
    if (c < static_cast<T>(0))
      Printf("WARNING c < 0. Function not convex. Using c = 0");
    if (e < static_cast<T>(0))
      Printf("WARNING e < 0. Function not convex. Using e = 0");
    c = std::max(c, static_cast<T>(0));
    e = std::max(e, static_cast<T>(0));
  }
};


// Local Functions.
namespace {
//  Evaluate abs(x)
__DEVICE__ inline double Abs(double x) { return fabs(x); }
__DEVICE__ inline float Abs(float x) { return fabsf(x); }

//  Evaluate acos(x)
__DEVICE__ inline double Acos(double x) { return acos(x); }
__DEVICE__ inline float Acos(float x) { return acosf(x); }

//  Evaluate cos(x)
__DEVICE__ inline double Cos(double x) { return cos(x); }
__DEVICE__ inline float Cos(float x) { return cosf(x); }

//  Evaluate e^x
__DEVICE__ inline double Exp(double x) { return exp(x); }
__DEVICE__ inline float Exp(float x) { return expf(x); }

//  Evaluate log(x)
__DEVICE__ inline double Log(double x) { return log(x); }
__DEVICE__ inline float Log(float x) { return logf(x); }

//  Evaluate max(x, y)
__DEVICE__ inline double Max(double x, double y) { return fmax(x, y); }
__DEVICE__ inline float Max(float x, float y) { return fmaxf(x, y); }

//  Evaluate max(x, y)
__DEVICE__ inline double Min(double x, double y) { return fmin(x, y); }
__DEVICE__ inline float Min(float x, float y) { return fminf(x, y); }

//  Evaluate x^y
__DEVICE__ inline double Pow(double x, double y) { return pow(x, y); }
__DEVICE__ inline float Pow(float x, float y) { return powf(x, y); }

//  Evaluate sqrt(x)
__DEVICE__ inline double Sqrt(double x) { return sqrt(x); }
__DEVICE__ inline float Sqrt(float x) { return sqrtf(x); }

//  Evaluate tol
template <typename T>
__DEVICE__ inline T Tol() { return 1e-10; }
template <>
__DEVICE__ inline double Tol() { return 1e-10; }
template <>
__DEVICE__ inline float Tol() { return 1e-5f; }

// Evalution of max(0, x).
template <typename T>
__DEVICE__ inline T MaxPos(T x) {
  return Max(static_cast<T>(0), x);
}

//  Evalution of max(0, -x).
template <typename T>
__DEVICE__ inline T MaxNeg(T x) {
  return Max(static_cast<T>(0), -x);
}

//  Evalution of sign(x)
template <typename T>
__DEVICE__ inline T Sign(T x) {
  return x >= 0 ? 1 : -1;
}

// Evaluate the principal branch of the Lambert W function.
// ref: http://keithbriggs.info/software/LambertW.c
template <typename T>
__DEVICE__ inline T LambertW(T x) {
  const T kEm1 = static_cast<T>(0.3678794411714423215955237701614608);
  const T kE = static_cast<T>(2.7182818284590452353602874713526625);
  if (x == 0) {
    return 0;
  } else if (x < -kEm1 + 1e-4) {
    T q = x + kEm1, r = Sqrt(q), q2 = q * q, q3 = q2 * q;
    return
     -static_cast<T>(1.0)
     +static_cast<T>(2.331643981597124203363536062168) * r
     -static_cast<T>(1.812187885639363490240191647568) * q
     +static_cast<T>(1.936631114492359755363277457668) * r * q
     -static_cast<T>(2.353551201881614516821543561516) * q2
     +static_cast<T>(3.066858901050631912893148922704) * r * q2
     -static_cast<T>(4.175335600258177138854984177460) * q3
     +static_cast<T>(5.858023729874774148815053846119) * r * q3
     -static_cast<T>(8.401032217523977370984161688514) * q3 * q;
  } else {
    T w;
    if (x < 1) {
      T p = Sqrt(static_cast<T>(2.0 * (kE * x + 1.0)));
      w = static_cast<T>(-1.0 + p * (1.0 + p * (-1.0 / 3.0 + p * 11.0 / 72.0)));
    } else {
      w = Log(x);
    }
    if (x > 3)
      w -= Log(w);
    for (unsigned int i = 0; i < 10; i++) {
      T e = Exp(w);
      T t = w * e - x;
      T p = w + static_cast<T>(1);
      t /= static_cast<T>(e * p - 0.5 * (p + 1.0) * t / p);
      w -= t;
    }
    return w;
  }
}

// Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root.
// ref: http://math.stackexchange.com/questions/60376
template <typename T>
__DEVICE__ inline T CubicSolve(T p, T q, T r) {
  T s = p / 3, s2 = s * s, s3 = s2 * s;
  T a = -s2 + q / 3;
  T b = s3 - s * q / 2 + r / 2;
  T a3 = a * a * a;
  T b2 = b * b;
  if (a3 + b2 >= 0) {
    T A = Pow(Sqrt(a3 + b2) - b, static_cast<T>(1) / 3);
    return -s - a / A + A;
  } else {
    T A = Sqrt(-a3);
    T B = Acos(-b / A);
    T C = Pow(A, static_cast<T>(1) / 3);
    return -s + (C - a / C) * Cos(B / 3);
  }
}
}  // namespace

// Proximal operator definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and five parameters (a, b, c, d and rho)
// and returns the evaluation of
//
//   x -> Prox{c * f(a * x - b) + d * x + e * x ^ 2},
//
// where Prox{.} is the proximal operator with penalty parameter rho.
template <typename T>
__DEVICE__ inline T ProxAbs(T v, T rho) {
  return MaxPos(v - 1 / rho) - MaxNeg(v + 1 / rho);
}

template <typename T>
__DEVICE__ inline T ProxNegEntr(T v, T rho) {
  return LambertW(Exp(rho * v - 1) * rho) / rho;
}

template <typename T>
__DEVICE__ inline T ProxExp(T v, T rho) {
  return v - LambertW(Exp(v) / rho);
}

template <typename T>
__DEVICE__ inline T ProxHuber(T v, T rho) {
  return Abs(v) < 1 + 1 / rho ? v * rho / (1 + rho) : v - Sign(v) / rho;
}

template <typename T>
__DEVICE__ inline T ProxIdentity(T v, T rho) {
  return v - 1 / rho;
}

template <typename T>
__DEVICE__ inline T ProxIndBox01(T v, T rho) {
  return v <= 0 ? 0 : v >= 1 ? 1 : v;
}

template <typename T>
__DEVICE__ inline T ProxIndEq0(T v, T rho) {
  return 0;
}

template <typename T>
__DEVICE__ inline T ProxIndGe0(T v, T rho) {
  return v <= 0 ? 0 : v;
}

template <typename T>
__DEVICE__ inline T ProxIndLe0(T v, T rho) {
  return v >= 0 ? 0 : v;
}

template <typename T>
__DEVICE__ inline T ProxLogistic(T v, T rho) {
  // Initial guess based on piecewise approximation.
  T x;
  if (v < static_cast<T>(-2.5))
    x = v;
  else if (v > static_cast<T>(2.5) + 1 / rho)
    x = v - 1 / rho;
  else
    x = (rho * v - static_cast<T>(0.5)) / (static_cast<T>(0.2) + rho);

  // Newton iteration.
  T l = v - 1 / rho, u = v;
  for (unsigned int i = 0; i < 5; ++i) {
    T inv_ex = 1 / (1 + Exp(-x));
    T f = inv_ex + rho * (x - v);
    T g = inv_ex * (1 - inv_ex) + rho;
    if (f < 0)
      l = x;
    else
      u = x;
    x = x - f / g;
    x = Min(x, u);
    x = Max(x, l);
  }

  // Guarded method if not converged.
  for (unsigned int i = 0; u - l > Tol<T>() && i < 100; ++i) {
    T g_rho = 1 / (rho * (1 + Exp(-x))) + (x - v);
    if (g_rho > 0) {
      l = Max(l, x - g_rho);
      u = x;
    } else {
      u = Min(u, x - g_rho);
      l = x;
    }
    x = (u + l) / 2;
  }
  return x;
}

template <typename T>
__DEVICE__ inline T ProxMaxNeg0(T v, T rho) {
  T z = v >= 0 ? v : 0;
  return v + 1 / rho <= 0 ? v + 1 / rho : z;
}

template <typename T>
__DEVICE__ inline T ProxMaxPos0(T v, T rho) {
  T z = v <= 0 ? v : 0;
  return v >= 1 / rho ? v - 1 / rho : z;
}

template <typename T>
__DEVICE__ inline T ProxNegLog(T v, T rho) {
  return (v + Sqrt(v * v + 4 / rho)) / 2;
}

template <typename T>
__DEVICE__ inline T ProxRecipr(T v, T rho) {
  v = Max(v, static_cast<T>(0));
  return CubicSolve(-v, static_cast<T>(0), -1 / rho);
}

template <typename T>
__DEVICE__ inline T ProxSquare(T v, T rho) {
  return rho * v / (1 + rho);
}

template <typename T>
__DEVICE__ inline T ProxZero(T v, T rho) {
  return v;
}

// Evaluates the proximal operator of f.
template <typename T>
__DEVICE__ inline T ProxEval(const FunctionObj<T> &f_obj, T v, T rho) {
  const T a = f_obj.a, b = f_obj.b, c = f_obj.c, d = f_obj.d, e = f_obj.e;
  v = a * (v * rho - d) / (e + rho) - b;
  rho = (e + rho) / (c * a * a);
  switch (f_obj.h) {
    case kAbs: v = ProxAbs(v, rho); break;
    case kNegEntr: v = ProxNegEntr(v, rho); break;
    case kExp: v = ProxExp(v, rho); break;
    case kHuber: v = ProxHuber(v, rho); break;
    case kIdentity: v = ProxIdentity(v, rho); break;
    case kIndBox01: v = ProxIndBox01(v, rho); break;
    case kIndEq0: v = ProxIndEq0(v, rho); break;
    case kIndGe0: v = ProxIndGe0(v, rho); break;
    case kIndLe0: v = ProxIndLe0(v, rho); break;
    case kLogistic: v = ProxLogistic(v, rho); break;
    case kMaxNeg0: v = ProxMaxNeg0(v, rho); break;
    case kMaxPos0: v = ProxMaxPos0(v, rho); break;
    case kNegLog: v = ProxNegLog(v, rho); break;
    case kRecipr: v = ProxRecipr(v, rho); break;
    case kSquare: v = ProxSquare(v, rho); break;
    case kZero: default: v = ProxZero(v, rho); break;
  }
  return (v + b) / a;
}


// Function definitions.
//
// Each of the following functions corresponds to one of the Function enums.
// All functions accept one argument x and four parameters (a, b, c, and d)
// and returns the evaluation of
//
//   x -> c * f(a * x - b) + d * x.
template <typename T>
__DEVICE__ inline T FuncAbs(T x) {
  return Abs(x);
}

template <typename T>
__DEVICE__ inline T FuncNegEntr(T x) {
  return x <= 0 ? 0 : x * Log(x);
}

template <typename T>
__DEVICE__ inline T FuncExp(T x) {
  return Exp(x);
}

template <typename T>
__DEVICE__ inline T FuncHuber(T x) {
  T xabs = Abs(x);
  T xabs2 = xabs * xabs;
  return xabs < static_cast<T>(1) ? xabs2 / 2 : xabs - static_cast<T>(0.5);
}

template <typename T>
__DEVICE__ inline T FuncIdentity(T x) {
  return x;
}

template <typename T>
__DEVICE__ inline T FuncIndBox01(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncIndEq0(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncIndGe0(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncIndLe0(T x) {
  return 0;
}

template <typename T>
__DEVICE__ inline T FuncLogistic(T x) {
  return Log(1 + Exp(x));
}

template <typename T>
__DEVICE__ inline T FuncMaxNeg0(T x) {
  return MaxNeg(x);
}

template <typename T>
__DEVICE__ inline T FuncMaxPos0(T x) {
  return MaxPos(x);
}

template <typename T>
__DEVICE__ inline T FuncNegLog(T x) {
  x = Max(static_cast<T>(0), x);
  return -Log(x);
}

template <typename T>
__DEVICE__ inline T FuncRecpr(T x) {
  x = Max(static_cast<T>(0), x);
  return 1 / x;
}

template <typename T>
__DEVICE__ inline T FuncSquare(T x) {
  return x * x / 2;
}

template <typename T>
__DEVICE__ inline T FuncZero(T x) {
  return 0;
}

// Evaluates the function f.
template <typename T>
__DEVICE__ inline T FuncEval(const FunctionObj<T> &f_obj, T x) {
  T dx = f_obj.d * x;
  T ex = f_obj.e * x * x / 2;
  x = f_obj.a * x - f_obj.b;
  switch (f_obj.h) {
    case kAbs: x = FuncAbs(x); break;
    case kNegEntr: x = FuncNegEntr(x); break;
    case kExp: x = FuncExp(x); break;
    case kHuber: x = FuncHuber(x); break;
    case kIdentity: x = FuncIdentity(x); break;
    case kIndBox01: x = FuncIndBox01(x); break;
    case kIndEq0: x = FuncIndEq0(x); break;
    case kIndGe0: x = FuncIndGe0(x); break;
    case kIndLe0: x = FuncIndLe0(x); break;
    case kLogistic: x = FuncLogistic(x); break;
    case kMaxNeg0: x = FuncMaxNeg0(x); break;
    case kMaxPos0: x = FuncMaxPos0(x); break;
    case kNegLog: x = FuncNegLog(x); break;
    case kRecipr: x = FuncRecpr(x); break;
    case kSquare: x = FuncSquare(x); break;
    case kZero: default: x = FuncZero(x); break;
  }
  return f_obj.c * x + dx + ex;
}


// Evaluates the proximal operator Prox{f_obj[i]}(x_in[i]) -> x_out[i].
//
// @param f_obj Vector of function objects.
// @param rho Penalty parameter.
// @param x_in Array to which proximal operator will be applied.
// @param x_out Array to which result will be written.
template <typename T>
void ProxEval(const std::vector<FunctionObj<T> > &f_obj, T rho, const T* x_in,
              size_t stride_in, T* x_out, size_t stride_out) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < f_obj.size(); ++i)
    x_out[i * stride_out] = ProxEval(f_obj[i], x_in[i * stride_in], rho);
}


// Returns evalution of Sum_i Func{f_obj[i]}(x_in[i]).
//
// @param f_obj Vector of function objects.
// @param x_in Array to which function will be applied.
// @param x_out Array to which result will be written.
// @returns Evaluation of sum of functions.
template <typename T>
T FuncEval(const std::vector<FunctionObj<T> > &f_obj, const T* x_in,
           size_t stride) {
  T sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
  for (unsigned int i = 0; i < f_obj.size(); ++i)
    sum += FuncEval(f_obj[i], x_in[i * stride]);
  return sum;
}

#ifdef __CUDACC__
// Strided iterator from thrust examples.
template <typename It>
class strided_range {
 public:
  typedef typename thrust::iterator_difference<It>::type diff_t;

  struct StrideF : public thrust::unary_function<diff_t, diff_t> {
    diff_t stride;
    StrideF(diff_t stride) : stride(stride) { }
    __host__ __device__
    diff_t operator()(const diff_t& i) const { 
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<diff_t> CountingIt;
  typedef typename thrust::transform_iterator<StrideF, CountingIt> TransformIt;
  typedef typename thrust::permutation_iterator<It, TransformIt> PermutationIt;
  typedef PermutationIt strided_iterator_t;

  // Construct strided_range for the range [first,last).
  strided_range(It first, It last, diff_t stride)
      : first(first), last(last), stride(stride) { }
 
  strided_iterator_t begin() const {
    return PermutationIt(first, TransformIt(CountingIt(0), StrideF(stride)));
  }

  strided_iterator_t end() const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }
  
 protected:
  It first;
  It last;
  diff_t stride;
};

template <typename T>
struct ProxEvalF : thrust::binary_function<FunctionObj<T>, T, T> {
  T rho;
  __device__ ProxEvalF(T rho) : rho(rho) { }
  __device__ T operator()(const FunctionObj<T> &f_obj, T x) {
    return ProxEval(f_obj, x, rho);
  }
};

template <typename T>
void ProxEval(const thrust::device_vector<FunctionObj<T> > &f_obj, T rho,
              const T *x_in, size_t stride_in, T *x_out, size_t stride_out) {
  size_t N = f_obj.size();
  strided_range<thrust::device_ptr<T> > x_in_strided(
      thrust::device_pointer_cast(const_cast<T*>(x_in)),
      thrust::device_pointer_cast(const_cast<T*>(x_in) + stride_in * N),
      stride_in);
  strided_range<thrust::device_ptr<T> > x_out_strided(
      thrust::device_pointer_cast(x_out),
      thrust::device_pointer_cast(x_out + stride_out * N), stride_out);
  thrust::transform(thrust::device, f_obj.cbegin(), f_obj.cend(),
       x_in_strided.begin(), x_out_strided.begin(), ProxEvalF<T>(rho));
}

template <typename T>
struct FuncEvalF : thrust::binary_function<FunctionObj<T>, T, T> {
  __device__ T operator()(const FunctionObj<T> &f_obj, T x) {
    return FuncEval(f_obj, x);
  }
};

template <typename T>
T FuncEval(const thrust::device_vector<FunctionObj<T> > &f_obj, const T *x_in,
           size_t stride) {
  strided_range<thrust::device_ptr<T> > x_in_strided(
      thrust::device_pointer_cast(const_cast<T*>(x_in)),
      thrust::device_pointer_cast(const_cast<T*>(x_in) + stride * f_obj.size()),
      stride);
  return thrust::inner_product(f_obj.cbegin(), f_obj.cend(),
      x_in_strided.begin(), static_cast<T>(0), thrust::plus<T>(),
      FuncEvalF<T>());
}

#endif  // __CUDACC__

#endif  // PROX_LIB_H_

