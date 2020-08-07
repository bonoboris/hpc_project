#pragma once
#include <vector>

#define MASTER 0

// Alias for a vector (default value type: float)
template <typename T=float>
using Vec = std::vector<T>;

// Alias for a matrix=vector<vector> (default value type: float)
template <typename T=float>
using Mat = Vec<Vec<T>>;

#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)

// Expand to a pair of iterator of Vec delimiting to range [beg, end)
// if beg or end are out of range, Vec boundary iterators are used instead.
#define RANGE(Vec, beg, end) Vec.begin()+std::max(0,beg), Vec.begin()+ std::min((int)Vec.size(),end)

#define IS_SCALAR(T, RT) std::enable_if_t<std::is_scalar_v<T>, RT>
#define IS_NOT_SCALAR(T, RT) std::enable_if_t< ! std::is_scalar_v<T>, RT>

#define printColSpacing 2
#define printDefaultPrecision 6
