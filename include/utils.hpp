// Various utilities related to vector, matrix generation printing as well as C types to MPI type correspondence

#pragma once

#include <functional>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "constants.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
/// Mat factory
/////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Mat<T> makeMat(size_t nbRows, size_t nbCols, T val = T())
{
	return Mat<T>(nbRows, Vec<T>(nbCols, val));
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Random vector and matrix generators
/////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class RandGen
{
public:
	static std::mt19937_64 gen;
	static std::uniform_real_distribution<T> dist;

	static T rand()
	{
		return dist(gen);
	}

	static Vec<T> randomVec(size_t n)
	{
		Vec<T> vec(n);
		std::generate(vec.begin(), vec.end(), rand);
		return vec;
	}

	static Mat<T> randomMat(size_t n, size_t m)
	{
		Mat<T> mat(n);
		std::generate(mat.begin(), mat.end(), std::bind(randomVec, m));
		return mat;
	}

	static Mat<T> randomMat(size_t n) { return randomMat(n, n); }

};

template<typename T>
std::mt19937_64 RandGen<T>::gen = std::mt19937_64(10);

template<typename T>
std::uniform_real_distribution<T> RandGen<T>::dist = std::uniform_real_distribution<T>(0.f, 1.f);

/////////////////////////////////////////////////////////////////////////////////////////////
/// Vec and Mat stream and string related
/////////////////////////////////////////////////////////////////////////////////////////////

template<typename It>
static size_t _findMaxStrLen(const It& begin, const It& end, size_t precision = 2)
{
	std::stringstream ss;
	ss << std::setprecision(precision);

	size_t maxStrLen = 0;
	for (auto it = begin; it < end; it++)
	{
		ss << *it;
		auto strLen = ss.str().size();
		if (strLen > maxStrLen)
			maxStrLen = strLen;
		ss.str("");
	}

	return maxStrLen;
}

template<typename It>
std::ostream& _insert_in_col(std::ostream& os, const It& begin, const It& end, size_t colWidth)
{
	for (auto it = begin; it < end; it++)
		os << std::setw(colWidth) << std::left << *it;
	return os;
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const Vec<T>& vec)
{
	if (vec.empty())
		return os;

	auto maxStrLen = _findMaxStrLen(vec.begin(), vec.end(), os.precision());

	return _insert_in_col(os, vec.begin(), vec.end(), maxStrLen + printColSpacing);
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const Mat<T>& mat)
{
	if (mat.empty())
		return os;

	size_t maxStrLen = 0;
	for (auto vec : mat)
	{
		auto candidate = _findMaxStrLen(vec.begin(), vec.end(), os.precision());
		if (candidate > maxStrLen)
			maxStrLen = candidate;
	}

	for (auto vec : mat)
		_insert_in_col(os, vec.begin(), vec.end(), maxStrLen + printColSpacing) << std::endl;

	return os;
}

template<typename Scal>
IS_SCALAR(Scal, std::string) toString(const Scal& val, size_t precision = printDefaultPrecision)
{
	std::stringstream ss;
	ss << std::setprecision(precision) << val;
	return ss.str();
}

template<typename T>
std::string toString(const Vec<T>& vec, size_t precision = printDefaultPrecision)
{
	if (vec.empty())
		return std::string();
	std::stringstream ss;
	ss << std::setprecision(precision) << vec;
	return ss.str();
}

template<typename T>
std::string toString(const Mat<T>& mat, size_t precision = printDefaultPrecision)
{
	if (mat.empty())
		return std::string();

	std::stringstream ss;
	ss << std::setprecision(precision) << mat;
	return ss.str();
}


/////////////////////////////////////////////////////////////////////////////////////////////
/// Cast MPI_Datatype and c++ type as a member typedef
/////////////////////////////////////////////////////////////////////////////////////////////

// // Forbid MPI_Datatype -> c++ type non specialized cast
// template<MPI_Datatype dtype>
// struct toType;

// Forbid c++ type -> MPI_Datatype non specialized cast
template<typename T>
MPI_Datatype fromType();

// Macro for specializing cast between pairs of type / MPI_Datatype
#define _CASTTYPE(mpi_type, cpp_type)									\
template<> MPI_Datatype fromType<cpp_type>() { return mpi_type; }		\
// template<> struct toType<mpi_type> {typedef typename cpp_type type;};

// List of authorized cast
_CASTTYPE(MPI_INT, int)
_CASTTYPE(MPI_FLOAT, float)
_CASTTYPE(MPI_DOUBLE, double)
_CASTTYPE(MPI_LONG_DOUBLE, long double)

#undef _CASTTYPE

// // Macro to cast MPI_Datatype into corresponding type
// #define TOTYPE(mpi_type) toType<mpi_type>::type
// Macro to cast MPI_Datatype into corresponding type
#define FROMTYPE(cpp_type) fromType<cpp_type>()