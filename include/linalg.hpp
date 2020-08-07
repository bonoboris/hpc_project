/// Templatized local common vector and matrix related operations.

#pragma once

#include "utils.hpp"
#include "Matrix.hpp"

#define DIM_ERR throw std::runtime_error("Invalid dimension(s) for operation!")

namespace linalg
{
	// z = x + y
	//	N: data size
	template<typename T>
	void add(size_t N, const T* px, const T* py, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = px[i] + py[i];
	}

	// return x + y
	template<typename T>
	Vec<T> add(const Vec<T>& x, const Vec<T>& y)
	{
		if (x.size() != y.size())
			DIM_ERR;
		Vec<T> z(x.size());
		add(x.size(), x.data(), y.data(), z.data());
		return z;
	}

	// z = x - y
	//	N: data size
	template<typename T>
	void sub(size_t N, const T* px, const T* py, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = px[i] - py[i];
	}

	// return x - y
	template<typename T>
	Vec<T> sub(const Vec<T>& x, const Vec<T>& y)
	{
		if (x.size() != y.size())
			DIM_ERR;
		Vec<T> z(x.size());
		sub(x.size(), x.data(), y.data(), z.data());
		return z;
	}

	// z = a*x
	//	N: data size
	template<typename T>
	void scale(size_t N, T a, const T* px, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = a * px[i];
	}

	// return a*x
	template<typename T>
	Vec<T> scale(T a, const Vec<T>& x)
	{
		Vec<T> z(x.size());
		scale(x.size(), a, x.data(), z.data());
		return z;
	}

	// z[i] = x[i] * y[i]
	//	N: data size
	template<typename T>
	void elementwise_mult(size_t N, const T* px, const T* py, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = py[i] * px[i];
	}

	template<typename T>
	Vec<T> elementwise_mult(const Vec<T>& x, const Vec<T>& y)
	{
		if (x.size() != y.size())
			DIM_ERR;
		Vec<T> z(x.size());
		elementwise_mult(x.size(), x.data(), y.data(), z.data());
		return z;
	}

	// z[i] = x[i] / y[i]
	//	N: data size
	template<typename T>
	void elementwise_div(size_t N, const T* px, const T* py, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = px[i] / py[i];
	}

	template<typename T>
	Vec<T> elementwise_div(const Vec<T>& x, const Vec<T>& y)
	{
		if (x.size() != y.size())
			DIM_ERR;
		Vec<T> z(x.size());
		elementwise_div(x.size(), x.data(), y.data(), z.data());
		return z;
	}

	// z[i] = 1 / x[i]
	//	N: data size
	template<typename T>
	void elementwise_inverse(size_t N, const T* px, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = 1 / px[i];
	}

	template<typename T>
	Vec<T> elementwise_inverse(const Vec<T>& x)
	{
		Vec<T> z(x.size());
		elementwise_div(x.size(), x.data(), z.data());
		return z;
	}

	// z = a*x + y 
	//	a scalar
	//	x [N]
	//	y [N]
	template<typename T>
	void axpy(size_t N, T a, const T* px, const T* py, T* pz)
	{
		for (auto i = 0; i < N; i++)
			pz[i] = a * px[i] + py[i];
	}

	// z = a*x +y
	template<typename T>
	Vec<T> axpy(T a, const Vec<T>& x, const Vec<T>& y)
	{
		if (x.size() != y.size())
			DIM_ERR;
		Vec<T> z(x.size());
		axpy(x.size(), a, x.data(), y.data(), z.data());
		return z;
	}

	// return x.y
	//	N: data size
	template<typename T>
	T dot(size_t N, const T* px, const T* py)
	{
		auto res = T();
		for (auto i = 0; i < N; i++)
			res += px[i] * py[i];
		return res;
	}


	// Strided version of dot product
	// Allow to compute dot product with column vectors inside row order matrix.
	template<typename T>
	T sdot(size_t N, const T* px, const T* py, size_t stride_x, size_t stride_y)
	{
		auto res = T();
		for (auto i = 0; i < N; i++)
			res += px[i*stride_x] * py[i*stride_y];
		return res;
	}

	// return x.y
	template<typename T>
	T dot(const Vec<T>& x, const Vec<T>& y)
	{
		return dot(x.size(), x.data(), y.data());
	}

	// return ||x|| (L2 vector norm)
	//	N: data size

	template<typename T>
	T L2(size_t N, const T* px)
	{
		auto res = T();
		for (auto i = 0; i < N; i++)
			res += px[i] * px[i];
		return std::sqrt(res);
	}

	// return x.y
	template<typename T>
	T L2(const Vec<T>& x)
	{
		return L2(x.size(), x.data());
	}

	// 
	template<typename T>
	void matVec(size_t M, size_t N, const T* pA, const T* px, T* pz)
	{
		for (auto i = 0; i < M; i++)
			pz[i] = dot(N, pA + i * N, px);
	}

	// return Ax
	template<typename T>
	Vec<T> matVec(const Matrix<T>& A, const Vec<T>& x)
	{
		if (A.cols() != x.size())
			DIM_ERR;
		Vec<T> z(A.rows());
		matVec(A.rows(), A.cols(), A.data(), x.data(), z.data());
		return z;
	}

	// return Ax
	template<typename T>
	Vec<T> matVec(const Mat<T>& A, const Vec<T>& x)
	{
		if (A.front().size() != x.size())
			DIM_ERR;
		Vec<T> z;
		for (const auto& row : A)
			z.push_back(dot(row, x));
		return z;
	}

	// C = AB
	//	C[M,N]
	//	A[M,K]
	//	B[K,N]
	// all row major
	template<typename T>
	void matMul(size_t M, size_t N, size_t K, const T* pA, const T* pB, T* pC)
	{
		for (auto i = 0; i < M; i++)
			for (auto j = 0; j < N; j++)
			{
				pC[i*N + j] = 0;
				for (auto k = 0; k < K; k++)
					pC[i*N + j] += pA[i*K + k] * pB[k*N + j];
			}
	}

	// return A*B
	template<typename T>
	Matrix<T> matMul(const Matrix<T>& A, const Matrix<T>& B)
	{
		Matrix<T> C(A.rows(), B.cols());
		if (A.cols() != B.rows())
			DIM_ERR;
		matMul(C.rows(), C.cols(), A.cols(), A.data(), B.data(), C.data());
		return C;
	}

	// return A*B
	template<typename T>
	Mat<T> matMul(const Mat<T>& A, const Mat<T>& B)
	{
		if (A.front().size() != B.size())
			DIM_ERR;

		auto M = A.size();
		auto N = B.front().size();
		auto K = B.size();

		auto C = makeMat<T>(M, N);
		for (auto i = 0; i < M; i++)
			for (auto j = 0; j < N; j++)
				for (auto k = 0; k < K; k++)
					C[i][j] += A[i][k] * B[k][j];
		return C;
	}

	// C = transpose(A)
	//	A[M,N]
	//	C[N,M]
	template<typename T>
	void transpose(size_t M, size_t N, const T* A, T* C)
	{
		for (auto i = 0; i < M; i++)
			for (auto j = 0; j < N; j++)
				C[j*M + i] = A[i*N + j];
	}

	template<typename T>
	Matrix<T> transpose(const Matrix<T> A)
	{
		Matrix<T> C(A.cols(), A.rows());
		transpose(A.rows(), A.cols(), A.data(), C.data());
		return C;
	}

	template<typename T>
	Mat<T> transpose(const Mat<T> A)
	{
		auto M = A.size();
		auto N = A.front().size();
		auto C = makeMat<T>(N, M);
		for (auto i = 0; i < M; i++)
			for (auto j = 0; j < N; j++)
				C[j][i] = A[i][j];
		return C;
	}
}