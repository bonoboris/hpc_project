// Distributed implementation of common operation on vectors and matrix using MPI

#pragma once

#include "MpiWrapper.hpp"
#include "Matrix.hpp"
#include "linalg.hpp"

#include <string>

// Assuming every process have their part of the input
// Return the scalar product (x.y)  for every process
template<typename T>
T mpiDot(const Params& p, const Vec<T>& x_part, const Vec<T>& y_part)
{
	auto part = linalg::dot(x_part, y_part);
	T dot;
	AllReduce(p, part, MPI_SUM, dot);

	return dot;
}

// Assuming every process have their part of the input
// Return the L2 norm of the vector for every process
template<typename T>
T mpiL2(const Params& p, const Vec<T>& x_part)
{
	return std::sqrt(mpiDot(p, x_part, x_part));
}

// Assuming every process have their part of the input, normalize
// Return ||x|| for every process
// Side effect: z_part contains x_part normalized (by normL2 of x)
template<typename T>
T mpiL2Normalize(const Params& p, const Vec<T>& x_part, Vec<T>& z_part)
{
	auto norm = mpiL2(p, x_part);
	z_part = linalg::scale(T(1) / norm, x_part);
	return norm;
}

// Perform matix vector multiplication z = Ax
// A is scattered on the all grid and x is scattered by rows
// When this function returns z is scattered by columns
template<typename T>
void mpiMatVec(const Params2D& p, const Matrix<T>& A_block, const Vec<T> x_part, Vec<T>& z_part)
{
	auto z_p = linalg::matVec(A_block, x_part);

	int I = p.rank() / p.numBlocksInRow(); // index of the row each process belong to
	int J = p.rank() % p.numBlocksInRow(); // index of the column each process belong to

	// reduce partial results by row to the process belonging to the diagonal
	Reduce(p.row2D(), z_p, MPI_SUM, z_part, I);
	// diagonal processes broadcast their part to their columns;
	Bcast(p.col2D(), z_part, J);
}

// Perform block matrix matrix product when each process of a 2d grid already have possession
// of their blocks.
template<typename T>
Params2D mpiMatMat(const Params2D& pA, const Matrix<T>& A_block, const Params2D& pB, const Matrix<T>& B_block, Matrix<T>& C_block)
{
	// 2 out of 3 dimensions are fixed 
	int m = pA.rowsBlock();
	int n = pB.colsBlock();

	C_block = Matrix<T>(m, n);

	// loop over the number of block in a row of A
	for (size_t k = 0; k < pA.numBlocksInRow(); k++)
	{
		// this dimension can vary with value of k
		int s = pA.colsBlock(k);  // should be equal to pB.rowsBlock()

		Matrix<T> A_p(m, s);
		Matrix<T> B_p(s, n);

		// If k == col rank bcast A_block to row
		if (pA.row2D().rank() == k)
			A_p = A_block;
		Bcast(pA.row2D(), A_p, k);

		// If k == row rank send B_block to col 
		if (pB.col2D().rank() == k)
			B_p = B_block;
		Bcast(pB.col2D(), B_p, k);

		// Compute blocks product and accumulate in C_block
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				for (int k = 0; k < s; k++)
					C_block.at(i,j) += A_p.at(i, k) * B_p.at(k, j);

		// Return a new params object associated with C
	}
	return Params2D::multParams(pA, pB);
}

// Transpose matrix: C = transpose(A)
// each process already has their block of A in A_block and finish with their transposed block in C_block
// return the Params asssociated with C
// RQ: only work with square grid of process
template<typename T>
Params2D mpiTranspose(const Params2D& p, const Matrix<T>& A_block, Matrix<T>& C_block)
{

	Params2D tp = p.transpose();
	std::cout.flush();

	C_block = Matrix<T>(tp.rowsBlock(), tp.colsBlock());
	// rank of the destination process in the global comm
	auto dstRank = p.row2D().rank() * p.numBlocksInRow() + p.col2D().rank();

	// Only extra diagonal process need to exchange data
	if (dstRank != p.rank())
		SendRcv(p, A_block.transpose(), C_block, dstRank);
	else
		C_block = A_block.transpose();

	return tp;
}