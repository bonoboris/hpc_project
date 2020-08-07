/// Implementation of Lanczos and gradient methods to solve symmetric, positive, definite linear system of equations
/// Main script generate and solve such a problem using both methods in local and distributed fashion then compare the results.  

#include "MpiRoutines.hpp"
#include "linalg.hpp"

#include <iostream>
#include <type_traits>
#include <fstream>

#define SIZE 100
#define ERR_THRESH 10e-10

#define LANCZOS_Y_ZEROS true

#define PRINT_Xp false

using namespace std;


template<typename Real_T>
struct ResStruct
{
	Vec<Real_T> x;
	Vec<Real_T> errors;

	ResStruct() = default;
	ResStruct(const Vec<Real_T>& x, const Vec<Real_T>& errors) : x(x), errors(errors) {};
};


template<typename Real_T>
ResStruct<Real_T> LanczosLocal(const Mat<Real_T>& A, const Vec<Real_T>& b, const Vec<Real_T>& x0)
{
	auto N = A.size();

	//// Variable definitions

	Mat<Real_T>
		V_t;	// transposed matrix of Lanczos orthonormal base: V_t
	Vec<Real_T>
		// T (when fully compute ) is a (N, N+1) matrix where it principal diagonal block T'verify T' = Vt A V
		T_d,		// Diagonal coefficients of matrix T = AV 
		T_e,		// Extra diagonal coefficient of T
		// Crout factorization of T: T = LDLt
		L_e,		// L_e: Lower diagonal of bi-dialgonal lower triangular matrix L of Crout factorization with diagonal term = 1
		D_d,		// D_d: Diagonal of diagonal matrix D of Crout factorization
		y,			// y = Vt g0
		u,			// u verify L
		z,			// 
		x,
		g,
		w,		// Used as intermediate for successive computation Lanczos base vectors
		g0,		// Initial residual g0 = Ax0 - b
		residualErrors;		

	auto norm_b = linalg::L2(b);

	g0 = linalg::sub(linalg::matVec(A, x0), b);
	auto norm_g0 = linalg::L2(g0);

	residualErrors.push_back(norm_g0 / norm_b); // initial error

	w = linalg::scale(1 / norm_g0, g0);
	V_t.push_back(w);
	y.push_back(-norm_g0);
	u.push_back(y.back());

	w = linalg::matVec(A, V_t.back());

	T_d.push_back(linalg::dot(V_t.back(), w));
	D_d.push_back(T_d.back());
	
	// Following the formula, the first approx is 
	x = linalg::axpy(y.back()/T_d.back(), V_t.back(), x0);
	g = linalg::sub(linalg::matVec(A, x), b);
	auto err = linalg::L2(g) / norm_b;
	residualErrors.push_back(err);

	w = linalg::axpy(-T_d.back(), V_t.back(), w);

	for (auto k = 2; k < N; k++)
	{
		T_e.push_back(linalg::L2(w));
		w = linalg::scale(1 / T_e.back(), w);

		V_t.push_back(w);
		
		LANCZOS_Y_ZEROS ? y.push_back(0.) : y.push_back(-linalg::dot(w, g0));

		w = linalg::matVec(A, w);

		T_d.push_back(linalg::dot(V_t.back(), w));

		L_e.push_back(T_e.back() / D_d.back());
		D_d.push_back(T_d.back() - L_e.back()*T_e.back());

		u.push_back(y.back() - L_e.back()*u.back());

		z.resize(k);
		z.back() = u.back() / D_d.back();
		for (auto i = k - 2; i >= 0; i--)
			z[i] = u[i] / D_d[i] - L_e[i] * z[i + 1];

		x = linalg::add(x0, linalg::matVec(linalg::transpose(V_t), z));
		if (PRINT_Xp)
			cout << "Iteration " << k - 1 << endl << x << endl;

		g = linalg::sub(linalg::matVec(A, x), b);
		auto err = linalg::L2(g) / norm_b;
		residualErrors.push_back(err);
		if (err < ERR_THRESH)
			break;

		w = linalg::axpy(-T_e.back(), V_t[V_t.size()-2], w);
		w = linalg::axpy(-T_d.back(), V_t.back(), w);
	}

	return ResStruct<Real_T>( x, residualErrors );
}

template<typename Real_T>
ResStruct<Real_T> LanczosMpi(const Params2D& p, const BlockMatrix<Real_T>& A_tot, const Vec<Real_T>& b_tot, const Vec<Real_T>& x0_tot)
{
	Params pCol = p.col1D();
	Params pRow = p.row1D();

	auto N = p.cols();
	auto m = p.rowsBlock();
	auto n = p.colsBlock();

	/*	Except mentioned otherwise for every vector/matrix "obj" associated with variable "Obj" and "Obj_tot":
			- Obj_tot refers to the full representation of obj
			- Obj refers to the part/block owned by a process		
	*/
	///// Scatering inputs

	Matrix<Real_T> A(m, n);
	Vec<Real_T> b(n), x0(n);

	// send each process its block
	Scatterv(p, A_tot, A);

	// scatter vecs to each col master
	if (pCol.isMaster())
	{
		Scatterv(pRow, x0_tot, x0);
		Scatterv(pRow, b_tot, b);
	}
	// each col master broadcast it part to the rest of the col
	

	Bcast(pCol, x0);
	Bcast(pCol, b);

	//// Variable definitions

	// transposed matrix of Lanczos orthonormal base: V_t
	// The ending _ is because this variable is an intermediate, in the end each process should hold a block of Lanczos basis in a Matrix
	Mat<Real_T> Vt_;	
	
	Vec<Real_T> // All those vectors are built terms by terms by all process, at any moment all process have full knoledge of already computed terms
		// T (when fully compute ) verify T = Vt A V
		T_d,						// Diagonal coefficients of matrix T = AV 
		T_e,						// Extra diagonal coefficient of T

		// Crout factorization of T: T = L D Lt
		D_d,						// D_d: Diagonal terms of diagonal matrix D
		L_e,						// L_e: Lower diagonal of bi-dialgonal lower triangular matrix L, where L diagonal term = 1

		y,							// y = Vg0
		u,							// u verify L u = y
		z(n),						// z verify LDLt z = y and x0 + V.z is an approx solution of the problem

		a_rec = { Real_T(1) },		// multiplicative factors in recurrence needed to compute z
		b_rec = { Real_T(0) },		// additive factors in recurrence needed to compute z

		residualErrors;						// vector of the residual errors at each iteration (counting initial error)

	Vec<Real_T>
		w(n),						// Used as intermediate for successive computation Lanczos base vectors
		g0(n),						// Initial residual g0 = Ax0 - b
		x(n),						// approximate solution (update each iteration)
		Ax(n);						// Hold the image of the approximate solution by A

	//// Computations

	auto norm_b = mpiL2(pRow, b);

	mpiMatVec(p, A, x0, Ax);
	g0 = linalg::sub(Ax, b);
	auto norm_g0 = mpiL2Normalize(pRow, g0, w);
	
	residualErrors.push_back(norm_g0 / norm_b);

	Vt_.push_back(w);

	y.push_back(-norm_g0);

	mpiMatVec(p, A, w, w);
	T_d.push_back(mpiDot(pRow, Vt_.back(), w));
	D_d.push_back(T_d.back());
	
	u.push_back(y.back());

	// Following the formula, the first approx is 
	x = linalg::axpy(y.back() / T_d.back(), Vt_.back(), x0);
	mpiMatVec(p, A, x, Ax);
	auto err = mpiL2(pRow, linalg::sub(Ax, b)) / norm_b;
	residualErrors.push_back(err);

	w = linalg::axpy(-T_d.back(), Vt_.back(), w);

	for (auto k = 2; k < N; k++)
	{
		T_e.push_back(mpiL2Normalize(pRow, w, w));

		Vt_.push_back(w);

		// In exact arithmetic every component of y except for its first one should be zeros
		LANCZOS_Y_ZEROS ? y.push_back(0.) : y.push_back(-mpiDot(pRow, w, g0)); 
		
		mpiMatVec(p, A, w, w);

		T_d.push_back(mpiDot(pRow, Vt_.back(), w));

		L_e.push_back(T_e.back() / D_d.back());

		a_rec.push_back(-L_e.back());
		b_rec.push_back(u.back() / D_d.back());

		D_d.push_back(T_d.back() - L_e.back()*T_e.back());

		u.push_back(y.back() - L_e.back()*u.back());

		z = solveAscendingRec(p, u.back() / D_d.back(), a_rec, b_rec);

		x = linalg::add(x0, computeVz(p, Vt_, z));

		mpiMatVec(p, A, x, Ax);
		auto err = mpiL2(pRow, linalg::sub(Ax, b)) / norm_b;
		residualErrors.push_back(err);
		if (err < ERR_THRESH)
			break;

		w = linalg::axpy(-T_e.back(), Vt_[Vt_.size() -2], w);
		w = linalg::axpy(-T_d.back(), Vt_.back(), w);
	}

	Vec<Real_T>x_tot(N);
	Gatherv(pRow, x, x_tot);
	
	ResStruct<Real_T> resStruct;
	if (p.isMaster())
	{
		resStruct.x = x_tot;
		resStruct.errors = residualErrors;
	}

	return resStruct;
}

// Solve K term of "ascending" recurrence where:
// z[K-1] = zLast
// for k in [0;K-1[
//		z[k[ = b[k] + a[k] * z[k+1]
// 
// Each process in a given column 0 <= J < P return the Jth part of the solution z;
//
// a and b vectors are of size K, their first term are ignored
template<typename Real_T>
Vec<Real_T> solveAscendingRec(const Params2D& p, Real_T zLast, const Vec<Real_T>& a, const Vec<Real_T>& b)
{
	// The recurrence is solved by solving 2 sub recurrences
	// First the last process (P-1, P-1) computes the last terms of sub-sub range for last process of each column (P-1, ...)
	// Last process of each columns receive their last term and compute every last terms of the sub-sub range for every process of their column
	// Each process computes terms of their sub-sub range
	// The computed sub-sub range are then allgather by column into regular sub range (associated with a 1D Params of range size K)

	//// For every communicator the "master" will be the last process.
	//// Create 1D params adapted to the problem size
	Params pRow(p.row1D().comm(), a.size(), p.row1D().commSize()-1);	// communicator for last row 
	Params pCol(p.col1D().comm(), pRow.len(), pRow.commSize()-1);		// sub communicator in col


	// Let's take care of small recurences with direct computation for every process
	if (pRow.rangeSize() < 2 * p.commSize())
	{
		// we need to broad cast zLast, which is in possession of the last process, to every process
		Bcast(p, zLast, p.commSize()-1);
		Vec<Real_T> direct_z(pRow.rangeSize() - pRow.start());
		direct_z.back() = zLast;

		for (auto i = pRow.rangeSize() - pRow.start() - 1; i > 0; i--)
			direct_z[i-1] = b[i + pRow.start()] + a[i + pRow.start()] * direct_z[i];

		return Vec<Real_T>(RANGE(direct_z, 0, pRow.len()));
	}


	// Each columns is going to solve the recurence for its part of the result vector

	// First we reduce the size of the recurence by the total number of process 
	Real_T A = 1;
	Real_T B = 0;


	cout.flush();
	for (int i = pRow.start() + pCol.end() - 1; i >= pRow.start() + pCol.start(); i--)
	{
		A *= a[i];
		B = b[i] + a[i]*B;
	}

	// We Gather the coeffs A and B to their column master
	Vec<Real_T> A_col(pCol.commSize());
	Vec<Real_T> B_col(pCol.commSize());
	Vec<Real_T> Z_col(pCol.commSize());
	
	Gather(pCol, A, A_col);
	Gather(pCol, B, B_col);

	// Columns master reduce their coef even more
	if (pCol.isMaster())
	{
		A = 1;
		B = 0;
		for (int i = A_col.size()-1; i >= 0; i--)
		{
			A *= A_col[i];
			B = B_col[i] + A_col[i] * B;
		}

		// We gather the coeff A and B to the global master
		Vec<Real_T> A_all(pRow.commSize());
		Vec<Real_T> B_all(pRow.commSize());
		Vec<Real_T> Z_all(pRow.commSize());
		
		Gather(pRow, A, A_all);
		Gather(pRow, B, B_all);

		// Master computes first terms for each columns
		if (pRow.isMaster())
		{
			Z_all.back() = zLast;
			for (int i = Z_all.size() - 1; i > 0; i--)
				Z_all[i - 1] = B_all[i] + A_all[i] * Z_all[i];
		}

		// Scatter the terms accross the columns master
		Scatter(pRow, Z_all, zLast);

		// Each columns master computes first terms for their slave processes
		Z_col.back() = zLast;
		for (int i = Z_col.size() - 1; i > 0; i--)
			Z_col[i - 1] = B_col[i] + A_col[i] * Z_col[i];
	}

	// Scatter the terms accross their slaves
	Scatter(pCol, Z_col, zLast);

	//cout << "Process " << p.rank() << " " << pCol.start() << " " << pCol.end() << " " << pRow.start() << " " << pRow.end() << endl;

	// Computes terms
	Vec<Real_T> z_sub(pCol.len());
	z_sub.back() = zLast;
	for (int i = pCol.len() - 1; i > 0; i--)
		z_sub[i - 1] = b[pRow.start() + pCol.start() + i] + a[pRow.start() + pCol.start() + i] * z_sub[i];
	
	// Gather the subpart into bigger parts by columns
	Vec<Real_T> z(pRow.len());
	AllGatherv(pCol, z_sub, z);
	return z;
}

// Compute Matrix product V z as appearing in lanczos algoritm (where V a free family of Lanczos vectors of size K and z is the solution to T z = Vt g0
template<typename Real_T>
Vec<Real_T> computeVz(const Params2D& params, const Mat<Real_T>& Vt_, const Vec<Real_T>& z_part)
{
	Vec<Real_T> res(params.colsBlock());
	Params2D pt(params.row1D().comm(), params.col1D().comm(), Vt_.size(), params.cols(), params.comm());

	// At this point for a given process (I,J)
	// Vt_ holds every parts of the already lanczos vectors associate to col J

	// Vt contains just the block of lanczos in a 2D grid
	Matrix<Real_T> Vt(Mat<Real_T>(Vt_.begin() + pt.col1D().start(), Vt_.begin() + pt.col1D().end()));

	Matrix<Real_T> V;
	Params2D p = mpiTranspose(pt, Vt, V);

	mpiMatVec(p, V, z_part, res);
	return res;
}

template<typename Real_T>
ResStruct<Real_T> ConjugateGradientLocal(const Mat<Real_T>& A, const Vec<Real_T>& b, const Vec<Real_T>& x0)
{
	auto N = A.size();
	
	//// Variable definitions
	Vec<Real_T>
		// Both basis generated during the conjugate gradient method are not necessary for the resolution, we just need to store the last computed vectors
		g,		// last computed residual / orthogonal basis vector
		w,		// last computed "descent" basis vector 
		
		v,		// itermediate of computation
		x,		// approximate solution (updated every iteration)
		residualErrors;		// vector of the residual errors at each iteration (counting initial error)
	
	Real_T
		rho,		// Diagonal terms of the diagonal matrix L where L verify W = G L-t
		gamma,
		v_dot_w;

	//// Computations

	auto norm_b = linalg::L2(b);

	g = linalg::sub(linalg::matVec(A, x0), b);
	w = g;
	x = x0;
	residualErrors.push_back(linalg::L2(g) / norm_b); // initial residual error

	for (auto k = 1; k < N; k++)
	{
		v = linalg::matVec(A, w);
		v_dot_w = linalg::dot(v, w);
		rho = - linalg::dot(g, w) / v_dot_w;
		x = linalg::axpy(rho, w, x);
		
		if (PRINT_Xp)
			cout << "Iteration " << k - 1 << endl << x << endl;

		g = linalg::axpy(rho, v, g);

		auto err = linalg::L2(g) / norm_b;
		residualErrors.push_back(err);
		if (err < ERR_THRESH)
			break;
		gamma = -linalg::dot(g, v) / v_dot_w;
		w = linalg::axpy(gamma, w, g);
	}

	return ResStruct<Real_T>(x, residualErrors);
}

template<typename Real_T>
ResStruct<Real_T> ConjugateGradientMpi(const Params2D& p, const BlockMatrix<Real_T>& A_tot, const Vec<Real_T>& b_tot, const Vec<Real_T>& x0_tot)
{
	Params pCol = p.col1D();
	Params pRow = p.row1D();

	auto N = p.cols();
	auto m = p.rowsBlock();
	auto n = p.colsBlock();

	/*	Except mentioned otherwise for every vector/matrix "obj" associated with variable "Obj" and "Obj_tot":
			- Obj_tot refers to the full representation of obj
			- Obj refers to the part/block owned by a process
	*/

	///// Scatering inputs

	Matrix<Real_T> A(m, n);
	Vec<Real_T> b(n), x0(n);

	// send each process its block
	Scatterv(p, A_tot, A);

	// scatter vecs to each col master
	if (pCol.isMaster())
	{
		Scatterv(pRow, x0_tot, x0);
		Scatterv(pRow, b_tot, b);
	}
	// each col master broadcast it part to the rest of the col
	Bcast(pCol, x0);
	Bcast(pCol, b);

	//// Variable definitions

	Vec<Real_T>
		g(n),
		v(n),
		w(n),
		x(n),
		Ax(n),
		errors;
	Real_T
		rho,
		gamma,
		v_dot_w;

	//// Computations
	
	auto norm_b = mpiL2(pRow, b);

	mpiMatVec(p, A, x0, g);
	g = linalg::sub(g, b);

	errors.push_back(mpiL2(pRow, g) / norm_b); // initial residual error

	w = g;
	x = x0;
	for (auto k = 1; k < N; k++)
	{
		mpiMatVec(p, A, w, v);
		v_dot_w = mpiDot(pRow, v, w);
		rho = -mpiDot(pRow, g, w) / v_dot_w;
		x = linalg::axpy(rho, w, x);
		g = linalg::axpy(rho, v, g);
		auto err = mpiL2(pRow, g) / norm_b;
		errors.push_back(err);
		if (err < ERR_THRESH)
			break;
		gamma = -mpiDot(pRow, g, v) / v_dot_w;
		w = linalg::axpy(gamma, w, g);
	}

	mpiMatVec(p, A, x, Ax);
	Vec<Real_T> x_tot(N);
	Gatherv(pRow, x, x_tot);
	cout.flush();

	ResStruct<Real_T> resStruct;
	if (p.isMaster())
	{
		resStruct.x = x_tot;
		resStruct.errors = errors;

	}
	return resStruct;
}


int main(int argc, char* argv[])
{
	using Real_T = float;

	MPI_Init(&argc, &argv);

	cout << setprecision(6);

	int worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// size of the process square
	int subcommSize = static_cast<int>(floor(sqrt(worldSize)));

	// Create a subcommSize * subcommSize grid of process and their global, row and columns communicators
	// associated to a SIZE * SIZE square matrix
	Params2D p = Params2D::makeGrid(MPI_COMM_WORLD, subcommSize, SIZE);

	// Discard processes that don't fit into the grid
	if (p.isEmptyParams())
	{
		MPI_Finalize();
		return 0;
	}
	//// Inputs declaration
	BlockMatrix<Real_T> A;
	Vec<Real_T> b, x0, x_true;


	//// Inputs definition
	if (p.isMaster())
	{
		cout << "Using floattant type with " << sizeof(Real_T) << " bytes." << endl;
		A = BlockMatrix<Real_T>(p);
		A.randomSPD(0);

		//cout << A << std::endl << std::endl;
		x_true = RandGen<Real_T>::randomVec(SIZE);
		b = linalg::matVec(A.asMatrix(), x_true);
		x0 = RandGen<Real_T>::randomVec(SIZE);
	}

	//// Structs to store solution and succesive errors

	ResStruct<Real_T>
		lanczosMpiRes, lanczosLocalRes,
		conjugateGradientLocalRes, conjugateGradientMpiRes;
	
	// Only relevant return values for master
	lanczosMpiRes = LanczosMpi<Real_T>(p, A, b, x0);

	// if (p.rank() == 1)
	// 	cout << p << std::endl;
	// MPI_Finalize();
	// return 0;

	conjugateGradientMpiRes = ConjugateGradientMpi<Real_T>(p, A, b, x0);

	if (p.isMaster())
	{
		cout << "Lanczos Mpi: (" << lanczosMpiRes.errors.size() - 1 << " Iterations)" << endl
			<< "Normalized residual errors:\n\t" << lanczosMpiRes.errors << endl; //<< "Solution:\n\t" << lanczosMpiRes.x << endl << endl;

		cout << "Conjugate Gradient Mpi: (" << conjugateGradientMpiRes.errors.size() - 1 << " Iterations)" << endl
			<< "Normalized residual errors:\n\t" << conjugateGradientMpiRes.errors << endl; //<< "Solution:\n\t" << conjugateGradientMpiRes.x << endl << endl;

		lanczosLocalRes = LanczosLocal<Real_T>(A.asMat(), b, x0);
		conjugateGradientLocalRes = ConjugateGradientLocal<Real_T>(A.asMat(), b, x0);

		cout << "Lanczos Local: (" << lanczosLocalRes.errors.size() - 1 << " Iterations)" << endl
			<< "Normalized residual errors:\n\t" << lanczosLocalRes.errors << endl; //<< "Solution:\n\t" << lanczosLocalRes.x << endl << endl;

		cout << "Conjugate Gradient Local: (" << conjugateGradientLocalRes.errors.size() - 1 << " Iterations)" << endl
			<< "Normalized residual errors:\n\t" << conjugateGradientLocalRes.errors << endl; //<< "Solution:\n\t" << conjugateGradientLocalRes.x << endl << endl;

	}
	MPI_Finalize();

	return 0;
}