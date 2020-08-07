// Classes to hold parameters of a thread in the distributed system

#pragma once

#include "utils.hpp"  

#include <mpi.h>

#include <iostream>
#include <thread>

class Params;
class Params2D;
Params scaleRange(const Params&, int);

// Hold a process information within a communicator.
// Associate a 1D range of a given size to this communicator and split this range
// in N approx equal parts, where N is the number of process in the communicator
// Provides useful methods for MPI communication functions.
class Params {
	friend Params2D;
	friend Params scaleRange(const Params&, int);
protected:
	int _commRank;			//< Rank of the process in the communicator
	int _commSize;			//< Size of the communicator
	MPI_Comm _comm;			//< Communicator
	int _commMaster;		//< Rank of the master in the communicator
	int _rangeSize;			//< Number of element associted with the communicator

	Vec<int> _displs;			//< Starting indexes of the sub domains of every process in the communicator
	Vec<int> _counts;			//< Lenghts of the sub domains of every process in the communicator

public:
	Params() = default;
	Params(MPI_Comm comm, int rangeSize, int master = MASTER) : _comm(comm), _rangeSize(rangeSize), _commMaster(master)
	{
		MPI_Comm_rank(_comm, &_commRank);
		MPI_Comm_size(_comm, &_commSize);
		
		int q = _rangeSize / _commSize,
			r = _rangeSize % _commSize;
		for (auto i = 0; i < _commSize; ++i)
		{
			// displs[0] = 0
			// displs[i+1] = displs[i] + counts[i]
			_displs.push_back((i == 0) ? 0 : _displs.back() + _counts.back());
			_counts.push_back(i < r ? q + 1 : q);
		}
	}

	int rank() const { return _commRank; }
	int commSize() const { return _commSize; }
	int rangeSize() const { return _rangeSize; }
	MPI_Comm comm() const { return _comm; }
	int master() const { return _commMaster; }

	bool isEmptyParams() const { return _displs.empty(); }

	bool isMaster() const { return _commRank == _commMaster; }
	bool isLast() const { return _commRank == _commSize - 1; }

	const int* displs() const { return _displs.data(); }
	const Vec<int>& displs_v() const { return _displs; }
	const int* counts() const { return _counts.data(); }
	const Vec<int>& counts_v() const { return _counts; }

	// This process 1D sub domain start index (included)
	int start() const { return _displs[_commRank]; }

	// The process with given rank 1D sub domain start index (included)
	int start(int rank) const { return _displs[rank]; }

	// This process 1D sub domain length
	int len() const { return _counts[_commRank]; }

	// The process with given rank 1D sub domain length
	int len(int rank) const { return _counts[rank]; }
	
	// This process 1D sub domain end index (excluded)
	int end() const { return _displs[_commRank] + _counts[_commRank]; }

	// The process with given rank 1D sub domain end index (excluded)
	int end(int rank) const { return _displs[rank] + _counts[rank]; }
};

// Scale the domain and sub domains data members
//	inputs: params, val = 3;
//		ex: params.counts() = {5,5,5,4,4}
//	output: .counts() = {15,15,15,12,12}
Params scaleRange(const Params& params, int val)
{
	Params scaledParams = params;

	scaledParams._counts.clear();
	scaledParams._displs.clear();

	scaledParams._rangeSize *= val;

	std::transform(params._counts.begin(), params._counts.end(), std::back_inserter(scaledParams._counts),
		[&val](int count) {return count * val; });
	
	std::transform(params._displs.begin(), params._displs.end(), std::back_inserter(scaledParams._displs),
		[&val](int displ) {return displ * val; });

	return scaledParams;
}

// Extend the functionnalities of Params class for 2D grid of process where
// each row and each column have its communicator; as well as a global communicator
// This class inherited members refers to a global communicator, in which all
// process of the grid belong to and are ordered in row major fashion.
// Define 4 new member _row2D and _col2D which hold the Params for a process row
// communicator and column comminucator,
// and _row1D and _col1D, their 1D dimension counterpart, to use when working with 1D vectors
// and match the matrix split.
class Params2D : public Params
{
protected:
	Params _row2D;	//< Params for the row communicator this process belongs to
	Params _col2D;	//< Params for the colum communicator this process belongs to
	Params _row1D;	//< Params for 1D data exchange (of a row vector) on the row communicator this process belongs to
	Params _col1D;	//< Params for 1D data exchange (of a col vector) on the col communicator this process belongs to
	Params _1D;
public:
	Params2D() = default;
	
	Params2D(MPI_Comm row_comm, MPI_Comm col_comm, int nbRows, int nbCols, MPI_Comm global_comm,
		int row_master = MASTER, int col_master = MASTER, int global_master = MASTER):
		_row1D(Params(row_comm, nbCols, row_master)), _col1D(Params(col_comm, nbRows, col_master))
	{
		// _row1D and _col1D Params members are initialized with a 1D domain partition of the range [0, M), [0,N)
		// where N and M are the dimensions of the 2D domain assciated with the process grid 
		// We use this information to compute global params

		_comm = global_comm;
		_commMaster = global_master;

		MPI_Comm_rank(global_comm, &_commRank);
		MPI_Comm_size(global_comm, &_commSize);

		_rangeSize = nbRows * nbCols;

		for (auto rowCnt: _row1D._counts)
			for (auto colCnt: _col1D._counts)
			{
				// displs[0] = 0
				// displs[k+1] = displs[k] + counts[k]
				_displs.push_back(_displs.empty() ? 0 : _displs.back() + _counts.back());
				_counts.push_back(rowCnt * colCnt);
			}
		
		// Now we can derive _row2D and _col2D from _row1D and _col1D

		_row2D = scaleRange(_row1D, _col1D.len());
		_col2D = scaleRange(_col1D, _row1D.len());
	}
	
	//////// Sub comunicator////////

	// return Params for block exchange on this process row
	const Params& row2D() const { return _row2D; }

	// return Params for block exchange on this process col
	const Params& col2D() const { return _col2D; }

	// return Params for vec exchange on this process row
	const Params& row1D() const { return _row1D; }
	
	// return Params for vec exchange on this process col
	const Params& col1D() const { return _col1D; }

	//////// Process grid dimensions / Number of block ////////

	const int numBlocksInRow() const { return _row2D.commSize(); }
	const int numBlocksInCol() const { return _col2D.commSize(); }

	//////// 2D array dimensions ////////

	// return the total number of rows 
	const int rows() const { return _col1D.rangeSize(); }

	// return the total number of cols 
	const int cols() const { return _row1D.rangeSize(); }

	//////// Block dimensions ////////

	// return the number of rows of this process block
	const int rowsBlock() const { return _col1D.len(); }

	// return the number of rows of blocks with given row index
	const int rowsBlock(int rowBlockIndex) const { return _col1D.len(rowBlockIndex); }

	// return the number of cols of this process block
	const int colsBlock() const { return _row1D.len(); }

	// return the number of cols of blocks with given col index
	const int colsBlock(int colBlockIndex) const { return _row1D.len(colBlockIndex); }

	// if this Params2D is associated with a MxN matrix return a Params objcect associated with the transposed matrix NxM
	Params2D transpose() const
	{
		return Params2D(_row2D._comm, _col2D._comm, _row1D._rangeSize, _col1D._rangeSize, _comm, _row2D._commMaster, _col2D._commMaster, _commMaster);
	}
	
	MPI_Comm createColMajor1DComm() const
	{
		MPI_Comm comm;
		MPI_Comm_split(_comm, 0, _row2D.rank() * _row2D.commSize() + _col2D.rank(), &comm);
		return comm;
	}

	// Create a (gridNbRows, gridNbCols) grid of paramater from the first processes in comm
	// And return Params2D object  created from the grid and the 2D range [0,nbRows)*[0nbCols)
	static Params2D makeGrid(MPI_Comm srcComm, int gridNbRows, int gridNbCols, int nbRows, int nbCols)
	{

		int srcCommRank;
		MPI_Comm_rank(srcComm, &srcCommRank);

		// Discard extra processes by giving them undefined color
		auto globalColor = (srcCommRank < gridNbRows * gridNbCols) ? 0 : MPI_UNDEFINED;

		// Make a comminucator for valid process
		MPI_Comm globalComm;
		MPI_Comm_split(srcComm, globalColor, srcCommRank, &globalComm);

		// Return empty params for extra processed 
		if (globalComm == MPI_COMM_NULL)
			return Params2D();

		MPI_Comm rowComm, colComm;
		
		int rowColor = srcCommRank / gridNbCols; // process row index
		int colColor = srcCommRank % gridNbCols; // process col index
		
		MPI_Comm_split(globalComm, rowColor, colColor, &rowComm);
		MPI_Comm_split(globalComm, colColor, rowColor, &colComm);
		
		//std::cout << toString(globalComm) + " " + toString(rowComm) + " " + toString(colComm) + "\n";

		return Params2D (rowComm, colComm, nbRows, nbCols, globalComm, 0, 0, 0);
	}

	// Create a (gridSize, gridSize) grid of paramater from the first processes in comm
	// And return Params2D object  created from the grid and the 2D range [0,nbRows)*[0nbCols)
	// if nbCols is not supplied, nbCols = nbRows
	static Params2D makeGrid(MPI_Comm srcComm, int gridSize, int nbRows, int nbCols = -1)
	{
		if (nbCols == -1)
			nbCols = nbRows;
		return makeGrid(srcComm, gridSize, gridSize, nbRows, nbCols);
	}

	// If pA is associated to a matrix A of dim MxS and pB to a matrix B of dim SxN
	// this function return the params associated to the matrix C = AB of dim MxN
	// (Use values of a for global communicator and its master)
	static Params2D multParams(const Params2D& pA, const Params2D& pB)
	{
		auto rpA = pA._row1D;
		auto cpB = pB._col1D;
		return Params2D(rpA._comm, cpB._comm, rpA._rangeSize, cpB._rangeSize, pA._comm, rpA._commMaster, cpB._commMaster, pA._commMaster);
	}

};

std::ostream& operator<<(std::ostream& os, const Params& params)
{
	os << "Comm: " << params.comm() << "\tSize: " << params.commSize() << "\tRank: " << params.rank() << "\tMaster: " << params.master() << std::endl
		<< "RangeSize: " << params.rangeSize() << std::endl
		<< "Counts: " << params.counts_v() << std::endl
		<< "Displs: " << params.displs_v() << std::endl;
	return os;
}

std::ostream& operator<<(std::ostream& os, const Params2D& params)
{
	os << "GLOBAL" << std::endl << static_cast<Params>(params) << std::endl
		<< "ROW" << std::endl << params.row2D() << std::endl
		<< "COL" << std::endl << params.col2D() << std::endl;
	return os;
}