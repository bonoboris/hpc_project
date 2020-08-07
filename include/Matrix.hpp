// Implements 2 representation of a matrix
// - Matrix: row major container
// - BlockMatrix: N*M blocks where each block is a row major container

#pragma once

#include <exception>
#include <iomanip>
#include <string>
#include <ostream>
#include <iostream>
#include <vector>

#include "constants.hpp"
#include "Params.hpp"
#include "utils.hpp"

template<typename T>
class Matrix;

template<typename T>
class BlockMatrix;

template<typename T>
std::ostream& operator<<(std::ostream &, const Matrix<T>&);

template<typename T>
std::ostream& operator<<(std::ostream &, const BlockMatrix<T>&);

// Matrix class, row-major, continous data storage, castable to/from Mat type (Mat<T> = std::vector< std::vector<T> >)
template<typename T>
class Matrix
{
	friend std::ostream& operator<< <T>(std::ostream &, const Matrix&);

public:
	using size_type = size_t;
	using value_type = T;
	using reference = T&;
	using const_reference = const T&;
	using pointer = T*;
	using const_pointer = const T*;

protected:
	size_type m_rows;
	size_type m_cols;
	size_type m_size;
	Vec<T> m_data;

public:
	Matrix() = default;
	
	Matrix(size_type numRows, size_type numCols, value_type val = value_type()) :
		m_rows(numRows), m_cols(numCols), m_size(numRows*numCols), m_data(Vec<value_type>(m_size, val))
	{}

	Matrix(size_type numRows, size_type numCols, const pointer data):
		m_rows(numRows), m_cols(numCols), m_size(numRows*numCols), m_data(Vec<value_type>(data, data + m_size))
	{}

	Matrix(const Mat<value_type>& mat):
		m_rows(mat.size()), m_cols(mat.front().size()), m_size(m_rows*m_cols)
	{
		m_data.reserve(m_size);
		for (auto row: mat)
			m_data.insert(m_data.end(), row.begin(), row.end());
	};

	pointer data() { return m_data.data(); }
	const_pointer data() const { return m_data.data(); }
	size_type rows() const { return m_rows; }
	size_type cols() const { return m_cols; }	
	size_type size() const { return m_size; }

	bool empty() const { return m_size == 0; }

	void randomize() { m_data = RandGen<value_type>::randomVec(m_size); }

	Matrix<value_type> transpose() const
	{
		Matrix<value_type> tmat(m_cols, m_rows);
		for (auto i = 0; i < m_rows; i++)
			for (auto j = 0; j < m_cols; j++)
				tmat.at(j, i) = at(i, j);
		return tmat;
	}

	reference at(size_type i, size_type j)
	{
		if (i >= m_rows || j >= m_cols)
			throw std::out_of_range("Index(es) are out of range.");
		return m_data[i*m_cols + j];
	}

	const_reference at(size_type i, size_type j) const
	{
		if (i >= m_rows || j >= m_cols)
			throw std::out_of_range("Index(es) are out of range.");
		return m_data[i*m_cols + j];
	}

	operator Mat<value_type>() const
	{
		Mat<value_type> mat;
		mat.reserve(m_rows);
		for (auto it = m_data.cbegin(); it < m_data.end(); it += m_cols)
			mat.push_back(Vec<value_type>(it, it + m_cols));
		return mat;
	}

	std::string toString(size_t precision = printDefaultPrecision) const
	{
		if (empty())
			return std::string();

		std::stringstream ss;
		ss << std::setprecision(precision) << *this;
		return ss.str();
	}
};

// Block Matrix class where each block is a row-major, continous block data in storage
template<typename T>
class BlockMatrix
{
	friend std::ostream& operator<< <T>(std::ostream &, const BlockMatrix&);
public:
	using size_type = size_t;
	using value_type = T;
	using reference = T & ;
	using const_reference = const T&;
	using pointer = T * ;
	using const_pointer = const T*;
	using size_vec = Vec<size_type>;

private:
	size_type m_rows;
	size_type m_cols;
	size_type m_size;
	Vec<value_type> m_data;

	size_vec m_rowCounts;
	size_vec m_colCounts;

	size_vec m_displs;

public:
	BlockMatrix() : m_rows(0), m_cols(0), m_size(0) {};

	BlockMatrix(const size_vec& rowCounts, const size_vec& colCounts, value_type val = value_type()) :
		m_rowCounts(rowCounts), m_colCounts(colCounts)
	{
		_init_members();
		m_data = Vec<value_type>(m_size, val);
	};

	BlockMatrix(const Params2D& params, value_type val = value_type())
	{
		const Params& rp = params.row1D(),
			cp = params.col1D();

		m_rowCounts = size_vec(rp.counts(), rp.counts() + rp.commSize());
		m_colCounts = size_vec(cp.counts(), cp.counts() + cp.commSize());
		_init_members();

		m_data = Vec<value_type>(m_size, val);
	}

	// Create a block matrix from mat (copy and realigne mat data)
	BlockMatrix(size_vec& rowCounts, size_vec& colCounts, const Mat<value_type>& mat) :
		BlockMatrix(rowCounts, colCounts)
	{
		for (auto i = 0; i < m_rows; i++)
			for (auto j = 0; j < m_cols; j++)
				at(i, j) = mat[i][j];
	}

	// Create a block matrix from mat (copy and realigne mat data)
	BlockMatrix(const Params2D& params, const Mat<value_type>& mat) :
		BlockMatrix(params)
	{
		for (auto i = 0; i < m_rows; i++)
			for (auto j = 0; j < m_cols; j++)
				at(i, j) = mat[i][j];
	}

	// Total number of rows
	size_type rows() const { return m_rows; };

	// Total number of cols
	size_type numCols() const { return m_cols; };

	// Total number of elements
	size_type size() const { return m_size; };

	bool empty() const { return m_size == 0; };

	// The number of blocks in row dimension
	size_type numBlocksRow() const { return m_rowCounts.size(); };

	// The number of blocks in col dimension  
	size_type numBlocksCol() const { return m_colCounts.size(); };

	// The number of rows of blocks in the I-th block row
	size_type numRowsBlock(size_type I) { return m_rowCounts[I]; };

	// The number of cols of blocks in the J-th block col
	size_type numColsBlock(size_type J) { return m_colCounts[J]; };

	// Offset in data array of the element (i,j) of the block (I,J)
	size_type displ(size_type I, size_type J, size_type i = 0, size_type j = 0) const
	{
		return m_displs[I*numBlocksCol() + J] + i * m_colCounts[J] + j;
	};

	// Number of elements of the block (I,J)
	size_type count(size_type I, size_type J) const { return m_rowCounts[I] * m_colCounts[J]; };

	// Create Matrix object from 
	Matrix<T> block(size_type I, size_type J) { return Matrix<T>(m_rowCounts[I], m_colCounts[J], m_data + displ(I, J)); };
	const Matrix<T> block(size_type I, size_type J) const { return Matrix<T>(m_rowCounts[I], m_colCounts[J], m_data + displ(I, J)); };

	//Pointer to the data (block layout)
	pointer data() { return m_data.data(); }

	//Pointer to the data (block layout)
	const_pointer data() const { return m_data.data(); }

	void randomize() { m_data = RandGen<value_type>::randomVec(m_size); }

	// Generate a random symmetric positive define matrix (if square maxtrix)
	void randomSPD(size_t method = 0)
	{
		if (method == 0) // create SPD matrix from Cholesky decomposition, decomposition lower triangle matrix must have strictly positive diagonal terms
		{
			Matrix<value_type> lowerTri = Matrix<value_type>(m_rows, m_cols);
			lowerTri.randomize();

			// Lets make non diagonal entries take value between -1 and 1 and diagonal entries between 3 and 4
			for (auto i = 0; i < m_rows; i++)
			{
				lowerTri.at(i, i) += 0.05*m_rows;
				for (auto j = 0; j < i; j++)
				{
					//lowerTri.at(i, j) = 2 * lowerTri.at(i, j) - 1;
					lowerTri.at(j, i) = 0;
				}
			}

			for (auto i = 0; i < m_rows; i++)
				for (auto j = 0; j <= i; j++)
				{
					at(i, j) = 0;
					for(auto k = 0; k <=std::min(i,j); k++)
						at(i, j) += lowerTri.at(i, k) * lowerTri.at(j,k);
					at(j, i) = at(i,j);
				}
		}
		else // Symmetric matrix with strictly dominant diagonal is SPD
		{
			randomize();
			for (auto i = 0; i < m_rows; i++)
			{
				for (auto j = 0; j < i; j++)
					at(j, i) = at(i, j);		// make symetric
				at(i, i) += m_rows + 1;			// make diagonal strictly dominant ( |Mii| > SUM j:0->N (|Mij|)  )
			}
		}
	}
	
	BlockMatrix<value_type> transpose() const
	{
		BlockMatrix<value_type> tmat(m_colCounts, m_rowCounts);

		for (auto i = 0; i < m_rows; i++)
			for (auto j = 0; j < m_rows; j++)
				tmat.at(j, i) = value_type(at(i, j));
		return tmat;
	}

	reference at(size_type I, size_type J, size_type i, size_type j)
	{
		return m_data[displ(I,J,i,j)];
	}

	const_reference at(size_type I, size_type J, size_type i, size_type j) const
	{
		return m_data[displ(I, J, i, j)];
	}

	reference at(size_type i, size_type j)
	{
		if (i >= m_rows || j >= m_cols)
			throw std::out_of_range("Index(es) are out of range.");
		size_type I0, J0; // hold the absolute coord of the block first element
		size_type I = _getRowBlockIdx(i, I0);
		size_type J = _getColBlockIdx(j, J0);
		return at(I, J, i - I0, j - J0);
	}

	const_reference at(size_type i, size_type j) const
	{
		if (i >= m_rows || j >= m_cols)
			throw std::out_of_range("Index(es) are out of range.");
		size_type I0, J0; // hold the absolute coord of the block first element
		size_type I = _getRowBlockIdx(i, I0);
		size_type J = _getColBlockIdx(j, J0);
		return at(I, J, i - I0, j - J0);
	}

	Mat<value_type> asMat() const
	{
		Mat<value_type> mat;
		mat.reserve(m_rows);
		for (auto i = 0; i < m_rows; i++)
		{
			Vec<value_type> vec;
			vec.reserve(m_cols);
			for (auto j = 0; j < m_cols; j++)
				vec.push_back(at(i, j));
			mat.push_back(vec);
		}
		return mat;
	}

	Matrix<value_type> asMatrix() const
	{
		Matrix<value_type> matrix(m_rows, m_cols);
		for (auto i = 0; i < m_rows; i++)
			for (auto j = 0; j < m_cols; j++)
				matrix.at(i, j) = at(i,j);
	
		return matrix;
	}

	std::string toString(size_t precision = printDefaultPrecision) const
	{
		if (empty())
			return std::string();

		std::stringstream ss;
		ss << std::setprecision(precision) << *this;
		return ss.str();
	}

private:
	// from m_rowCounts and m_colCounts, set all other data member except m_data
	void _init_members()
	{
		m_rows = 0;
		m_cols = 0;
		m_size = 0;
		size_type displcnt = 0;
		bool first = true;
		for (auto rcnt : m_rowCounts)
		{
			m_rows += rcnt;
			for (auto ccnt : m_colCounts)
			{
				if (first)
					m_cols += ccnt;
				m_displs.push_back(displcnt);
				displcnt += rcnt * ccnt;
			}
			first = false;
		}
		m_size = m_cols * m_rows;
	}

	size_type _getRowBlockIdx(size_type i, size_type& blockRowStartIdx = size_type()) const
	{
		if (i >= m_rows)
			throw std::out_of_range("Index is out of range.");
		size_type I = 0;
		size_type cnt = m_rowCounts[0];
		
		while (cnt<=i)
			cnt += m_rowCounts[++I];

		blockRowStartIdx = cnt - m_rowCounts[I];

		return I;
	}

	size_type _getColBlockIdx(size_type j, size_type& blockColStartIdx = size_type()) const
	{
		if (j >= m_cols)
			throw std::out_of_range("Index is out of range.");
		size_type J = 0;
		size_type cnt = m_colCounts[0];

		while (cnt <= j)
			cnt += m_colCounts[++J];

		blockColStartIdx = cnt - m_colCounts[J];
		return J;
	}
};

template<typename T>
std::ostream& operator<<(std::ostream &os, const Matrix<T>& matrix)
{
	if (matrix.empty())
		return os;

	const auto& data = matrix.m_data;

	auto maxStrLen = _findMaxStrLen(data.begin(), data.end(), os.precision());

	for (auto it = data.cbegin(); it < data.cend(); it += matrix.m_cols)
		_insert_in_col(os, it, it + matrix.m_cols, maxStrLen + printColSpacing) << std::endl;

	return os;
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const BlockMatrix<T>& matrix)
{
	if (matrix.empty())
		return os;

	const auto& data = matrix.m_data;

	auto maxStrLen = _findMaxStrLen(data.begin(), data.end(), os.precision());

	for (auto i = 0; i < matrix.m_rows; i++)
	{
		for (auto j = 0; j < matrix.m_cols; j++)
			os << std::setw(maxStrLen + printColSpacing) << std::left << matrix.at(i, j);
		os << std::endl;
	}
	return os;
}

template<typename T>
std::string toString(const Matrix<T>& matrix, size_t precision = printDefaultPrecision)
{
	return matrix.toString(precision);
}

template<typename T>
std::string toString(const BlockMatrix<T>& matrix, size_t precision = printDefaultPrecision)
{
	return matrix.toString(precision);
}