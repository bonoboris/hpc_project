// Wrapper for MPI messaging function to accept more complex types

#pragma once

#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#include <type_traits>

#include "Params.hpp"

// p is only used to get the communicator and sending rank
// used tag is src rank 
template<typename SendCont, typename RecvCont>
int SendRcv(const Params& p, const SendCont& sendCont, RecvCont& recvCont, int dstRank)
{
	return MPI_Sendrecv(sendCont.data(), sendCont.size(), FROMTYPE(typename SendCont::value_type), dstRank, p.rank(),
		recvCont.data(), recvCont.size(), FROMTYPE(typename RecvCont::value_type), dstRank, dstRank, p.comm(), MPI_STATUSES_IGNORE);
}

template<typename SendCont, typename RecvCont>
int Scatterv(const Params& p, const SendCont& sendCont, RecvCont& recvCont)
{
	return MPI_Scatterv(sendCont.data(), p.counts(), p.displs(), FROMTYPE(typename SendCont::value_type), recvCont.data(), p.len(), FROMTYPE(typename RecvCont::value_type), p.master(), p.comm());
}

template<typename SendCont, typename Scalar>
int Scatter(const Params& p, const SendCont& sendCont, Scalar& val, int sendRank = -1)
{
	if (sendRank == -1)
		sendRank = p.master();
	return MPI_Scatter(sendCont.data(), 1, FROMTYPE(typename SendCont::value_type), &val, 1, FROMTYPE(Scalar), sendRank, p.comm());
}

template<typename SendCont, typename RecvCont>
int Gatherv(const Params& p, const SendCont& sendCont, RecvCont& recvCont)
{
	return MPI_Gatherv(sendCont.data(), p.len(), FROMTYPE(typename SendCont::value_type), recvCont.data(), p.counts(), p.displs(), FROMTYPE(typename RecvCont::value_type), p.master(), p.comm());
}

template<typename Scalar, typename RecvCont>
int Gather(const Params& p, const Scalar& val, RecvCont& recvCont, int recvRank = -1)
{
	if (recvRank == -1)
		recvRank = p.master();
	return MPI_Gather(&val, 1, FROMTYPE(Scalar), recvCont.data(), 1, FROMTYPE(typename RecvCont::value_type), recvRank, p.comm());
}

template<typename SendCont, typename RecvCont>
int AllGatherv(const Params& p, const SendCont& sendCont, RecvCont& recvCont)
{
	return MPI_Allgatherv(sendCont.data(), p.len(), FROMTYPE(typename SendCont::value_type), recvCont.data(), p.counts(), p.displs(), FROMTYPE(typename RecvCont::value_type), p.comm());
}

// Broadcast single scalar variable
template<typename Scal>
IS_SCALAR(Scal, int) Bcast(const Params& p, Scal& val, int sendRank = -1)
{
	if (sendRank == -1)
		sendRank = p.master();
	return MPI_Bcast(&val, 1, FROMTYPE(Scal), sendRank, p.comm());
}

// Broadcast a container (size of data to send is determine by size method of the Container sendCont)
// if sendRank is not spcified the sending process is p.master())
template<typename Container>
IS_NOT_SCALAR(Container, int) Bcast(const Params& p, Container& sendCont, int sendRank = -1)
{
	if (sendRank == -1)
		sendRank = p.master();
	return MPI_Bcast(sendCont.data(), sendCont.size(), FROMTYPE(typename Container::value_type), sendRank, p.comm());
}


// Reduce a container (size of data to reduce is determines by size() method of the Container sendCont)
// if sendRank is not spcified the sending process is p.master())
template<typename Container>
IS_NOT_SCALAR(Container, int) Reduce(const Params& p, Container& sendCont, MPI_Op operation, Container& resCont, int recvRank = -1)
{
	if (recvRank == -1)
		recvRank = p.master();
	return MPI_Reduce(sendCont.data(), resCont.data(), sendCont.size(), FROMTYPE(typename Container::value_type), operation, recvRank, p.comm());
}

template<typename Scalar>
IS_SCALAR(Scalar, int) Reduce(const Params& p, Scalar& sendVal, MPI_Op operation, Scalar& resVal)
{
	return MPI_Reduce(&sendVal, &resVal, 1, FROMTYPE(Scalar), operation, p.master(), p.comm());
}

// AllReduce a container (size of data to reduce is determined by size() method of the Container sendCont)
// if sendRank is not spcified the sending process is p.master())
template<typename Container>
IS_NOT_SCALAR(Container, int) AllReduce(const Params& p, Container& sendCont, MPI_Op operation, Container& resCont)
{
	return MPI_Allreduce(sendCont.data(), resCont.data(), sendCont.size(), FROMTYPE(typename Container::value_type), operation, p.comm());
}


template<typename Scalar>
IS_SCALAR(Scalar, int) AllReduce(const Params& p, Scalar& sendVal, MPI_Op operation, Scalar& resVal)
{
	return MPI_Allreduce(&sendVal, &resVal, 1, FROMTYPE(Scalar), operation, p.comm());
}