/* Optional C++11 MPI wrappers for scope-profiler.
 *
 * Include this header instead of calling the corresponding MPI routines when
 * communication timing and metadata are wanted. The wrappers preserve MPI's
 * return codes and write ordinary scope-profiler regions, so no trace-format
 * extension or post-processing plugin is required.
 */
#ifndef SCOPE_PROFILER_MPI_HPP
#define SCOPE_PROFILER_MPI_HPP

#include "scope_profiler.hpp"

#include <mpi.h>

#include <cstring>
#include <sstream>
#include <string>
#include <utility>

namespace sp {
namespace mpi {

namespace detail {

inline long long datatype_bytes(int count, MPI_Datatype datatype) noexcept
{
#ifdef SP_DISABLE_PROFILING
    (void)count;
    (void)datatype;
    return -1;
#else
    int size = 0;
    if (count < 0 || MPI_Type_size(datatype, &size) != MPI_SUCCESS) {
        return -1;
    }
    return static_cast<long long>(count) * static_cast<long long>(size);
#endif
}

inline long long communicator_id(MPI_Comm communicator) noexcept
{
    return static_cast<long long>(MPI_Comm_c2f(communicator));
}

inline const char *kind_for(const char *operation) noexcept
{
    if (std::strcmp(operation, "wait") == 0) {
        return "wait";
    }
    if (std::strcmp(operation, "barrier") == 0 ||
        std::strcmp(operation, "bcast") == 0 ||
        std::strcmp(operation, "reduce") == 0 ||
        std::strcmp(operation, "allreduce") == 0) {
        return "collective";
    }
    return "point-to-point";
}

inline std::string region_name(
    const char *operation,
    long long bytes,
    int peer,
    int root,
    int tag,
    MPI_Comm communicator,
    const char *request_operation = 0)
{
    std::ostringstream name;
    name << "mpi:" << operation
         << " kind=" << kind_for(operation)
         << " bytes=" << bytes
         << " peer=" << peer
         << " root=" << root
         << " tag=" << tag
         << " comm=" << communicator_id(communicator);
    if (request_operation != 0) {
        name << " request=" << request_operation;
    }
    return name.str();
}

inline int region(const std::string &name)
{
    return sp_region_at(name.c_str(), __FILE__, __LINE__);
}

} // namespace detail

#ifdef SP_DISABLE_PROFILING
#  define SP_MPI_PROFILE_SCOPE(operation, bytes, peer, root, tag, communicator) \
    ((void)0)
#else
#  define SP_MPI_PROFILE_SCOPE(operation, bytes, peer, root, tag, communicator) \
    std::string sp_mpi_region_name = ::sp::mpi::detail::region_name( \
        (operation), (bytes), (peer), (root), (tag), (communicator)); \
    ::sp::Scope sp_mpi_scope(::sp::mpi::detail::region(sp_mpi_region_name))
#endif

class Request {
public:
    Request() noexcept
        : request_(MPI_REQUEST_NULL), bytes_(-1), peer_(-1), tag_(-1),
          communicator_(MPI_COMM_WORLD), operation_("unknown"), error_(MPI_SUCCESS)
    {
    }

    Request(
        MPI_Request request,
        long long bytes,
        int peer,
        int tag,
        MPI_Comm communicator,
        const char *operation,
        int error = MPI_SUCCESS) noexcept
        : request_(request), bytes_(bytes), peer_(peer), tag_(tag),
          communicator_(communicator), operation_(operation), error_(error)
    {
    }

    Request(const Request &) = delete;
    Request &operator=(const Request &) = delete;

    Request(Request &&other) noexcept
        : request_(other.request_), bytes_(other.bytes_), peer_(other.peer_),
          tag_(other.tag_), communicator_(other.communicator_),
          operation_(other.operation_), error_(other.error_)
    {
        other.request_ = MPI_REQUEST_NULL;
    }

    Request &operator=(Request &&) = delete;

    MPI_Request &native() noexcept { return request_; }
    const MPI_Request &native() const noexcept { return request_; }
    bool active() const noexcept { return request_ != MPI_REQUEST_NULL; }
    int error() const noexcept { return error_; }

private:
    MPI_Request request_;
    long long bytes_;
    int peer_;
    int tag_;
    MPI_Comm communicator_;
    const char *operation_;
    int error_;

    friend int wait(Request &, MPI_Status *);
};

inline int send(
    const void *buffer,
    int count,
    MPI_Datatype datatype,
    int destination,
    int tag,
    MPI_Comm communicator)
{
    SP_MPI_PROFILE_SCOPE(
        "send", detail::datatype_bytes(count, datatype), destination, -1, tag,
        communicator);
    return MPI_Send(buffer, count, datatype, destination, tag, communicator);
}

inline int recv(
    void *buffer,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm communicator,
    MPI_Status *status = MPI_STATUS_IGNORE)
{
    SP_MPI_PROFILE_SCOPE(
        "recv", detail::datatype_bytes(count, datatype), source, -1, tag,
        communicator);
    return MPI_Recv(buffer, count, datatype, source, tag, communicator, status);
}

inline Request isend(
    const void *buffer,
    int count,
    MPI_Datatype datatype,
    int destination,
    int tag,
    MPI_Comm communicator)
{
    const long long bytes = detail::datatype_bytes(count, datatype);
    SP_MPI_PROFILE_SCOPE(
        "isend", bytes, destination, -1, tag, communicator);
    MPI_Request request = MPI_REQUEST_NULL;
    int error = MPI_Isend(
        buffer, count, datatype, destination, tag, communicator, &request);
    return Request(
        request, bytes, destination, tag, communicator, "isend", error);
}

inline Request irecv(
    void *buffer,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm communicator)
{
    const long long bytes = detail::datatype_bytes(count, datatype);
    SP_MPI_PROFILE_SCOPE("irecv", bytes, source, -1, tag, communicator);
    MPI_Request request = MPI_REQUEST_NULL;
    int error = MPI_Irecv(
        buffer, count, datatype, source, tag, communicator, &request);
    return Request(request, bytes, source, tag, communicator, "irecv", error);
}

inline int wait(Request &request, MPI_Status *status = MPI_STATUS_IGNORE)
{
#ifndef SP_DISABLE_PROFILING
    std::string name = detail::region_name(
        "wait", request.bytes_, request.peer_, -1, request.tag_,
        request.communicator_, request.operation_);
    Scope scope(detail::region(name));
#endif
    request.error_ = MPI_Wait(&request.request_, status);
    return request.error_;
}

inline int barrier(MPI_Comm communicator)
{
    SP_MPI_PROFILE_SCOPE("barrier", 0, -1, -1, -1, communicator);
    return MPI_Barrier(communicator);
}

inline int bcast(
    void *buffer,
    int count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm communicator)
{
    SP_MPI_PROFILE_SCOPE(
        "bcast", detail::datatype_bytes(count, datatype), -1, root, -1,
        communicator);
    return MPI_Bcast(buffer, count, datatype, root, communicator);
}

inline int reduce(
    const void *send_buffer,
    void *receive_buffer,
    int count,
    MPI_Datatype datatype,
    MPI_Op operation,
    int root,
    MPI_Comm communicator)
{
    SP_MPI_PROFILE_SCOPE(
        "reduce", detail::datatype_bytes(count, datatype), -1, root, -1,
        communicator);
    return MPI_Reduce(
        send_buffer, receive_buffer, count, datatype, operation, root,
        communicator);
}

inline int allreduce(
    const void *send_buffer,
    void *receive_buffer,
    int count,
    MPI_Datatype datatype,
    MPI_Op operation,
    MPI_Comm communicator)
{
    SP_MPI_PROFILE_SCOPE(
        "allreduce", detail::datatype_bytes(count, datatype), -1, -1, -1,
        communicator);
    return MPI_Allreduce(
        send_buffer, receive_buffer, count, datatype, operation, communicator);
}

} // namespace mpi
} // namespace sp

#undef SP_MPI_PROFILE_SCOPE

#endif /* SCOPE_PROFILER_MPI_HPP */
