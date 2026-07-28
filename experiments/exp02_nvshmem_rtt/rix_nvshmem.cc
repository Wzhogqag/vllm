// SPDX-License-Identifier: Apache-2.0
// Minimal C wrapper around NVSHMEM host init + put + barrier.
//
// Why this exists: nvshmemx_init_attr() is a static inline that stamps struct
// versions and calls an internal init_thread -- not cleanly callable from
// ctypes. We let the C++ compiler expand those inlines/macros correctly, and
// expose flat C symbols that ctypes CAN call.
//
// Host-only headers (nvshmem_host.h) avoid pulling in device cccl deps.
//
// Build: see build.sh
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "nvshmem_host.h"
#include "nvshmemx_api.h"

extern "C" {

// UID is a fixed 128-byte blob. rank0 fills it; caller broadcasts it out-of-band.
int rix_get_uniqueid(void* out_uid_128) {
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    int rc = nvshmemx_get_uniqueid(&uid);
    if (rc == 0) std::memcpy(out_uid_128, &uid, sizeof(uid));
    return rc;
}

// Init with a UID blob every rank already agrees on.
// NOTE: we call the EXPORTED nvshmemx_hostlib_init_attr (in libnvshmem_host.so),
// NOT the static-inline nvshmemx_init_attr -- the inline pulls in
// nvshmemi_init_thread which lives only in the device static lib. We stamp the
// struct versions explicitly via the same macro the inline would use.
int rix_init_with_uid(const void* uid_128, int myrank, int nranks) {
    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_128, sizeof(uid));
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int rc = nvshmemx_set_attr_uniqueid_args(myrank, nranks, &uid, &attr);
    if (rc != 0) return rc;
    nvshmemx_init_init_attr_ver_only(attr);  // stamp version fields
    return nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

int  rix_my_pe()  { return nvshmem_my_pe(); }
int  rix_n_pes()  { return nvshmem_n_pes(); }
void* rix_malloc(size_t n) { return nvshmem_malloc(n); }
void rix_free(void* p)     { nvshmem_free(p); }
void rix_barrier_all()     { nvshmem_barrier_all(); }
void rix_quiet()           { nvshmem_quiet(); }
void rix_finalize()        { nvshmemx_hostlib_finalize(); }

// One put on a stream (stream passed as raw cudaStream_t handle from Python).
void rix_putmem_on_stream(void* dest, const void* src, size_t bytes, int pe,
                          void* stream) {
    nvshmemx_putmem_on_stream(dest, src, bytes, pe, (cudaStream_t)stream);
}

}  // extern "C"
