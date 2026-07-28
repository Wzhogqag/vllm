# SPDX-License-Identifier: Apache-2.0
"""ctypes binding to OUR minimal wrapper librix_nvshmem.so (not NVSHMEM directly).

NVSHMEM's init path uses static-inline functions + version-stamping macros that
ctypes cannot call cleanly. So rix_nvshmem.cc wraps them behind flat C symbols
(rix_*), compiled by build.sh. This module just binds those flat symbols.

Symbols (all confirmed exported via nm -D librix_nvshmem.so):
  rix_get_uniqueid(void* out128) -> int
  rix_init_with_uid(const void* uid128, int myrank, int nranks) -> int
  rix_my_pe/rix_n_pes -> int ; rix_malloc(size)->ptr ; rix_free(ptr)
  rix_putmem_on_stream(dest, src, bytes, pe, stream) ; rix_barrier_all ; rix_quiet
"""
import ctypes
import os

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librix_nvshmem.so")
UID_BYTES = 128


class Nvshmem:
    def __init__(self, lib_path: str = _LIB):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} not built -- run ./build.sh first")
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._bind()

    def _bind(self):
        l = self.lib
        l.rix_get_uniqueid.argtypes = [ctypes.c_void_p]
        l.rix_get_uniqueid.restype = ctypes.c_int
        l.rix_init_with_uid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        l.rix_init_with_uid.restype = ctypes.c_int
        l.rix_my_pe.restype = ctypes.c_int
        l.rix_n_pes.restype = ctypes.c_int
        l.rix_malloc.argtypes = [ctypes.c_size_t]
        l.rix_malloc.restype = ctypes.c_void_p
        l.rix_free.argtypes = [ctypes.c_void_p]
        l.rix_putmem_on_stream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        l.rix_barrier_all.argtypes = []
        l.rix_quiet.argtypes = []
        l.rix_finalize.argtypes = []

    def get_uniqueid_bytes(self) -> bytes:
        buf = (ctypes.c_byte * UID_BYTES)()
        rc = self.lib.rix_get_uniqueid(ctypes.byref(buf))
        if rc != 0:
            raise RuntimeError(f"rix_get_uniqueid failed rc={rc}")
        return bytes(buf)

    def init_with_uid(self, uid_bytes: bytes, myrank: int, nranks: int):
        assert len(uid_bytes) == UID_BYTES, len(uid_bytes)
        buf = (ctypes.c_byte * UID_BYTES).from_buffer_copy(uid_bytes)
        rc = self.lib.rix_init_with_uid(ctypes.byref(buf), myrank, nranks)
        if rc != 0:
            raise RuntimeError(f"rix_init_with_uid failed rc={rc}")

    def my_pe(self) -> int:
        return self.lib.rix_my_pe()

    def n_pes(self) -> int:
        return self.lib.rix_n_pes()

    def malloc(self, nbytes: int) -> int:
        p = self.lib.rix_malloc(nbytes)
        if not p:
            raise RuntimeError(f"rix_malloc({nbytes}) returned NULL")
        return p

    def free(self, ptr: int):
        self.lib.rix_free(ctypes.c_void_p(ptr))

    def putmem_on_stream(self, dest_ptr: int, src_ptr: int, nbytes: int, pe: int,
                         stream_ptr: int):
        self.lib.rix_putmem_on_stream(
            ctypes.c_void_p(dest_ptr), ctypes.c_void_p(src_ptr),
            nbytes, pe, ctypes.c_void_p(stream_ptr),
        )

    def barrier_all(self):
        self.lib.rix_barrier_all()

    def quiet(self):
        self.lib.rix_quiet()
