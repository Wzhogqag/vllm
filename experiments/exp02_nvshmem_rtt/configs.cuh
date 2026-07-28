// SPDX-License-Identifier: Apache-2.0
// Slim replacement for DeepEP's configs.cuh — provides ONLY what
// deepep_ibgda_device.cuh / deepep_utils.cuh need, sourced from official
// NVSHMEM headers. DeepEP's real configs.cuh ships an fp8 compat shim that
// conflicts with CUDA 13's cuda_fp8.h; we don't need any fp8 here, so we drop
// it and pull the NVSHMEM device/IBGDA headers directly.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Timeout used by receiver spin-poll loops (from DeepEP configs.cuh).
#define NUM_TIMEOUT_CYCLES 20000000000ull  // ~10s at ~2GHz
#define FINISHED_SUM_TAG 1024  // used by deepep_utils.cuh barrier helper

// Official NVSHMEM headers: these define nvshmemi_ibgda_device_state_d,
// nvshmemi_device_state_d, the IBGDA types, and NVSHMEMI_IBGDA_* constants.
#include <infiniband/mlx5dv.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
