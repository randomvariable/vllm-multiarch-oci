// SPDX-License-Identifier: Apache-2.0
// The pinned NCCL tree enables its GPUNetIO transport unconditionally on Linux.
// This build omits libibverbs headers, so omit that optional
// transport and the GIN GDAKI backend that references its symbols.
#include "gin_host_gdaki.h"

ncclResult_t ncclGinGdakiCreateContext(void*, int, int, int, int, int, void**, ncclNetDeviceHandle_t**) { return ncclInternalError; }
ncclResult_t ncclGinGdakiDestroyContext(void*) { return ncclInternalError; }
ncclResult_t ncclGinGdakiRegMrSym(void*, void*, size_t, int, uint64_t, void**, void**) { return ncclInternalError; }
ncclResult_t ncclGinGdakiDeregMrSym(void*, void*) { return ncclInternalError; }
ncclResult_t ncclGinGdakiProgress(void*) { return ncclInternalError; }
ncclResult_t ncclGinGdakiQueryLastError(void*, bool*) { return ncclInternalError; }
