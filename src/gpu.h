#pragma once

#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <iostream>

#define EXTRA_COMPRESSED_SIZE (512)
#define NUM_THREADS 32
#define PROCESSORS_GRIDS_FACTOR 8

#define COMPRESSED_PACKET_SIZE (8192 + EXTRA_COMPRESSED_SIZE)
#define UNCOMPRESSED_PACKET_SIZE (COMPRESSED_PACKET_SIZE - EXTRA_COMPRESSED_SIZE)
#define PACKET_HEADER_LENGTH 4
#define cutilCheckError

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#ifdef __cplusplus
}
#endif /* __cplusplus */