#pragma once

#include "compressor.hpp"
#include "progress_monitor.hpp"

namespace gip
{
    class GPUCompressor : public Compressor
    {
    private:
        constexpr static int NUM_STREAMS = 4;

        uint32_t numBlocks;
        uint32_t totalThreads;
        size_t uncompressedDataBufferSize;
        cudaStream_t inputStreams[NUM_STREAMS]; //(cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
        cudaStream_t outputStream;              //(cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));

        uint8_t *deviceUncompressedData;
        uint8_t *deviceCompressedData;
        char *inputPacketBuffer;
        uint8_t *outputPacketBuffer;

        void allocateResource();
        void cleanResource();

    public:
        CompressionInfo compress(ProgressMonitor *monitor);
        CompressionInfo decompress(ProgressMonitor *monitor);
        GPUCompressor();
        ~GPUCompressor();

        void chooseDevice(const int id);

        static unsigned short getPacketSize(const uint8_t *packet)
        {
            return ((unsigned short *)(((char *)packet)))[0];
        }
    };
}