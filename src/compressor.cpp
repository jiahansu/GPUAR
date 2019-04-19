#include "compressor.hpp"

using namespace gip;

gip::Compressor::Compressor() : openFileName(""), saveFileName(""), process_timer(nullptr), io_timer(nullptr), 
    openFile(nullptr), saveFile(nullptr), compressed(nullptr), data(nullptr), range(nullptr)
{
    if (UNCOMPRESSED_PACKET_SIZE % sizeof(READ_ELEMENT) > 0)
    {
        throw string("The input packet size must be the multiple of READ_ELEMENT!");
    }

    if (UNCOMPRESSED_PACKET_SIZE >= MAX_PROBABILITY - UPPER(EOF_CHAR))
    {
        throw string("The packet's size was too large that to occur overflow problem");
    }

    sdkCreateTimer(&this->process_timer);
    sdkCreateTimer(&this->io_timer);

    //	cudaMallocHost((void**)&input,INPUT_BUFFER_SIZE);
    //	cudaMallocHost((void**)&hashtable,sizeof(short)*HASH_VALUES);
    cudaMallocHost((void **)&data, UNCOMPRESSED_PACKET_SIZE);
    cudaMallocHost((void **)&compressed, COMPRESSED_PACKET_SIZE);
    cudaMallocHost((void **)&this->range, sizeof(AdaptiveProbabilityRange));
}

void gip::Compressor::generateRandomFile(const size_t size)
{
    int d;

    saveFile = fopen(this->saveFileName.c_str(), "wb");

    for (int i = 0; i < size; i += 4)
    {
        d = rand();
        if (fwrite(&d, sizeof(int), 1, saveFile) <= 0)
        {
            throw "Write raw data to file failed";
        }
    }

    this->closeFiles();
}

gip::Compressor::~Compressor()
{
    sdkDeleteTimer(&this->process_timer);
    sdkDeleteTimer(&this->io_timer);

    //checkCudaErrors(cudaFreeHost(input));
    //checkCudaErrors(cudaFreeHost(hashtable));
    checkCudaErrors(cudaFreeHost(data));
    checkCudaErrors(cudaFreeHost(compressed));
    checkCudaErrors(cudaFreeHost(this->range));
}