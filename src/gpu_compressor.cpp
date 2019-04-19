#include "gpu_compressor.hpp"
#include "file_header.hpp"

using namespace gip;

gip::GPUCompressor::GPUCompressor(): numBlocks(0), totalThreads(0), uncompressedDataBufferSize(0), inputStreams{}, outputStream(0), 
    deviceUncompressedData(nullptr), deviceCompressedData(nullptr), inputPacketBuffer(nullptr), outputPacketBuffer(nullptr)
{

    const int deviceID = gpuGetMaxGflopsDeviceId();
    /*
    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        this->inputStreams[i] = 0;
    }*/

    this->chooseDevice(deviceID);

    initConstantRange();
}

void gip::GPUCompressor::cleanResource()
{

    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        if (this->inputStreams[i] > 0)
        {
            checkCudaErrors(cudaStreamDestroy(this->inputStreams[i]));
        }
    }

    if (this->outputStream > 0)
    {
        checkCudaErrors(cudaStreamDestroy(this->outputStream));
    }

    checkCudaErrors(cudaFree(deviceCompressedData));
    checkCudaErrors(cudaFree(deviceUncompressedData));
    checkCudaErrors(cudaFreeHost(inputPacketBuffer));
    checkCudaErrors(cudaFreeHost(outputPacketBuffer));
}

gip::GPUCompressor::~GPUCompressor()
{

    this->cleanResource();
}

void gip::GPUCompressor::allocateResource()
{

    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&(this->inputStreams[i])));
    }

    checkCudaErrors(cudaStreamCreate(&(this->outputStream)));

    checkCudaErrors(cudaMalloc((void **)&deviceUncompressedData, this->uncompressedDataBufferSize));
    checkCudaErrors(cudaMalloc((void **)&deviceCompressedData, COMPRESSED_PACKET_SIZE * this->totalThreads));

    checkCudaErrors(cudaMallocHost((void **)&inputPacketBuffer, COMPRESSED_PACKET_SIZE * NUM_STREAMS));
    checkCudaErrors(cudaMallocHost((void **)&outputPacketBuffer, COMPRESSED_PACKET_SIZE));
}

void gip::GPUCompressor::chooseDevice(int deviceID)
{
    //this->~GPUCompressor();
    this->cleanResource();

    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, deviceID);
    cudaChooseDevice(&deviceID, &prop);
    //checkCudaErrors(cudaSetDevice(deviceID));
    this->numBlocks = prop.multiProcessorCount * PROCESSORS_GRIDS_FACTOR;
    this->totalThreads = NUM_THREADS * this->numBlocks;
    this->uncompressedDataBufferSize = UNCOMPRESSED_PACKET_SIZE * this->totalThreads;

    this->allocateResource();
}

CompressionInfo gip::GPUCompressor::compress(ProgressMonitor *monitor)
{
    CompressionInfo info;

    openFile = fopen(this->openFileName.c_str(), "rb");
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    size_t readSize = 0;
    size_t size = 0;
    FileHeader fileHeader;
    cudaError_t result;
    uint32_t inputStreamIndex;
    uint32_t inputBufferOffset;
    uint32_t numPackets = 0;
    uint32_t packetSize;
    uint32_t outNumPackets;
    uint32_t writeOutNumPackets;
    uint32_t blocks;
    try
    {
        monitor->reset();

        cutilCheckError(sdkResetTimer(&process_timer));
        cutilCheckError(sdkResetTimer(&io_timer));

        if (openFile != NULL && saveFile != NULL)
        {

            cutilCheckError(sdkStartTimer(&io_timer));
            if (fseek(saveFile, FileHeader::HEADER_LENGTH, SEEK_SET) != 0)
            {
                throw "Seek file failed";
            }

            info.uncompressedFileSize = this->getFileSize(openFile);
            fseek(openFile, 0, SEEK_SET);

            cutilCheckError(sdkStopTimer(&io_timer));

            info.compressedFileSize = FileHeader::HEADER_LENGTH;

            do
            {
                outNumPackets = numPackets;
                numPackets = 0;
                inputStreamIndex = 0;
                readSize = 0;
                writeOutNumPackets = 0;

                cutilCheckError(sdkStartTimer(&io_timer));
                //int x = MAX_INPUT_DATA_BLOCK;
                while ((!feof(openFile) && readSize < this->uncompressedDataBufferSize) || writeOutNumPackets < outNumPackets)
                {
                    if (writeOutNumPackets < outNumPackets)
                    {
                        cudaMemcpyAsync(outputPacketBuffer, deviceCompressedData + writeOutNumPackets * COMPRESSED_PACKET_SIZE, COMPRESSED_PACKET_SIZE, cudaMemcpyDeviceToHost, this->outputStream);
                    }

                    if (!feof(openFile))
                    {
                        cudaStreamSynchronize(this->inputStreams[inputStreamIndex]);
                        inputBufferOffset = inputStreamIndex * UNCOMPRESSED_PACKET_SIZE;

                        size = fread(inputPacketBuffer + inputBufferOffset, sizeof(unsigned char), UNCOMPRESSED_PACKET_SIZE, openFile);

                        if (size > 0)
                        {
                            cudaMemcpyAsync(deviceUncompressedData + readSize, inputPacketBuffer + inputBufferOffset, size, cudaMemcpyHostToDevice, this->inputStreams[inputStreamIndex]);
                            ++numPackets;
                            readSize += size;
                            info.processedUncompressedSize += size;
                            inputStreamIndex = (inputStreamIndex + 1) % NUM_STREAMS;
                        }
                    }

                    if (writeOutNumPackets < outNumPackets)
                    {
                        ++writeOutNumPackets;
                        cudaStreamSynchronize(this->outputStream);
                        packetSize = GPUCompressor::getPacketSize((const unsigned char *)outputPacketBuffer);

                        if (fwrite(outputPacketBuffer, packetSize, 1, saveFile) <= 0)
                        {
                            throw "Write compressed data to output file failed";
                        }
                        info.compressedFileSize += packetSize;
                    }
                    monitor->updateProgress(&info);
                }

                for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
                {
                    cudaStreamSynchronize(this->inputStreams[i]);
                }

                cutilCheckError(sdkStopTimer(&io_timer));

                if (readSize > 0)
                {

                    blocks = ceil((double)readSize / ((double)UNCOMPRESSED_PACKET_SIZE * NUM_THREADS));
                    cutilCheckError(sdkStartTimer(&process_timer));
                    garCompressExecutor((const unsigned char *)deviceUncompressedData, readSize, deviceCompressedData, blocks);

                    result = cudaThreadSynchronize();

                    if (result != cudaSuccess)
                    {
                        throw string("Fail to execute kernel code: ") + string(cudaGetErrorString(result));
                    }

                    cutilCheckError(sdkStopTimer(&process_timer));
                }
            } while (!feof(openFile) || numPackets > 0);

            cutilCheckError(sdkStartTimer(&io_timer));
            if (fseek(saveFile, 0, SEEK_SET))
            {
                throw "Seek file failed";
            }
            fileHeader.setCompressedFileSize(info.compressedFileSize);
            fileHeader.setUncompressedFileSize(info.uncompressedFileSize);
            if (fwrite(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, saveFile) <= 0)
            {
                throw "Write data to file failed";
            }
            cutilCheckError(sdkStopTimer(&io_timer));
        }
        else
        {

            throw string("Can not open input file: ") + this->openFileName;
        }
    }
    catch (exception e)
    {
        this->closeFiles();
        throw;
    }
    cutilCheckError(sdkStartTimer(&io_timer));
    this->closeFiles();
    cutilCheckError(sdkStopTimer(&io_timer));

    info.processTime = sdkGetTimerValue(&process_timer);
    info.ioTime = sdkGetTimerValue(&io_timer);

    return info;
}

CompressionInfo gip::GPUCompressor::decompress(ProgressMonitor *monitor)
{
    CompressionInfo info;

    openFile = fopen(this->openFileName.c_str(), "rb");
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    size_t readSize = 0;
    size_t size = 0;
    FileHeader fileHeader;
    cudaError_t result;
    uint32_t inputStreamIndex;
    uint32_t inputBufferOffset;
    uint32_t numPackets = 0;
    uint32_t packetSize;
    uint32_t outNumPackets;
    uint32_t writeOutNumPackets;
    uint32_t blocks;
    uint32_t maxCompressedPackets = NUM_THREADS * this->numBlocks;
    size_t r;
    //size_t writeSize =0;

    try
    {
        monitor->reset();

        cutilCheckError(sdkResetTimer(&process_timer));
        cutilCheckError(sdkResetTimer(&io_timer));

        if (openFile != NULL && saveFile != NULL)
        {

            cutilCheckError(sdkStartTimer(&io_timer));
            fseek(openFile, 0, SEEK_SET);
            if (fread(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, openFile) <= 0)
            {
                throw "Incorrect file format";
            }
            else
            {
                cutilCheckError(sdkStopTimer(&io_timer));
                readSize = FileHeader::HEADER_LENGTH;
                if (fileHeader.checkHeaderVersion())
                {
                    info = fileHeader.getInfo();

                    cutilCheckError(sdkStopTimer(&io_timer));
                    do
                    {
                        outNumPackets = numPackets;
                        numPackets = 0;
                        inputStreamIndex = 0;
                        //readSize=0;
                        writeOutNumPackets = 0;

                        cutilCheckError(sdkStartTimer(&io_timer));
                        while ((readSize < info.compressedFileSize && numPackets < maxCompressedPackets) || writeOutNumPackets < outNumPackets)
                        {
                            if (writeOutNumPackets < outNumPackets)
                            {
                                cudaMemcpyAsync(outputPacketBuffer, deviceUncompressedData + writeOutNumPackets * UNCOMPRESSED_PACKET_SIZE, UNCOMPRESSED_PACKET_SIZE, cudaMemcpyDeviceToHost, this->outputStream);
                            }

                            if (readSize < info.compressedFileSize)
                            {
                                cudaStreamSynchronize(this->inputStreams[inputStreamIndex]);

                                inputBufferOffset = inputStreamIndex * COMPRESSED_PACKET_SIZE;
                                size = fread(inputPacketBuffer + inputBufferOffset, PACKET_HEADER_LENGTH, 1, openFile);

                                if (size > 0)
                                {
                                    size = getCompressedSize(inputPacketBuffer + inputBufferOffset); //read(compressed,2);
                                    r = fread(inputPacketBuffer + inputBufferOffset + PACKET_HEADER_LENGTH, sizeof(uint8_t), size - PACKET_HEADER_LENGTH, openFile);

                                    if(r!=size - PACKET_HEADER_LENGTH){
                                        throw "Invalid file length";
                                    }

                                    cudaMemcpyAsync(deviceCompressedData + (numPackets * COMPRESSED_PACKET_SIZE), inputPacketBuffer + inputBufferOffset, size, cudaMemcpyHostToDevice, this->inputStreams[inputStreamIndex]);
                                    ++numPackets;
                                    readSize += size;

                                    inputStreamIndex = (inputStreamIndex + 1) % NUM_STREAMS;
                                }
                                else
                                {
                                    throw "Incorrect file format";
                                }
                            }

                            if (writeOutNumPackets < outNumPackets)
                            {
                                ++writeOutNumPackets;
                                cudaStreamSynchronize(this->outputStream);
                                packetSize = UNCOMPRESSED_PACKET_SIZE; //getUncompressedSize((const unsigned char*)outputPacketBuffer);

                                if (info.uncompressedFileSize - info.processedUncompressedSize < UNCOMPRESSED_PACKET_SIZE)
                                {
                                    packetSize = info.uncompressedFileSize - info.processedUncompressedSize;
                                }
                                if (packetSize > 0)
                                {
                                    if (fwrite(outputPacketBuffer, packetSize, 1, saveFile) <= 0)
                                    {
                                        throw "Write uncompressed data to output file failed";
                                    }
                                }
                                //writeSize +=packetSize;
                                info.processedUncompressedSize += packetSize;
                            }
                            monitor->updateProgress(&info);
                        }

                        for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
                        {
                            cudaStreamSynchronize(this->inputStreams[i]);
                        }

                        cutilCheckError(sdkStopTimer(&io_timer));

                        if (numPackets > 0)
                        {

                            blocks = ceil((double)numPackets / ((double)NUM_THREADS));
                            cutilCheckError(sdkStartTimer(&process_timer));
                            garDecompressExecutor((const unsigned char *)deviceCompressedData, numPackets * COMPRESSED_PACKET_SIZE, deviceUncompressedData, blocks);

                            result = cudaThreadSynchronize();

                            if (result != cudaSuccess)
                            {
                                throw string("Fail to execute kernel code: ") + string(cudaGetErrorString(result));
                            }

                            cutilCheckError(sdkStopTimer(&process_timer));
                        }
                    } while (readSize < info.compressedFileSize || numPackets > 0);
                }
                else
                {
                    throw "Incorrect file format";
                }
            }
        }
        else
        {

            throw string("Can not open input file: ") + this->openFileName;
        }
    }
    catch (exception e)
    {
        this->closeFiles();
        throw;
    }
    cutilCheckError(sdkStartTimer(&io_timer));
    this->closeFiles();
    cutilCheckError(sdkStopTimer(&io_timer));

    info.processTime = sdkGetTimerValue(&process_timer);
    info.ioTime = sdkGetTimerValue(&io_timer);

    return info;
}