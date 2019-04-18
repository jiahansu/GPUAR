
/*
* NOTICE TO USER:
*
* This source code is subject to Jia-Han Su.
*/

/*
   GPUAR source code
   Version 0.1 - first release
*/

#include "container.h"

using namespace gip;

void gip::ProgressMonitor::updateProgress(const CompressionInfo *info)
{
    const unsigned short lastPercent = this->currentRatio * 100;
    unsigned short currentPercent;

    this->currentRatio = (double)info->processedUncompressedSize / info->uncompressedFileSize;
    currentPercent = this->currentRatio * 100;

    if ((lastPercent / 10) != (currentPercent / 10))
    {
        std::cout << currentPercent << "%.." << std::flush;
        if (currentPercent >= 100)
        {
            std::cout << "Closing file..";
        }
    }
}

gip::Compressor::Compressor() : openFileName(""), saveFileName(""), process_timer(nullptr), io_timer(nullptr)
{
    if (UNCOMPRESSED_PACKET_SIZE % sizeof(READ_ELEMENT) > 0)
    {
        throw string("The input packet size must be the multiple of READ_ELEMENT!");
    }

    if (UNCOMPRESSED_PACKET_SIZE >= MAX_PROBABILITY - UPPER(EOF_CHAR))
    {
        throw string("The packet's size was too large that to occur overflow problem");
    }

    this->openFile = NULL;
    this->saveFile = NULL;
    sdkCreateTimer(&this->process_timer);
    sdkCreateTimer(&this->io_timer);

    //	cudaMallocHost((void**)&input,INPUT_BUFFER_SIZE);
    //	cudaMallocHost((void**)&hashtable,sizeof(short)*HASH_VALUES);
    cudaMallocHost((void **)&data, UNCOMPRESSED_PACKET_SIZE);
    cudaMallocHost((void **)&compressed, COMPRESSED_PACKET_SIZE);
    cudaMallocHost((void **)&this->range, sizeof(AdaptiveProbabilityRange));
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

gip::CPUCompressor::CPUCompressor()
{
}

CompressionInfo gip::CPUCompressor::decompress(ProgressMonitor *monitor)
{
    CompressionInfo info;

    openFile = fopen(this->openFileName.c_str(), "rb");
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    int size = 0;
    FileHeader fileHeader;
    size_t r;
    unsigned int packetSize;
    probability_t cumProb;
    try
    {
        monitor->reset();
        if (openFile != NULL && saveFile != NULL)
        {

            cutilCheckError(sdkResetTimer(&process_timer));
            cutilCheckError(sdkResetTimer(&io_timer));

            cutilCheckError(sdkStartTimer(&io_timer));
            //read file header
            if (fread(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, openFile) <= 0)
            {
                ;
                throw "Incorrect file format";
            }
            else
            {
                //memcpy(fileHeader.getData(),data,FileHeader::HEADER_LENGTH);

                cutilCheckError(sdkStopTimer(&io_timer));

                if (fileHeader.checkHeaderVersion())
                {
                    info = fileHeader.getInfo();

                    do
                    {
                        cutilCheckError(sdkStartTimer(&io_timer));
                        size = fread(compressed, PACKET_HEADER_LENGTH, 1, openFile);
                        cutilCheckError(sdkStopTimer(&io_timer));
                        if (size > 0)
                        {
                            packetSize = getCompressedSize(compressed); //read(compressed,2);

                            if (fread(compressed + PACKET_HEADER_LENGTH, packetSize - PACKET_HEADER_LENGTH, 1, openFile) > 0)
                            {
                                cutilCheckError(sdkStartTimer(&process_timer));
                                initializeAdaptiveProbabilityRangeList(this->range, cumProb);
                                r = arDecompress(compressed, packetSize, (unsigned char *)data, *range, cumProb);
                                info.processedUncompressedSize += r;
                                cutilCheckError(sdkStopTimer(&process_timer));

                                cutilCheckError(sdkStartTimer(&io_timer));
                                if (fwrite(data, r, 1, saveFile) <= 0)
                                {
                                    throw "Write raw data to file failed";
                                }
                                cutilCheckError(sdkStopTimer(&io_timer));

                                monitor->updateProgress(&info);
                            }
                            else
                            {
                                throw "Incorrect file format";
                            }
                        }
                    } while (!feof(openFile));
                }
                else
                {
                    throw "Incorrect file format";
                }
            }
        }
        else
        {

            throw "Open file failed";
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
//checkCudaErrors(cudaFreeHost(packetHeader));
gip::CPUCompressor::~CPUCompressor()
{
}

CompressionInfo gip::CPUCompressor::compress(ProgressMonitor *monitor)
{
    CompressionInfo info;
    openFile = fopen(this->openFileName.c_str(), "rb");
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    int size = 0;
    FileHeader fileHeader;
    size_t r;
    probability_t cumulativeProb;

    try
    {
        monitor->reset();

        cutilCheckError(sdkResetTimer(&process_timer));
        cutilCheckError(sdkResetTimer(&io_timer));

        cutilCheckError(sdkStartTimer(&io_timer));
        if (fseek(saveFile, FileHeader::HEADER_LENGTH, SEEK_SET) != 0)
        {
            throw "Seek file failed";
        }
        cutilCheckError(sdkStopTimer(&io_timer));

        info.compressedFileSize = FileHeader::HEADER_LENGTH;

        if (openFile != NULL && saveFile != NULL)
        {
            cutilCheckError(sdkStartTimer(&io_timer));
            info.uncompressedFileSize = this->getFileSize(openFile);
            fseek(openFile, 0, SEEK_SET);
            cutilCheckError(sdkStopTimer(&io_timer));
            do
            {

                cutilCheckError(sdkStartTimer(&io_timer));
                size = fread(data, sizeof(char), UNCOMPRESSED_PACKET_SIZE, openFile);
                cutilCheckError(sdkStopTimer(&io_timer));

                info.processedUncompressedSize += size;

                if (size > 0)
                {
                    //info.uncompressedFileSize+=size;

                    cutilCheckError(sdkStartTimer(&process_timer));

                    initializeAdaptiveProbabilityRangeList(this->range, cumulativeProb);
                    r = arCompress((const unsigned char *)data, size, (unsigned char *)compressed, *range, cumulativeProb);
                    cutilCheckError(sdkStopTimer(&process_timer));
                    info.compressedFileSize += r;

                    cutilCheckError(sdkStartTimer(&io_timer));
                    if (fwrite(compressed, r, 1, saveFile) <= 0)
                    {
                        throw "Write data to file failed";
                    }
                    cutilCheckError(sdkStopTimer(&io_timer));

                    monitor->updateProgress(&info);
                }
            } while (!feof(openFile));

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

            throw "Open files failed";
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
gip::GPUCompressor::GPUCompressor()
{

    const int deviceID = gpuGetMaxGflopsDeviceId();

    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        this->inputStreams[i] = 0;
    }
    this->outputStream = 0;

    deviceUncompressedData = NULL;
    deviceCompressedData = NULL;
    inputPacketBuffer = NULL;
    outputPacketBuffer = NULL;

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
                                    fread(inputPacketBuffer + inputBufferOffset + PACKET_HEADER_LENGTH, sizeof(unsigned char), size - PACKET_HEADER_LENGTH, openFile);

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

//#define _DEBUG

void test(int argc, char **argv)
{
    char original[] = "LZ compression is based on finding repeated strings: Five, six, seven, eight, nine, fifteen, sixteen, seventeen, fifteen, sixteen, seventeen.LZ compression is based on finding repeated strings: Five, six, seven, eight, nine, fifteen, sixteen, seventeen, fifteen, sixteen, seventeen.LZ compression is based on finding repeated strings: Five, six, seven, eight, nine, fifteen, sixteen, seventeen, fifteen, sixteen, seventeen.LZ compression is based on finding repeated strings: Five, six, seven, eight, nine, fifteen, sixteen, seventeen, fifteen, sixteen, seventeen.LZ compression is based on finding repeated strings: Five, six, seven, eight, nine, fifteen, sixteen, seventeen, fifteen, sixteen, seventeen.";
    unsigned char compressed[705 + 512];
    AdaptiveProbabilityRange range[1];
    size_t r;
    probability_t cumProb;

    //unsigned char* input=NULL;
    //short* hashtable=NULL;

    StopWatchInterface *timer = nullptr;

    sdkCreateTimer(&timer);

    for (int i = 0; i < 1; ++i)
    {

        memset(compressed, 0, 705 + 512);

        initializeAdaptiveProbabilityRangeList(range, cumProb);
        sdkResetTimer(&timer);
        sdkStartTimer(&timer);
        //r = arEncode((const unsigned char*)original,strlen(original),compressed,705+512,range);
        //r = lzCompress(original,(unsigned char*) compressed, strlen(original),hashtable,input);

        r = arCompress((const unsigned char *)original, strlen(original), (unsigned char *)compressed, *range, cumProb);
        //r = arCompress((const unsigned char*)original, strlen(original),(unsigned char*) compressed,range);
        sdkStopTimer(&timer);
        std::cout << "glz...Size:" << r << ", Time: " << sdkGetTimerValue(&timer) << std::endl;

        memset(original, 0, strlen(original));
        initializeAdaptiveProbabilityRangeList(range, cumProb);
        sdkResetTimer(&timer);
        sdkStartTimer(&timer);

        r = arDecompress((const unsigned char *)compressed, r, (unsigned char *)original, *range, cumProb);
        //r = lzDecompress(compressed, (unsigned char*)original);
        sdkStopTimer(&timer);
        std::cout << original << std::endl;
        std::cout << "glz decompressed...Size:" << r << ", Time: " << sdkGetTimerValue(&timer) << std::endl;
    }

    //cudaFreeHost(compressed);
    //cudaFreeHost(input);
    //cudaFreeHost(hashtable);
    exit(0);
}