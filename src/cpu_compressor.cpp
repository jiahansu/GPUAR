#include "cpu_compressor.hpp"
#include "file_header.hpp"

using namespace gip;

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