#pragma once

//#include "gpulz.h"
#include <string.h>
#include <math.h>
#include <string>
#include "gpuar.h"

using namespace std;

namespace gip
{

class CompressionInfo
{
  public:
    double ratio;
    double processTime;
    double ioTime;
    size_t processedUncompressedSize;
    size_t compressedFileSize;
    size_t uncompressedFileSize;

    CompressionInfo()
    {
        ratio = 0;
        processTime = 0;
        ioTime = 0;
        compressedFileSize = 0;
        uncompressedFileSize = 0;
        processedUncompressedSize = 0;
    }
};

class FileHeader
{

  public:
    constexpr static int VERSION_POSITION = 0;
    constexpr static int UNCOMPRESSED_FILE_SIZE_POSITION = VERSION_POSITION + 4;
    constexpr static int COMPRESSED_FILE_SIZE_POSITION = UNCOMPRESSED_FILE_SIZE_POSITION + sizeof(size_t);
    constexpr static int HEADER_LENGTH = COMPRESSED_FILE_SIZE_POSITION + sizeof(size_t);
    constexpr static unsigned char GLZ_VERSION_MAJOR = 0;
    constexpr static unsigned char GLZ_VERSION_MINOR = 1;
    constexpr static unsigned char GLZ_VERSION_REVISION = 0;

  private:
    unsigned char data[HEADER_LENGTH];

  public:
    FileHeader()
    {
        data[0] = GLZ_VERSION_MAJOR;
        data[1] = GLZ_VERSION_MINOR;
        data[2] = GLZ_VERSION_REVISION;
    }

    void *getData()
    {
        return &data;
    }

    void setData(const void *data)
    {
        memcpy(this->getData(), data, HEADER_LENGTH);
    }

    gip::CompressionInfo getInfo()
    {
        int *d = (int *)(data + UNCOMPRESSED_FILE_SIZE_POSITION);
        int *c = (int *)(data + COMPRESSED_FILE_SIZE_POSITION);

        gip::CompressionInfo info;

        info.uncompressedFileSize = d[0];
        info.compressedFileSize = c[0];

        return info;
    }

    void setUncompressedFileSize(const size_t size)
    {
        unsigned int *d = (unsigned int *)(data + UNCOMPRESSED_FILE_SIZE_POSITION);

        d[0] = size;
    }
    void setCompressedFileSize(const size_t size)
    {
        unsigned int *d = (unsigned int *)(data + COMPRESSED_FILE_SIZE_POSITION);

        d[0] = size;
    }

    bool checkHeaderVersion()
    {
        return data[0] == GLZ_VERSION_MAJOR && data[1] == GLZ_VERSION_MINOR && data[2] == GLZ_VERSION_REVISION;
    }
};

class ProgressMonitor
{
  private:
    double currentRatio;

  public:
    ProgressMonitor()
    {
        this->reset();
    }
    void updateProgress(const CompressionInfo *info);
    void reset()
    {
        currentRatio = 0;
    }
};

class Compressor
{
  protected:
    string openFileName;
    string saveFileName;
    StopWatchInterface *process_timer;
    StopWatchInterface *io_timer;

    FILE *openFile;
    FILE *saveFile;

    unsigned char *compressed;
    char *data;
    //unsigned char *input;
    //short *hashtable;
    AdaptiveProbabilityRange *range;

  public:
    size_t getFileSize(FILE *stream)
    {
        size_t t;
        fseek(openFile, 0, SEEK_SET);
        t = ftell(openFile);
        fseek(openFile, 0, SEEK_END);
        t = ftell(openFile) - t;

        return t;
    }
    void setOpenFileName(const string fileName)
    {
        this->openFileName = fileName;
    }
    void setSaveFileName(const string fileName)
    {
        this->saveFileName = fileName;
    }
    virtual CompressionInfo compress(ProgressMonitor *monitor) = 0;

    virtual CompressionInfo decompress(ProgressMonitor *monitor) = 0;
    Compressor();
    ~Compressor();

    void closeFiles()
    {
        if (this->saveFile != NULL)
        {
            fclose(this->saveFile);
        }
        if (this->openFile != NULL)
        {
            fclose(this->openFile);
        }
    }

    void generateRandomFile(const size_t size);
};

class CPUCompressor : public Compressor
{
  public:
    CPUCompressor();
    ~CPUCompressor();
    CompressionInfo compress(ProgressMonitor *monitor);
    CompressionInfo decompress(ProgressMonitor *monitor);
};

class GPUCompressor : public Compressor
{
  private:
    const static int NUM_STREAMS = 4;
    //const static int TOTAL_THREADS = NUM_GRIDS*NUM_THREADS;
    //const static int MAX_INPUT_DATA_BLOCK=UNCOMPRESSED_PACKET_SIZE*TOTAL_THREADS;
    uint32_t numBlocks;
    uint32_t totalThreads;
    size_t uncompressedDataBufferSize;
    cudaStream_t inputStreams[NUM_STREAMS]; //(cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    cudaStream_t outputStream;              //(cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));

    uint8_t *deviceUncompressedData;
    uint8_t *deviceCompressedData;
    //char* deviceBuffer;
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

    __inline static unsigned short getPacketSize(const uint8_t *packet)
    {
        return ((unsigned short *)(((char *)packet)))[0];
    }
};

} // namespace gip