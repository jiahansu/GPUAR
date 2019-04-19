#pragma once

//#include "gpulz.h"
#include <string.h>
#include <math.h>
#include <string>
#include "gpuar.h"
#include "compressor.hpp"

using namespace std;

namespace gip
{

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

} // namespace gip