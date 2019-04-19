#pragma once

#include "gpuar.h"

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
}