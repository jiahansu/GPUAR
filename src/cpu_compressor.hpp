#pragma once

#include "compressor.hpp"

using namespace std;

namespace gip
{
    class CPUCompressor : public Compressor
    {
    public:
        CPUCompressor();
        ~CPUCompressor();
        CompressionInfo compress(ProgressMonitor *monitor);
        CompressionInfo decompress(ProgressMonitor *monitor);
    };
}