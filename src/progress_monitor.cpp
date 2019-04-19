
/*
* NOTICE TO USER:
*
* This source code is subject to Jia-Han Su.
*/

/*
   GPUAR source code
   Version 0.1 - first release
*/

#include "progress_monitor.hpp"

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

