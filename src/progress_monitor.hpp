#pragma once

//#include "gpulz.h"
#include <string.h>
#include <math.h>
#include <string>
#include "compress_info.hpp"


using namespace std;

namespace gip
{
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






} // namespace gip