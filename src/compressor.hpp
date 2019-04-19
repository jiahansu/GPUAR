#pragma once

#include "compress_info.hpp"
#include "progress_monitor.hpp"

using namespace std;

namespace gip
{
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
}