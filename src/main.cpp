#include "cpu_compressor.hpp"
#include "gpu_compressor.hpp"


using namespace gip;

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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    //test(argc,argv);

    bool interactive = true;

    try
    {
        bool showeHelp = (argc <= 1);
        char *inputFileNameBuffer = NULL;
        char *outputFileNameBuffer = NULL;
        int deviceID = -1;
        bool hostMode = false;
        bool hasInput;
        bool hasOutput;
        string inputName;
        string outputName;
        bool hasDevice;
        bool hasCompress;
        int numDevices;

        //if (argc > 1) {

        if (showeHelp || checkCmdLineFlag(argc, (const char **)argv, "help"))
        {
            std::cout << "Usage: gpuar [options] --in=inputfile --out=outputfile" << std::endl
                      << std::endl;
            std::cout << "where options inclide:" << std::endl;
            std::cout << "c             compress input file" << std::endl;
            std::cout << "d             decompress input file" << std::endl;
            std::cout << "--in          input file" << std::endl;
            std::cout << "--out         outputt file" << std::endl;
            std::cout << "--help        print this help message" << std::endl;
            std::cout << "--host        execute kernel code on host (cpu mode), otherwise execute kernel code on CUDA device" << std::endl;

            std::cout << "--device      specify CUDA device, otherwise use device with highest Gflops/s" << std::endl;
            std::cout << "--nointeractive no interactive mode" << std::endl;

            exit(0);
        }
        else
        {
            hasCompress = !checkCmdLineFlag(argc, (const char **)argv, "d");
            hostMode = checkCmdLineFlag(argc, (const char **)argv, "host");
            hasInput = getCmdLineArgumentString(argc, (const char **)argv, "in", (char **)&inputFileNameBuffer);
            hasOutput = getCmdLineArgumentString(argc, (const char **)argv, "out", (char **)&outputFileNameBuffer);
            deviceID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
            interactive = !checkCmdLineFlag(argc, (const char **)argv, "nointeractive");
            cudaGetDeviceCount(&numDevices);

            if (deviceID > 0)
            {
                hasDevice = true;
            }
            else
            {
                hasDevice = false;
            }

            if (!hasInput)
            {
                throw string("Please specify the input file name by command: --in filename");
            }
            else
            {
                inputName = string(inputFileNameBuffer);
            }

            if (!hasOutput)
            {
                outputName = string("output.gip");
            }
            else
            {
                outputName = string(outputFileNameBuffer);
            }

            ProgressMonitor *monitor = new ProgressMonitor();
            Compressor *compressor = NULL;
            CompressionInfo info;
            double compressionRatio;

            if (hostMode || numDevices <= 0)
            {
                compressor = new CPUCompressor();
                std::cout << "Attention: execute kernel code on host." << std::endl;
            }
            else
            {
                compressor = new GPUCompressor();

                if (hasDevice && deviceID >= 0)
                {
                    std::cout << "Choose CUDA device: " << deviceID << "." << std::endl;
                    ((GPUCompressor *)compressor)->chooseDevice(deviceID);
                }
            }

            compressor->setOpenFileName(inputName);
            compressor->setSaveFileName(outputName);

            if (hasCompress)
            {
                std::cout << "Start to compress " << inputName << " to " << outputName << "." << std::endl;
                info = compressor->compress(monitor);
            }
            else
            {
                std::cout << "Start to decompress " << inputName << " to " << outputName << "." << std::endl;
                info = compressor->decompress(monitor);
            }

            compressionRatio = (double)info.compressedFileSize / info.uncompressedFileSize;

            std::cout << "Complete" << std::endl
                      << std::endl;
            std::cout << "Statistics: " << std::endl;
            std::cout << "Uncompressed file size " << info.uncompressedFileSize << " bytes" << std::endl;
            std::cout << "Compressed file size  " << info.compressedFileSize << " bytes" << std::endl;
            std::cout << "Compression ratio     " << compressionRatio << std::endl;
            std::cout << "Compute time          " << info.processTime / 1000 << " s" << std::endl;
            std::cout << "I/O time              " << info.ioTime / 1000 << " s" << std::endl;
            std::cout << "Score                 " << (1000 / (pow(compressionRatio, 0.6) * pow(info.processTime / 1000, 0.4))) << std::endl;
            //compressor->generateRandomFile(64*1024*1024);

            delete compressor;
            delete monitor;

            compressor = NULL;
            monitor = NULL;
        }
    }
    catch (char *e)
    {
        std::cerr << e << std::endl;
    }
    catch (string s)
    {
        std::cerr << s << std::endl;
    }
    //ArEncodeFile("data/out.glz", "data/out2.glz",false);
    if (interactive)
    {
        exit(0);
    }
}
