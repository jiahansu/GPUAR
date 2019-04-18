#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include "gpu.h"

/***************************************************************************
*                                CONSTANTS
***************************************************************************/
#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#define EOF_CHAR (UCHAR_MAX)

/***************************************************************************
*                                  MACROS
***************************************************************************/
/* set bit x to 1 in probability_t.  Bit 0 is MSB */
#define MASK_BIT(x) (probability_t)(1 << (PRECISION - (1 + (x))))

/* indices for a symbol's lower and upper cumulative probability ranges */
#define LOWER(c) (c)
#define UPPER(c) ((c) + 1)

typedef unsigned short probability_t; /* probability count type */

/* number of bits used to compute running code values */
#define PRECISION (8 * sizeof(probability_t))

/* 2 bits less than precision. keeps lower and upper bounds from crossing. */
#define MAX_PROBABILITY (1 << (PRECISION - 2))
#define READ_ELEMENT ulonglong2
#define bitbuffer unsigned char

struct __align__(4) AdaptiveProbabilityRange
{

    /* probability ranges for each symbol: [ranges[LOWER(c)], ranges[UPPER(c)]) */
    probability_t ranges[UPPER(EOF_CHAR) + 1];
    //probability_t cumulativeProb;   /* cumulative probability  of all ranges */
};

struct __align__(16) BitPointer
{
    //public:
    unsigned char *fp;      /* file pointer used by stdio functions */
                            //unsigned char *fpEnd;
    bitbuffer bitBuffer;    /* bits waiting to be read/written */
    unsigned char bitCount; /* number of bits in bitBuffer */
};

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    BitPointer createBitPointer(unsigned char *data);
    void writeRemaining(BitPointer *bfpOut, probability_t &lower, probability_t &upper, probability_t &underflowBits /* current underflow bit count */);
    void writeClose(BitPointer *stream);
    void applySymbolRange(const int symbol, AdaptiveProbabilityRange *r, probability_t &lower, probability_t &upper);
    unsigned short getCompressedSize(const void *src);
    void writeEncodedBits(BitPointer *bfpOut, probability_t &lower, probability_t &upper, probability_t &underflowBits);
    unsigned short getUncompressedSize(const void *src);
    void write(unsigned int f, void *dst, size_t bytes);
    unsigned int read(void const *src, unsigned int bytes);
    void initializeAdaptiveProbabilityRangeList(AdaptiveProbabilityRange *r, probability_t &cumProb);
    void initConstantRange();
    size_t arCompress(const unsigned char *fpIn, const size_t inSize, unsigned char *outFile, AdaptiveProbabilityRange &r, probability_t &cumProb);
    size_t arDecompress(const unsigned char *fpIn, const size_t inSize, unsigned char *fpOut, AdaptiveProbabilityRange &r, probability_t &cumProb);
    void garCompressExecutor(const unsigned char *source, size_t size, unsigned char *destination, unsigned int numBlocks);
    void garDecompressExecutor(const unsigned char *source, size_t size, unsigned char *destination, unsigned int numBlocks);
    void initializeDecoder(BitPointer *bfpIn, probability_t &lower, probability_t &upper, probability_t &code);
    probability_t getUnscaledCode(probability_t &lower, probability_t &upper, probability_t &code, probability_t &cumProb);
    void readEncodedBits(BitPointer *bfpIn, probability_t &lower, probability_t &upper, probability_t &code);
    //void write2(unsigned short  f, void *dst);

#ifdef __cplusplus
}
#endif /* __cplusplus */