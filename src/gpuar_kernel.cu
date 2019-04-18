/***************************************************************************
*                 Arithmetic Encoding and Decoding Library
*
*   Purpose : Use arithmetic coding to compress/decompress streams
*   Original author for host code implementation: Michael Dipperstein
*   Modify to CUDA implementation: Jia-Han Su.
*   Date    : September 2, 2009
*
*****************************************************************************/
#include "gpuar.h"
#include "assert.h"

//texture<unsigned char, 1, cudaReadModeElementType> tex;

__device__ __constant__ AdaptiveProbabilityRange INITIALIZED_RANGE[1];
__device__ __constant__ probability_t INITIALIZED_CUMULATIVE_PROB;

__host__ __device__ unsigned int read(void const *src, unsigned int bytes)
{

	unsigned char *p = (unsigned char*)src;
	switch (bytes)
	{
		case 4:
			return(*p | *(p + 1) << 8 | *(p + 2) << 16 | *(p + 3) << 24);
		case 3: 
			return(*p | *(p + 1) << 8 | *(p + 2) << 16);
		case 2:
			return(*p | *(p + 1) << 8);
		case 1: 
			return(*p);
	}
	return 0;
}

__host__ __device__ unsigned short getCompressedSize(const void* src){

	return read(((const char*)src),2);
}

__host__ __device__ unsigned short getUncompressedSize(const void* src){

	return read(((const char*)src)+2,2);
}

__host__ __device__ void write(unsigned int f, void *dst, size_t bytes)
{
	
	unsigned char *p = (unsigned char*)dst;

	switch (bytes)
	{
		case 4: 
			*p = (unsigned char)f;
			*(p + 1) = (unsigned char)(f >> 8);
			*(p + 2) = (unsigned char)(f >> 16);
			*(p + 3) = (unsigned char)(f >> 24);
			return;
		case 3:
			*p = (unsigned char)f;
			*(p + 1) = (unsigned char)(f >> 8);
			*(p + 2) = (unsigned char)(f >> 16);
			return;
		case 2:
			*p = (unsigned char)f;
			*(p + 1) = (unsigned char)(f >> 8);
			return;
		case 1:
			*p = (unsigned char)f;
			return;
	}
}




__host__ __device__ __inline int putChar(const int c, BitPointer* stream){
	stream->fp[0] = c;
	++stream->fp;


//		std::cout<<c<<std::endl;

	return c;
}

/***************************************************************************
*   Function   : writeChar
*   Description: This function writes the byte passed as a parameter to the
*                stream passed a parameter.
*   Parameters : c - the character to be written
*                stream - pointer to bit stream to write to
*   Effects    : Writes a byte to the file and updates buffer accordingly.
*   Returned   : On success, the character written, otherwise EOF.
***************************************************************************/
__host__ __device__ __inline int writeChar(const int c, BitPointer *stream)
{
    unsigned char tmp;

    if (stream->bitCount == 0)
    {
        /* we can just put byte from file */
		return putChar(c,stream);
        //return fputc(c, stream->fp);
    }else{

		/* figure out what to write */
		tmp = ((unsigned char)c) >> (stream->bitCount);
		tmp = tmp | ((stream->bitBuffer) << (8 - stream->bitCount));
		putChar(tmp, stream);
		stream->bitBuffer = c;


		return tmp;
    }
}

/***************************************************************************
*   Function   : writeBit
*   Description: This function writes the bit passed as a parameter to the
*                file passed a parameter.
*   Parameters : c - the bit value to be written
*                stream - pointer to bit  stream to write to
*   Effects    : Writes a bit to the bit buffer.  If the buffer has a byte,
*                the buffer is written to the file and cleared.
*   Returned   : On success, the bit value written, otherwise EOF.
***************************************************************************/
__host__ __device__ __inline int writeBit(const int c, BitPointer *stream)
{
    int returnValue = c;

    stream->bitCount++;
    stream->bitBuffer <<= 1;

    if (c != 0)
    {
        stream->bitBuffer |= 1;
    }

    /* write bit buffer if we have 8 bits */
    if (stream->bitCount == 8)
    {
		putChar(stream->bitBuffer,stream);

        /* reset buffer */
        stream->bitCount = 0;
        stream->bitBuffer = 0;
    }

    return returnValue;
}

/***************************************************************************
*   Function   : writeBits   (Little Endian)
*   Description: This function writes the specified number of bits from the
*                memory location passed as a parameter to the file passed
*                as a parameter.   Bits are written LSB to MSB.
*   Parameters : stream - pointer to bit stream to write to
*                bits - pointer to bits to write
*                count - number of bits to write
*   Effects    : Writes bits to the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.  bits is treated as
*                a little endian integer of length >= (count/8) + 1.
*   Returned   : EOF for failure, otherwise the number of bits written.  If
*                an error occurs after a partial write, the partially
*                written bits will not be unwritten.
***************************************************************************/
__host__ __device__ __inline int writeBits(BitPointer *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, tmp;
    int offset, remaining/*, returnValue*/;

    bytes = (unsigned char *)bits;
    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
		writeChar(bytes[offset],stream);
		//returnValue = BitFilePutChar(bytes[offset], stream);

        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* write remaining bits */
        tmp = bytes[offset];
        tmp <<= (8 - remaining);

        while (remaining > 0)
        {
            /*returnValue =*/ writeBit((tmp & 0x80), stream);

            tmp <<= 1;
            remaining--;
        }
    }

    return count;
}


__host__ __device__ __inline int forward(const int symbol){
	return symbol+(symbol&(-symbol));
}

__host__ __device__ __inline int backward(const int symbol){
	return symbol&(symbol-1);
}

__host__ __device__ __inline probability_t getRange(const int symbol,  AdaptiveProbabilityRange& r){
	int i=symbol;
	probability_t h=0;

	while(i!=0){
		h = h+r.ranges[i];
		i = backward(i);
	}

	return h;
}

__host__ __device__ __inline void update(const int symbol,  AdaptiveProbabilityRange& r){
	int i=symbol;

	while(i<=UPPER(EOF_CHAR)){
		r.ranges[i] = r.ranges[i]+1;
		i = forward(i);
	}

}

/***************************************************************************
*   Function   : applySymbolRange
*   Description: This function is used for both encoding and decoding.  It
*                applies the range restrictions of a new symbol to the
*                current upper and lower range bounds of an encoded stream.
*                If an adaptive model is being used, the probability range
*                list will be updated after the effect of the symbol is
*                applied.
*   Parameters : symbol - The symbol to be added to the current code range
*
*   Effects    : The current upper and lower range bounds are adjusted to
*                include the range effects of adding another symbol to the
*                encoded stream.  If an adaptive model is being used, the
*                probability range list will be updated.
*   Returned   : None
***************************************************************************/
__host__ __device__ __inline void applySymbolRange(const int symbol,  AdaptiveProbabilityRange& r, probability_t &lower,probability_t &upper, probability_t &cumulativeProb)
{
    unsigned int range;        /* must be able to hold max upper + 1 */
    unsigned int rescaled;     /* range rescaled for range of new symbol */
                                /* must hold range * max upper */

    /* for updating dynamic models */
    //int i;
    

    /***********************************************************************
    * Calculate new upper and lower ranges.  Since the new upper range is
    * dependant of the old lower range, compute the upper range first.
    ***********************************************************************/
    range = (unsigned int)(upper - lower) + 1;         /* current range */

    /* scale upper range of the symbol being coded */
    rescaled = (unsigned int)getRange(UPPER(symbol),r) * range;
    rescaled /= (unsigned int)cumulativeProb;

    /* new upper = old lower + rescaled new upper - 1*/
    upper = lower + (probability_t)rescaled - 1;

    /* scale lower range of the symbol being coded */
    rescaled = (unsigned int)getRange(LOWER(symbol),r) * range;
    rescaled /= (unsigned int)cumulativeProb;

    /* new lower = old lower + rescaled new upper */
    lower = lower + (probability_t)rescaled;


        /* add new symbol to model */
        ++cumulativeProb;
		
        update(UPPER(symbol),r);
		/*
		for (i = UPPER(symbol); i <= UPPER(EOF_CHAR); i++)
        {
            
            r.ranges[i] += 1;
        }*/

        /* half current values if cumulativeProb is too large */
		  /*
        if (r.cumulativeProb >= MAX_PROBABILITY)
        {
            probability_t original=0;    
			probability_t delta;       

			r.cumulativeProb = 0;
            
			#pragma unroll 256
            for (i = 1; i <= UPPER(EOF_CHAR); i++)
            {
                delta = r.ranges[i] - original;
                if (delta <= 2)
                {
                    
                    original = r.ranges[i];
                    r.ranges[i] = r.ranges[i - 1] + 1;
                }
                else
                {
                    original = r.ranges[i];
                    r.ranges[i] = r.ranges[i - 1] + (delta >>1);
                }

                r.cumulativeProb += (r.ranges[i] - r.ranges[i - 1]);
            }
        }*/


#ifdef _DEBUG
    if (lower > upper)
    {
        /* compile this in when testing new models. */
        assert("Panic: out of range");
		//std::cout<< "Panic: lower ("<< lower<<")> upper ("<<upper<<std::endl;
        //fprintf(stderr, "Panic: lower (%X)> upper (%X)\n", lower, upper);
    }
#endif
}

/***************************************************************************
*   Function   : writeEncodedBits
*   Description: This function attempts to shift out as many code bits as
*                possible, writing the shifted bits to the encoded output
*                file.  Only bits that will be unchanged when additional
*                symbols are encoded may be written out.
*
*                If the n most significant bits of the lower and upper range
*                bounds match, they will not be changed when additional
*                symbols are encoded, so they may be shifted out.
*
*                Adjustments are also made to prevent possible underflows
*                that occur when the upper and lower ranges are so close
*                that encoding another symbol won't change their values.
*   Parameters : bfpOut - pointer to open stream to write to.
*   Effects    : The upper and lower code bounds are adjusted so that they
*                only contain only bits that may be affected by the
*                addition of a new symbol to the encoded stream.
*   Returned   : None
***************************************************************************/
__host__ __device__ __inline void writeEncodedBits(BitPointer *bfpOut, probability_t &lower,probability_t &upper, probability_t &underflowBits)
{
    for (;;)
    {
        if ((upper & MASK_BIT(0)) == (lower & MASK_BIT(0)))
        {
            /* MSBs match, write them to output file */
            writeBit((upper & MASK_BIT(0)) != 0, bfpOut);

            /* we can write out underflow bits too */
            while (underflowBits > 0)
            {
                writeBit((upper & MASK_BIT(0)) == 0, bfpOut);
                underflowBits--;
            }
        }
        else if ((lower & MASK_BIT(1)) && !(upper & MASK_BIT(1)))
        {
            /***************************************************************
            * Possible underflow condition: neither MSBs nor second MSBs
            * match.  It must be the case that lower and upper have MSBs of
            * 01 and 10.  Remove 2nd MSB from lower and upper.
            ***************************************************************/
            underflowBits += 1;
            lower &= ~(MASK_BIT(0) | MASK_BIT(1));
            upper |= MASK_BIT(1);

            /***************************************************************
            * The shifts below make the rest of the bit removal work.  If
            * you don't believe me try it yourself.
            ***************************************************************/
        }
        else
        {
            /* nothing left to do */
            return ;
        }

        /*******************************************************************
        * Shift out old MSB and shift in new LSB.  Remember that lower has
        * all 0s beyond it's end and upper has all 1s beyond it's end.
        *******************************************************************/
        lower <<= 1;
        upper <<= 1;
        upper |= 1;
    }
}

/***************************************************************************
*   Function   : writeRemaining
*   Description: This function writes out all remaining significant bits
*                in the upper and lower ranges and the underflow bits once
*                the last symbol has been encoded.
*   Parameters : bfpOut - pointer to open stream to write to.
*   Effects    : Remaining significant range bits are written to the output
*                file.
*   Returned   : None
***************************************************************************/
__host__ __device__ __inline void writeRemaining(BitPointer *bfpOut,probability_t &lower,probability_t &upper,probability_t &underflowBits      /* current underflow bit count */)
{
    writeBit((lower & MASK_BIT(1)) != 0, bfpOut);

    /* write out any unwritten underflow bits */
    for (underflowBits++; underflowBits > 0; underflowBits--)
    {
        writeBit((lower & MASK_BIT(1)) == 0, bfpOut);
    }
}

/***************************************************************************
*   Function   : initializeAdaptiveProbabilityRangeList
*   Description: This routine builds the initial global list of upper and
*                lower probability ranges for each symbol.  This routine
*                is used by both adaptive encoding and decoding.
*                Currently it provides a uniform symbol distribution.
*                Other distributions might be better suited for known data
*                types (such as English text).
*   Parameters : NONE
*   Effects    : ranges array is made to contain initial probability ranges
*                for each symbol.
*   Returned   : NONE
***************************************************************************/
__host__ __device__ __inline void initializeAdaptiveProbabilityRangeList(AdaptiveProbabilityRange* r, probability_t& cumulativeProb)
{
    int c;

    cumulativeProb = 0;
    r->ranges[0] = 0;          /* absolute lower range */
	
	
    /* assign upper and lower probability ranges assuming */
	memset(r->ranges,0,sizeof(probability_t)*(UPPER(EOF_CHAR)+1) );
    #pragma unroll 256
    for (c = 1; c <= UPPER(EOF_CHAR); c++)
    {
        
		update(c,*r);
        ++cumulativeProb;
    }
}

/***************************************************************************
*   Function   : writeClose
*   Description: This function closes a bit file and frees all associated
*                data.
*   Parameters : stream - pointer to bit stream being closed
*   Effects    : The specified file will be closed and the file structure
*                will be freed.
*   Returned   : 0 for success or EOF for failure.
***************************************************************************/
__host__ __device__ __inline void writeClose(BitPointer *stream)
{

    /* write out any unwritten bits */
    if (stream->bitCount != 0)
    {
        (stream->bitBuffer) <<= 8 - (stream->bitCount);
        putChar(stream->bitBuffer, stream);   /* handle error? */
     }
}


__host__ __device__ BitPointer createBitPointer(unsigned char *data){
	BitPointer bp;
	
	bp.fp = data;                 /* file pointer used by stdio functions */
                  
    bp.bitBuffer=0;    /* bits waiting to be read/written */
    bp.bitCount=0;     /* number of bits in bitBuffer */
    
    return bp;
}	



__host__ void initConstantRange(){
	AdaptiveProbabilityRange r;
	probability_t cumulativeProb;
	initializeAdaptiveProbabilityRangeList(&r,cumulativeProb);
	cudaMemcpyToSymbol(INITIALIZED_RANGE[0],&r,sizeof(AdaptiveProbabilityRange));
	cudaMemcpyToSymbol(INITIALIZED_CUMULATIVE_PROB, &cumulativeProb,sizeof(probability_t));
}

/***************************************************************************
*   Function   : arCompress
*   Description: This routine generates a list of arithmetic code ranges for
*                a file and then uses them to write out an encoded version
*                of that file.
*   Parameters : inFile - Pointer of stream to encode
*                outFile - Pointer of stream to write encoded output to
*   Effects    : Binary data is arithmetically encoded
*   Returned   : TRUE for success, otherwise FALSE.
***************************************************************************/
__host__ __device__ size_t arCompress(const unsigned char *fpIn,const size_t size ,unsigned char *outFile,AdaptiveProbabilityRange& r, probability_t& cumulativeProb)
{
     unsigned char c;
    BitPointer bfpOut = createBitPointer(outFile+PACKET_HEADER_LENGTH);                 /* encoded output */
	
	    /* initialize coder start with full probability range [0%, 100%) */
    probability_t lower = 0;
    probability_t upper = ~0;                     /* all ones */
    probability_t underflowBits = 0;
	size_t length;
	ulonglong2 element;
	ulonglong2* elementPointer = (ulonglong2*)fpIn;
	size_t elementCount = ceil((float)size/(float)sizeof(ulonglong2));
	size_t remaining=size;
	unsigned int bytesOffset;

	

     /* initialize probability ranges asumming uniform distribution */
    /*
	#ifdef _DEBUG
	if(r->cumulativeProb!=256){
		assert("AdaptiveProbabilityRange was not initialized yet!");
	}

	#endif
	*/
	for(size_t i=0;i<elementCount;++i){
		element = elementPointer[i];
		bytesOffset = 0;
		//dataPointer = (unsigned char*)(&element);
		/* encode symbols one at a time */
		
		while(bytesOffset<sizeof(unsigned long long)&&remaining>0)
		{
			c =(unsigned char)(element.x >> (bytesOffset*8) );
			//c = fpIn[j];
			applySymbolRange(c,r,lower,upper, cumulativeProb);
			writeEncodedBits(&bfpOut,lower,upper,underflowBits);
			++bytesOffset;
			--remaining;
		}

		bytesOffset = 0;
		while(bytesOffset<sizeof(unsigned long long)&&remaining>0)
		{
			c =(unsigned char)(element.y >> (bytesOffset*8) );
			//c = fpIn[j];
			applySymbolRange(c,r,lower,upper,cumulativeProb);
			writeEncodedBits(&bfpOut,lower,upper,underflowBits);
			++bytesOffset;
			--remaining;
		}
    
    }

   // applySymbolRange(EOF_CHAR, r,lower,upper);    /* encode an EOF */
   // writeEncodedBits(&bfpOut,lower,upper,underflowBits);

    writeRemaining(&bfpOut,lower,upper,underflowBits);         /* write out least significant bits */
    writeClose(&bfpOut);
	length = bfpOut.fp-outFile;

	
	write(length, outFile ,2);
	write(size, outFile +2,2);

	return length;
}



__host__ __device__ __inline int getChar(BitPointer *stream){
	//;
	
	int x = stream->fp[0];
		++stream->fp;

	return x;

}


/***************************************************************************
*   Function   : readBit
*   Description: This function returns the next bit from the file passed as
*                a parameter.  The bit value returned is the msb in the
*                bit buffer.
*   Parameters : stream - pointer to bit stream to read from
*   Effects    : Reads next bit from bit buffer.  If the buffer is empty,
*                a new byte will be read from the file.
*   Returned   : 0 if bit == 0, 1 if bit == 1, and EOF if operation fails.
***************************************************************************/
__host__ __device__ int readBit(BitPointer *stream)
{
    int returnValue;

    if (stream->bitCount == 0)
    {
            returnValue = getChar(stream);
            stream->bitCount = 8;
            stream->bitBuffer = returnValue;
    }

    /* bit to return is msb in buffer */
    stream->bitCount--;
    returnValue = (stream->bitBuffer) >> (stream->bitCount);

    return (returnValue & 0x01);
}

/****************************************************************************
*   Function   : initializeDecoder
*   Description: This function starts the upper and lower ranges at their
*                max/min values and reads in the most significant encoded
*                bits.
*   Parameters : bfpIn - stream to read from
*   Effects    : upper, lower, and code are initialized.  The probability
*                range list will also be initialized if an adaptive model
*                will be used.
*   Returned   : TRUE for success, otherwise FALSE
****************************************************************************/
__host__ __device__  void initializeDecoder(BitPointer *bfpIn, probability_t &lower,probability_t &upper,probability_t &code  )
{
    int i;

    code = 0;

    /* read PERCISION MSBs of code one bit at a time */
    for (i = 0; i < PRECISION; i++)
    {
        code <<= 1;

        /* treat EOF like 0 */
        if(readBit(bfpIn) == 1)
        {
            code |= 1;
        }
    }

    /* start with full probability range [0%, 100%) */
    lower = 0;
    upper = ~0;         /* all ones */
}


/***************************************************************************
*   Function   : readChar
*   Description: This function returns the next byte from the file passed as
*                a parameter.
*   Parameters : stream - pointer to bit stream to read from
*   Effects    : Reads next byte from file and updates buffer accordingly.
*   Returned   : EOF if a whole byte cannot be obtained.  Otherwise,
*                the character read.
***************************************************************************/
__host__ __device__ int readChar(BitPointer *stream)
{
    int returnValue;
    unsigned char tmp;

    returnValue = getChar(stream);

    if (stream->bitCount == 0)
    {
        /* we can just get byte from file */
        return returnValue;
    }

    /* we have some buffered bits to return too */

        /* figure out what to return */
        tmp = ((unsigned char)returnValue) >> (stream->bitCount);
        tmp |= ((stream->bitBuffer) << (8 - (stream->bitCount)));

        /* put remaining in buffer. count shouldn't change. */
        stream->bitBuffer = returnValue;

        returnValue = tmp;

    return returnValue;
}

/***************************************************************************
*   Function   : readBits  (Little Endian)
*   Description: This function reads the specified number of bits from the
*                file passed as a parameter and writes them to the
*                requested memory location (LSB to MSB).
*   Parameters : stream - pointer to bit stream to read from
*                bits - address to store bits read
*                count - number of bits to read
*   Effects    : Reads bits from the bit buffer and file stream.  The bit
*                buffer will be modified as necessary.  bits is treated as
*                a little endian integer of length >= (count/8) + 1.
*   Returned   : EOF for failure, otherwise the number of bits read.  If
*                an EOF is reached before all the bits are read, bits
*                will contain every bit through the last successful read.
***************************************************************************/
__host__ __device__ int readBits(BitPointer *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes/*, shifts*/;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;

    offset = 0;
    remaining = count;

    /* read whole bytes */
    while (remaining >= 8)
    {
        returnValue = readChar(stream);

        bytes[offset] = (unsigned char)returnValue;
        remaining -= 8;
        offset++;
    }

    if (remaining != 0)
    {
        /* read remaining bits */
        //shifts = 8 - remaining;

        while (remaining > 0)
        {
            returnValue = readBit(stream);

            bytes[offset] <<= 1;
            bytes[offset] |= (returnValue & 0x01);
            remaining--;
        }

    }

    return count;
}



/****************************************************************************
*   Function   : getUnscaledCode
*   Description: This function undoes the scaling that ApplySymbolRange
*                performed before bits were shifted out.  The value returned
*                is the probability of the encoded symbol.
*   Parameters : None
*   Effects    : None
*   Returned   : The probability of the current symbol
****************************************************************************/
__host__ __device__ probability_t getUnscaledCode(probability_t &lower,probability_t &upper,probability_t &code, probability_t& cumulativeProb)
{
     unsigned int range;        /* must be able to hold max upper + 1 */
    unsigned int unscaled;

    range = (unsigned int)(upper - lower) + 1;

    /* reverse the scaling operations from ApplySymbolRange */
    unscaled = (unsigned int)(code - lower) + 1;
    unscaled = unscaled * (unsigned int)cumulativeProb - 1;
    unscaled /= range;

    return ((probability_t)unscaled);
}

/****************************************************************************
*   Function   : getSymbolFromProbability
*   Description: Given a probability, this function will return the symbol
*                whose range includes that probability.  Symbol is found
*                binary search on probability ranges.
*   Parameters : probability - probability of symbol.
*   Effects    : None
*   Returned   : -1 for failure, otherwise encoded symbol
****************************************************************************/
__host__ __device__ __inline int getSymbolFromProbability(probability_t probability,AdaptiveProbabilityRange& r)
{
    int first, last, middle;    /* indicies for binary search */

    first = 0;
    last = UPPER(EOF_CHAR);
    middle = last >>1;

    /* binary search */
    while (last >= first)
    {
        if (probability < getRange(LOWER(middle),r))
        {
            /* lower bound is higher than probability */
            last = middle - 1;
            middle = first + ((last - first) >>1);
            continue;
        }

        if (probability >= getRange(UPPER(middle),r))
        {
            /* upper bound is lower than probability */
            first = middle + 1;
            middle = first + ((last - first) >>1);
            continue;
        }

        /* we must have found the right value */
        return middle;
    }
	
    #if _DEBUG
    assert("Unknown Symbol");
    //cuPrintf("Unknown Symbol: %d (max: %d)\n", probability, r.ranges[UPPER(EOF_CHAR)]);
    #endif
    return -1;
}


/***************************************************************************
*   Function   : readEncodedBits
*   Description: This function attempts to shift out as many code bits as
*                possible, as bits are shifted out the coded input is
*                populated with bits from the encoded file.  Only bits
*                that will be unchanged when additional symbols are decoded
*                may be shifted out.
*
*                If the n most significant bits of the lower and upper range
*                bounds match, they will not be changed when additional
*                symbols are decoded, so they may be shifted out.
*
*                Adjustments are also made to prevent possible underflows
*                that occur when the upper and lower ranges are so close
*                that decoding another symbol won't change their values.
*   Parameters : bfpOut - pointer to open binary stream to read from.
*   Effects    : The upper and lower code bounds are adjusted so that they
*                only contain only bits that will be affected by the
*                addition of a new symbol.  Replacements are read from the
*                encoded stream.
*   Returned   : None
***************************************************************************/
__host__ __device__ void readEncodedBits(BitPointer *bfpIn,probability_t &lower,probability_t &upper,probability_t &code)
{
    int nextBit;        /* next bit from encoded input */

    for (;;)
    {
        if (( upper & MASK_BIT(0)) == (lower & MASK_BIT(0)))
        {
                        /* MSBs match, allow them to be shifted out*/
        }
        else if ((lower & MASK_BIT(1)) && !(upper & MASK_BIT(1)))
        {
            /***************************************************************
            * Possible underflow condition: neither MSBs nor second MSBs
            * match.  It must be the case that lower and upper have MSBs of
            * 01 and 10.  Remove 2nd MSB from lower and upper.
            ***************************************************************/
                        lower   &= ~(MASK_BIT(0) | MASK_BIT(1));
            upper  |= MASK_BIT(1);
            code ^= MASK_BIT(1);

            /* the shifts below make the rest of the bit removal work */
        }
        else
        {
            /* nothing to shift out */
            return;
        }

        /*******************************************************************
        * Shift out old MSB and shift in new LSB.  Remember that lower has
        * all 0s beyond it's end and upper has all 1s beyond it's end.
        *******************************************************************/
        lower <<= 1;
        upper <<= 1;
        upper |= 1;
        code <<= 1;

        if ((nextBit = readBit(bfpIn)) == EOF)
        {
            /* either all bits are shifted out or error occurred */
        }
        else
        {
            code |= nextBit;                /* add next encoded bit to code */
        }
    }

    //return;
}

/***************************************************************************
*   Function   : arDecompress
*   Description: This routine opens an arithmetically encoded file, reads
*                it's header, and builds a list of probability ranges which
*                it then uses to decode the rest of the file.
*   Parameters : inFile - Pointer to stream to decode
*                outFile - Pointer to stream to write decoded output to
*   Effects    : Encoded file is decoded
*   Returned   : TRUE for success, otherwise FALSE.
***************************************************************************/
__host__ __device__ size_t arDecompress(const unsigned char *fpIn,const size_t inSize ,unsigned char *fpOut,AdaptiveProbabilityRange& r, probability_t& cumulativeProb)
{
    int c;
    probability_t unscaled;
     BitPointer bfpIn = createBitPointer((unsigned char*)fpIn+PACKET_HEADER_LENGTH);
    

		    /* initialize coder start with full probability range [0%, 100%) */
    probability_t lower;
    probability_t upper;                     /* all ones */
    probability_t code;
	unsigned char* dstPointer = fpOut;
	const size_t decompressedSize = getUncompressedSize(fpIn);

	//bfpIn->fp = ;


    /* read start of code and initialize bounds, and adaptive ranges */
   initializeDecoder(&bfpIn,lower,upper,code);

    /* decode one symbol at a time */
    while( (dstPointer-fpOut)< decompressedSize)
    {
        //printf("%02X\t%d\t%d\n", lower, upper, code);
		
		/* get the unscaled probability of the current symbol */
        unscaled = getUnscaledCode(lower,upper,code,cumulativeProb);

        /* figure out which symbol has the above probability */
        if((c = getSymbolFromProbability(unscaled,r)) == -1)
        {
            /* error: unknown symbol */
            break;
        }
		
		dstPointer[0]=c;
		++dstPointer;


        //fputc((char)c, fpOut);

        /* factor out symbol */
        applySymbolRange(c,r,lower,upper, cumulativeProb);
        readEncodedBits(&bfpIn,lower,upper,code);
    }

    return dstPointer-fpOut;
}

__global__ void garCompress(const unsigned char *source, size_t size, unsigned char *destination)
{
  
  __shared__ AdaptiveProbabilityRange sharedMemory[NUM_THREADS];
  const int index = blockIdx.x * blockDim.x+ threadIdx.x;
  const unsigned int startPosition = index*UNCOMPRESSED_PACKET_SIZE;
  AdaptiveProbabilityRange* start =  sharedMemory + threadIdx.x;
  probability_t cumProb = INITIALIZED_CUMULATIVE_PROB;
  
  if(startPosition<size){
	size_t packetSize = size-startPosition;
	  
	start[0]= INITIALIZED_RANGE[0];
	if(packetSize>UNCOMPRESSED_PACKET_SIZE){
	    packetSize = UNCOMPRESSED_PACKET_SIZE;
	}
	arCompress(source+startPosition, packetSize, destination+(index*COMPRESSED_PACKET_SIZE),*start, cumProb);
	}
}

__global__ void garDecompress(const unsigned char *source, size_t size, unsigned char *destination)
{
  
  __shared__ AdaptiveProbabilityRange sharedMemory[NUM_THREADS];
  const int index = blockIdx.x * blockDim.x+ threadIdx.x;
  const size_t startPosition = index*COMPRESSED_PACKET_SIZE;
  const unsigned char * startSrc = source+startPosition;
  AdaptiveProbabilityRange* start =  sharedMemory + threadIdx.x;
  unsigned char* data=   destination+(index*UNCOMPRESSED_PACKET_SIZE);
  probability_t cumProb = INITIALIZED_CUMULATIVE_PROB;
  if(startPosition<size){
	//size_t packetSize = size-startPosition;
	start[0]= INITIALIZED_RANGE[0];
    /*size_t packetSize = */
    arDecompress(startSrc, getCompressedSize(startSrc),data,*start,cumProb);
	 //write(packetSize, data +2,2);
	
	}
	
}

void garCompressExecutor(const unsigned char *source, size_t size, unsigned char *destination,unsigned int numBlocks){
        
        garCompress<<<numBlocks,NUM_THREADS>>>(source, size, destination);

        #ifdef _DEBUG
        getLastCudaError("Execute garCompress kenenl failed");
        #endif
}

void garDecompressExecutor(const unsigned char *source, size_t size, unsigned char *destination,unsigned int numBlocks){
        
        garDecompress<<<numBlocks,NUM_THREADS>>>(source, size, destination);

        #ifdef _DEBUG
        getLastCudaError("Execute garCompress kenenl failed");
        #endif
}
