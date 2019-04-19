# GPUAR
A CUDA implementation of Arithmetic Coding

## System Requirement
1. CUDA compatible-GPU which supports compute capability 2.1 or higher.
2. Cuda Toolkit 9.0 or higher for compiling.

## Compile
``` bash
make
```
## Compression file by GPU
``` bash
./gpuar c --in data/random_64m.dat --out compress.gip
```
## Decompression file by GPU
``` bash
./gpuar d --in compress.gip --out decompress.dat
```
## Compress file by CPU (--host)
``` bash
./gpuar c --host --in data/random_64m.dat --out compress.gip
```
## Check corectness of file compression/decompression
Using md5sum command to print hash code from decompression file
``` bash
md5sum decompress.dat
md5sum data/random_64m.dat  
```
