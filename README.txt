GPU-Based Computation of 2D Least Median of Squares - 
                 Description of the code
---------------------------------------------------------------
Author: Gil Shapira, Open University Israel
Last updated: October 1, 2015

This readme file describes the CUDA based LMS regression code that is located for download in https://github.com/ligaripash/CudaLMS2D.git.
This project is a part of a research paper which can be found here: https://github.com/ligaripash/CudaLMS2D/files/5884/fast_gpu_based_2d_lms_and_applications.pdf

To build the solution you need to install NSIGHT (CUDA dev env for visual studio)- https://developer.nvidia.com/nvidia-nsight-visual-studio-edition
The code is written in C++. No third party libraries are used.
For visualization the program dumps it's output (the input points and LMS line) in a giv file format.
GIV is an easy to use open source software for geomerty primitive representation: http://giv.sourceforge.net/giv/

To run the software you'll need NVIDIA GPU with compute capability 3.0 (Kepler or later platforms).

The structure of the README file is as follows:


1. Input
---------
The input file is the only argument to the program. Specify full path or use active directory specification in the visual studio environment.
The number of points in the input file should be a power of two.
On the root directory you'll find an input file example named dat.txt.
The format of the input file is as follows:
The first number is the point count, the second is the points dimension (currently only dim = 2 is supported)
Later we have one point per line. The first number is the X coordinate of the point and second is the y coordinate.
For instance:

3 2
11 12
22.2 22
33.1 11
10 9

Is a valid input file containing 4 points.


2. Output
----------

1. The program outputs the LMS parameters (slope intercept and slab height) to stdout.
2. a GIV file depicting the input file with the LMS strip is dumped to dump_cuda_lms.giv. You can use the GIV application to open the file and view the data

3. Caveats
----------

Currently the maximum point count the program can handle is defined in a macro in LMS2D.cu.
It is set MAX_POINT_COUNT = 2048. Change this contstant to handle larger point sets.
