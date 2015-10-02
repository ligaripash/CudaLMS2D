
#include "LMS2D.h"

#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <Windows.h>
using namespace std;
#include <thrust/device_vector.h>
#include <sstream>



#define MAX_POINT_COUNT 2048


/*******************************************************************************************************/



LMS2D::LMS2D()
{
	initCUDA();
}


/*******************************************************************************************************/



LMS2D::~LMS2D()
{

//	cudaFree(mDeviceYcoordinatesPerVerticalLine);
	cudaFree(mDeviceIntersectionPoints);
	cudaFree(mDeviceMinBraceletPerIntersectionPoint);
	cudaFree(mDeviceMinBraceletMidPintPerIntersectionPoint);
	cudaFree(mDeviceInputLines);
	
	
	delete [] mHostInputLines;
	//free(mHostInputLines);
	free(mHostMinBraceletMidPintPerIntersectionPoint);
	//free(mHostMinBraceletPerIntersectionPoint );
	//free(mHostIntersectionPoints);
//	free(mHostMinBraceletReductionOutput);
	 

	

	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cerr<<"cudaDeviceReset failed!";
     
    }
}


/*******************************************************************************************************/


void LMS2D::initCUDA()
{
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		cerr << "cudaSetDevice failed!";
    }
}



/*******************************************************************************************************/

__device__ float calcYatXIntersection(int line_index, float* input_lines, float x)
{
	float m = input_lines[2*line_index];
	float n = input_lines[2*line_index + 1];

	float y = m * x + n;

	return y;
}


/*******************************************************************************************************/




__device__ void calculateMinBracelet(float intersection_p_x, 
									 float intersection_p_y, 
									 float* input_lines, 
									 int lines_count, 
									 int index_for_x_coord_per_intersection_point, 
									 float* y_intersection_per_vertical_line,
									 float* min_y_bracelet,
									 float*  bracelet_mid_point)
{

	//so..
	//Now we have an intersection point. We have to calculate the min k bracelet starting at this intersection point.
	//at first attempt we'll do the following:

	//1. calculate all intersection points with the line X=Xint.
	//2. Sort these points according to their y coordinate.
	//3. Find minimum bracelet for the point.
	//4. Find the minimum bracelet of all points.

	
	int intersection_points_count = lines_count * ( lines_count - 1) / 2;

	int index = index_for_x_coord_per_intersection_point;
	for (int i = 0; i < 256; i++){
		float line_y_at_x_intersection = calcYatXIntersection(i, input_lines, intersection_p_x);
//		float line_y_at_x_intersection = 20.0;
		y_intersection_per_vertical_line[index] = line_y_at_x_intersection;
		index += intersection_points_count;
	}

	// do selection sort
	
	int index_of_min = -1;
	float min_value;
	

	// now find the min bracelet for this intersecion point

	int k = lines_count / 2 ; // the median
	index = index_for_x_coord_per_intersection_point;
	min_value = FLT_MAX;

	int index_end_bracelet = index + k * intersection_points_count;

	for (int i = 0; i < k; i++){

		float delta = y_intersection_per_vertical_line[index_end_bracelet] - y_intersection_per_vertical_line[index];
		//float delta = 2.2;
		if (delta < min_value){
			min_value = delta;
			index_of_min = i;
		}
		index_end_bracelet += intersection_points_count;
		index += intersection_points_count;
	}

	// this is the mid point of the bracelet.
	
	*min_y_bracelet = min_value;

	*bracelet_mid_point	= y_intersection_per_vertical_line[index_for_x_coord_per_intersection_point + index_of_min * intersection_points_count] + min_value / 2;

}







/*******************************************************************************************************/




__device__ void calculateInstersectionPoint(int col, int row, float* lines, float* intersection_p_x, float* intersection_p_y)
{
	
	float m1 = lines[2*col];

	float n1 = lines[2*col + 1];


	float m2 = lines[2*row];

	float n2 = lines[2*row + 1];

	if ( m1 == m2 ){
		//	printf("m1 == m2 at row = $d and col = $d", row, col);
		//m1 += FLT_MIN;
		//		*intersection_p_x = *intersection_p_y = -99999.0f;
		//printf("line ");
		//return;
		*intersection_p_x = FLT_MAX;

		*intersection_p_y = FLT_MAX;

		return ;
	}

	*intersection_p_x = (n2 - n1) / (m1 - m2);

	*intersection_p_y = m1 * (*intersection_p_x) + n1;

	
}




/*******************************************************************************************************/



__device__ int getIndex(int col, int row, int line_count)
{
	int index = 2 * (row * line_count + col - ( (row+1) * (row + 2) / 2 ));

	return index;
}



/*******************************************************************************************************/


__global__ void computeIntersectionPoints(float* input_lines,
										  int lines_count,
										  float*  mDeviceIntersectionPoints)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= col) {
		return;
	}

	//now we have to calculate the instersection point of line #col width line #row

	float intersection_p_x;
	float intersection_p_y;

	calculateInstersectionPoint(col, row, input_lines, &intersection_p_x, &intersection_p_y);

	int index_for_x_coord_per_intersection_point = getIndex(col, row, lines_count);
	//int index_for_x_coord_per_intersection_point = 2;

	mDeviceIntersectionPoints[index_for_x_coord_per_intersection_point] = intersection_p_x;
	mDeviceIntersectionPoints[index_for_x_coord_per_intersection_point + 1] = intersection_p_y;


}


/*******************************************************************************************************/


inline __device__ void Comparator(
    float &keyA,
    float &keyB,
    unsigned int arrowDir
)
{
    float t;

    if ((keyA > keyB) == arrowDir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
    }
}



/*******************************************************************************************************/


__device__ void bitonicSortSharedKernel(
    float *s_key,
    int tid,
    unsigned int arrayLength)
{

	int sortDir = 1;

//	#pragma unroll

    for (unsigned int size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        unsigned int dir = (tid & (size >> 1)) != 0;

        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1)
        {
            __syncthreads();
            unsigned int pos = 2 * tid - (tid & (stride - 1));
            Comparator( s_key[pos +      0],  s_key[pos + stride],  dir     );
        }
    }

	
    //ddd == sortDir for the last bitonic merge step
    {
        for (unsigned int stride = arrayLength >> 1; stride > 0; stride >>= 1)
        {
            __syncthreads();
            unsigned int pos = 2 * tid - (tid & (stride - 1));
            Comparator(  s_key[pos +      0], s_key[pos + stride], sortDir);
            
        }
    }

    __syncthreads();
}



/*******************************************************************************************************/



__device__ void calculateMinBraceletSort(float local_y,
									 float intersection_p_y, 
									 int tid,
									 float* smem,			 
									 int line_count,					 
									 float* min_y_bracelet,
									 float* bracelet_mid_point)
{

	//First sort the values in shared memory
	bitonicSortSharedKernel(smem, tid, line_count);

	// Get the intersection_p_y location in the sorted sequence

	float f1 = smem[tid];
	//gil
	//float f2 = smem[tid + line_count / 2 - 1];
	float f2 = smem[tid + line_count / 2 ];

	if (f1 == intersection_p_y){

		if ((tid != 0) && smem[tid - 1] == intersection_p_y){
			return;
		}
		*min_y_bracelet = (abs(f2 - f1));
//		*bracelet_mid_point = (f1 + f2) / 2;
		*bracelet_mid_point = (f1 + f2) ;

	}
	if (f2 == intersection_p_y ){
		if ((tid != line_count - 1) && smem[tid + 1] == intersection_p_y){
			return;
		}

		*min_y_bracelet = (abs(f2 - f1));
//		*bracelet_mid_point = (f1 + f2) / 2;
		*bracelet_mid_point = (f1 + f2) ;

	}
	


}


/*******************************************************************************************************/


template <unsigned int blockSize>
__global__ void findGlobalMinimumBracelet(float* mDeviceMinBraceletPerIntersectionPoint, float* g_out)
{
	
	__shared__ float smem[2*MAX_POINT_COUNT];
	

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	float val1 = mDeviceMinBraceletPerIntersectionPoint[i];
	float val2 = mDeviceMinBraceletPerIntersectionPoint[i + blockSize];
	float my_min = min(val1, val2);

    smem[tid] = my_min;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            smem[tid] = my_min = min(my_min, smem[tid + s]);
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float *sdata= smem ;

        if (blockSize >=  64)
        {
            sdata[tid] = my_min = min(my_min,sdata[tid + 32]);
        }

        if (blockSize >=  32)
        {
            sdata[tid] = my_min = min(my_min, sdata[tid + 16]);
        }

        if (blockSize >=  16)
        {
            sdata[tid] = my_min = min(my_min,sdata[tid +  8]);
        }

        if (blockSize >=   8)
        {
            sdata[tid] = my_min = min(my_min, sdata[tid +  4]);
        }

        if (blockSize >=   4)
        {
            sdata[tid] = my_min = min(my_min, sdata[tid +  2]);
        }

        if (blockSize >=   2)
        {
            sdata[tid] = my_min = min(my_min,sdata[tid +  1]);
        }
    }

    // write result for this block to global mem
    if (tid == 0) g_out[2*blockIdx.x] = smem[0];

	 __syncthreads();

	if (smem[0] == val1){
		g_out[2*blockIdx.x + 1] = i;
	}

	if (smem[0] == val2){
		g_out[2*blockIdx.x + 1] = i + blockSize;
	}

}


/*

We activate this kernel with one thread block per intersection point.
Currently all y intersction will be done in shared memory.

*/

__global__ void FindMinimalBraceletPerIntersectionPoint(float* input_lines,
														int line_count, 
														float*  mDeviceIntersectionPoints, 
														float*  mDeviceMinBraceletPerIntersectionPoint,
														float*  mDeviceMinBraceletMidPintPerIntersectionPoint)
{
	
	//if (blockIdx.x != 0){
	//	return;
	//}
	extern __shared__ float smem[];

	float intersection_x;
	float intersection_y;

	int tid = threadIdx.y;


	intersection_x = mDeviceIntersectionPoints[blockIdx.x * 2];
	intersection_y = mDeviceIntersectionPoints[blockIdx.x * 2 + 1];

	if (intersection_x == FLT_MAX){
		// No intersection point - parallel lines
		if (tid == 0){
			mDeviceMinBraceletPerIntersectionPoint[blockIdx.x] = FLT_MAX;
		}
		
		return;
	}

	
	__syncthreads();

	int offset = line_count >> 1;
	float local_y1 = calcYatXIntersection(tid, input_lines, intersection_x);
	float local_y2 = calcYatXIntersection(tid + offset, input_lines, intersection_x);


	smem[tid] = local_y1;	
	smem[tid + offset] = local_y2;	
	__syncthreads();

	//if (blockIdx.x == 0){
	//	mDeviceMinBraceletPerIntersectionPoint[tid+256] = smem[tid];
	//}

	float min_y_bracelet = FLT_MAX;
	float bracelet_mid_point = FLT_MAX;


	calculateMinBraceletSort(local_y1,
		intersection_y, 
		tid,
		smem,			 
		line_count,					 
		&min_y_bracelet,
		&bracelet_mid_point);


	if (min_y_bracelet != FLT_MAX){

		mDeviceMinBraceletPerIntersectionPoint[blockIdx.x] = min_y_bracelet;
		mDeviceMinBraceletMidPintPerIntersectionPoint[blockIdx.x] = bracelet_mid_point;
	}
		

}


/*******************************************************************************************************/


void LMS2D::findGlobalMinBracelet(float* x_coord, float* min_bracelet, float* y_coord)
{

   int BLOCK_SIZE = 512 ;//was 1024
   
   while ((this->mInputIntersectionPointCount / (2*BLOCK_SIZE) <= 1)){
	   BLOCK_SIZE /= 2;
   }

   dim3 dim_min_bracelet(BLOCK_SIZE, 1);
   dim3 dim_grid_min_bracelet(this->mInputIntersectionPointCount / (2*BLOCK_SIZE),1);

    
   switch (BLOCK_SIZE){
   case 512:
	   findGlobalMinimumBracelet<512><<<dim_grid_min_bracelet, dim_min_bracelet>>>(mDeviceMinBraceletPerIntersectionPoint, mDeviceMinBraceletReductionOutput);
	   break;

   case 256:
	   findGlobalMinimumBracelet<256><<<dim_grid_min_bracelet, dim_min_bracelet>>>(mDeviceMinBraceletPerIntersectionPoint, mDeviceMinBraceletReductionOutput);
	   break;

   case 128:
	   findGlobalMinimumBracelet<128><<<dim_grid_min_bracelet, dim_min_bracelet>>>(mDeviceMinBraceletPerIntersectionPoint, mDeviceMinBraceletReductionOutput);
	   break;

   case 64:
	   findGlobalMinimumBracelet<64><<<dim_grid_min_bracelet, dim_min_bracelet>>>(mDeviceMinBraceletPerIntersectionPoint, mDeviceMinBraceletReductionOutput);
	   break;

   case 32:
	   findGlobalMinimumBracelet<32><<<dim_grid_min_bracelet, dim_min_bracelet>>>(mDeviceMinBraceletPerIntersectionPoint, mDeviceMinBraceletReductionOutput);
	   break;



   }
   

   int item_count_in_output = mInputIntersectionPointCount / (BLOCK_SIZE * 2);

   int output_size = 2 * item_count_in_output;

   cudaMemcpy(mHostMinBraceletReductionOutput, mDeviceMinBraceletReductionOutput,output_size * sizeof(float), cudaMemcpyDeviceToHost);

   float current_min = FLT_MAX;
   int current_index = -1;

   for ( int i = 0; i < item_count_in_output; i += 2){
	   if (mHostMinBraceletReductionOutput[i] < current_min){
		   current_min = mHostMinBraceletReductionOutput[i];
		   current_index = mHostMinBraceletReductionOutput[i+1];
	   }
   }

	cudaMemcpy(x_coord, mDeviceIntersectionPoints + 2 * current_index, sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(y_coord, mDeviceMinBraceletMidPintPerIntersectionPoint + current_index , sizeof(float), cudaMemcpyDeviceToHost);

	*y_coord /= -2;
	*min_bracelet = current_min;

}



/*******************************************************************************************************/


void LMS2D::compute()
{




	// 2. compute the intersection points of each pair of lines.
	// x = n2 - n1 / m1 - m2;
	// y = m1 * x + n1;
	// 3. For each intersection point do:
	//   3.1. Find the relevant direction (either up or down)
	//   3.2. Write all y coordinates above (bellow) each intersection point to memory
	//   3.3. Find the median.
	//   3.4. write the bracelet value to memory.
	// 4. Compute the minimum bracelet value.
	// 5. The LMS line is the dual of (Xmin, Ymin + min_bracelet).


	// Invoke kernel
	
    //dim3 dimBlock(16, 16);
	dim3 dimBlock(8, 8);
    dim3 dimGrid(mInputPointsCount / dimBlock.x, mInputPointsCount / dimBlock.y);


	cudaDeviceSynchronize();

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);


	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);


	//First calculate all intersection points


	cudaEvent_t cuda_start, cuda_stop, sstart, sstop, min_start, min_stop;
	float time_intersection_points, time_bracelet_per_itersection_point, time_minimum_bracelet;
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_stop);
	cudaEventCreate(&sstart);
	cudaEventCreate(&sstop);
	cudaEventCreate(&min_start);
	cudaEventCreate(&min_stop);

	cudaEventRecord( cuda_start, 0 );

	computeIntersectionPoints<<<dimGrid, dimBlock>>>(mDeviceInputLines,
													 mInputPointsCount,
													 mDeviceIntersectionPoints );


	cudaEventRecord( cuda_stop, 0 );
	cudaEventSynchronize( cuda_stop );
	cudaEventElapsedTime( &time_intersection_points, cuda_start, cuda_stop );


	dim3 dim_block_bracelet(1, mInputPointsCount / 2);

	dim3 dim_grid_bracelet(this->mInputIntersectionPointCount,1);

	
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	cudaEventRecord( sstart, 0 );

	cudaError_t rc;
	FindMinimalBraceletPerIntersectionPoint<<<dim_grid_bracelet, dim_block_bracelet, mInputPointsCount * 4>>>(mDeviceInputLines, 
		mInputPointsCount, 
		mDeviceIntersectionPoints,
		mDeviceMinBraceletPerIntersectionPoint,
		mDeviceMinBraceletMidPintPerIntersectionPoint);


	rc = cudaGetLastError();
	cudaEventRecord( sstop, 0 );
	cudaEventSynchronize( sstop );
	cudaEventElapsedTime( &time_bracelet_per_itersection_point, sstart, sstop );



    cudaDeviceSynchronize();
	QueryPerformanceCounter(&end);


	cudaEventRecord( min_start, 0 );


	findGlobalMinBracelet(&mLMSLineSlope, &mLMSMinBracelet, &mLMSLineIntercept);
	
	
   cudaEventRecord( min_stop, 0 );
   cudaEventSynchronize( min_stop );
   cudaEventElapsedTime( &time_minimum_bracelet, min_start, min_stop );



   //double time = (double)(end.QuadPart - start.QuadPart) / 1000.0;

   cout << "Time intersection points computation = " << time_intersection_points   << std::endl;
   cout << "Time bracelet per intersection points computation = " << time_bracelet_per_itersection_point   << std::endl;
   cout << "Time minimum bracelet computation = " << time_minimum_bracelet   << std::endl;
   cout << "Total time = " <<  (double)(end.QuadPart - start.QuadPart) / (freq.QuadPart / 1000.0)   << std::endl;

   cout << endl << "LMS line slope = " << mLMSLineSlope << " Intercept = " << mLMSLineIntercept << " Min bracelet = " << mLMSMinBracelet << endl;


  // dumpGIVFile(mLMSLineSlope, mLMSLineIntercept, mLMSMinBracelet);


	cudaEventDestroy( cuda_start );
	cudaEventDestroy( cuda_stop );

#if 0
	cudaMemcpy(mHostMinBraceletMidPintPerIntersectionPoint, mDeviceMinBraceletMidPintPerIntersectionPoint,mInputIntersectionPointCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(mHostMinBraceletPerIntersectionPoint, mDeviceMinBraceletPerIntersectionPoint,mInputIntersectionPointCount * sizeof(float), cudaMemcpyDeviceToHost);
	//for now skip the cuda reduction and copy to the host.
	//cudaMemcpy(mHostIntersectionPoints, mDeviceIntersectionPoints,2 * mInputIntersectionPointCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(mHostIntersectionPoints, mDeviceMinBraceletReductionOutput,MAX_POINT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
#endif	


}



/*******************************************************************************************************/



void LMS2D::dumpGIVFile()
{

	float x_coord = mLMSLineSlope;
	float y_coord = mLMSLineIntercept;
	float min_bracelet = mLMSMinBracelet;
//	static int index = 0;
	stringstream st;
	st << "dump_cuda_lms.giv";
	ofstream file(st.str());

	//gil
//	file << "$image "  << "noise.png" << endl;
	//First draw the input points
	file << "$noline" << endl <<
	"$marks circle" << endl;

	float min_x = FLT_MAX;
	float min_y = FLT_MAX;
	float max_x = FLT_MIN;
	float max_y = FLT_MIN;


	for (int i = 0; i < this->mInputPointsCount; i++){
		float x = this->mHostInputLines[2*i];
		float y = -this->mHostInputLines[2*i + 1];

		if ( x < min_x){
			min_x = x;
		}

		if ( y < min_y){
			min_y = y;
		}

		if ( x > max_x){
			max_x = x;
		}

		if (y > max_y){
			max_y = y;
		}

		file << x << " " << y << endl;
	}

	//draw the LMS line

	file << endl << endl;

	file << "$line LMS" << endl;
	file << "$color blue" << endl;

	float left_x = min_x;
	float left_y = left_x * x_coord + y_coord;

	float right_x = max_x;
	float right_y = right_x * x_coord + y_coord;

	file << left_x << " " << left_y << endl;
	file << right_x << " " << right_y << endl;

	//draw the slab


	file << endl << endl;

	file << "$line UpperSLAB" << endl;
	file << "$color yellow" << endl;

	left_y = left_x * x_coord + y_coord + min_bracelet / 2;
	right_y = right_x * x_coord + y_coord + min_bracelet / 2;

	file << left_x << " " << left_y << endl;
	file << right_x << " " << right_y << endl;

	file << endl << endl;

	file << "$line LowerSLAB" << endl;
	file << "$color yellow" << endl;

	left_y = left_x * x_coord + y_coord - min_bracelet / 2;
	right_y = right_x * x_coord + y_coord - min_bracelet / 2;

	file << left_x << " " << left_y << endl;
	file << right_x << " " << right_y << endl;


	//here we draw the results of the reference top sweep algorithm
#if 0

	float top_slope, top_intercept, top_slab_width;
	ifstream top_data("topological_sweep.out"); 

	top_data >> top_slope >> top_intercept >> top_slab_width;

	//draw the LMS line

	file << endl << endl;

	file << "$line LMS Top" << endl;
	file << "$color purple" << endl;

	left_x = min_x;
	left_y = left_x * top_slope + top_intercept;

	right_x = max_x;
	right_y = right_x * top_slope + top_intercept;

	file << left_x << " " << left_y << endl;
	file << right_x << " " << right_y << endl;

	//draw the slab


	file << endl << endl;

	file << "$line UpperSLAB Top" << endl;
	file << "$color orange" << endl;

	left_y = left_x * top_slope + top_intercept + top_slab_width / 2;
	right_y = right_x * top_slope + top_intercept + top_slab_width / 2;

	file << left_x << " " << left_y << endl;
	file << right_x << " " << right_y << endl;

	file << endl << endl;

	file << "$line LowerSLAB top" << endl;
	file << "$color orange" << endl;

	left_y = left_x * top_slope + top_intercept - top_slab_width / 2;
	right_y = right_x * top_slope + top_intercept - top_slab_width / 2;

	file << left_x << " " << left_y << endl;
	file << right_x << " " << right_y << endl;

	
#endif	
	
	//file << result.a << " " << result.b << " " << result.width << endl;

}




/*******************************************************************************************************/



void LMS2D::compute(float* input_points, int input_points_count, float* lms_line_slope, float* lms_line_intercept, float* min_bracelet)
{
	mInputPointsCount = input_points_count;

	for (int i = 0; i < mInputPointsCount; i++){
		mHostInputLines[2*i] = input_points[2*i]; //x -> m
		mHostInputLines[2*i + 1] = -input_points[2*i + 1]; //y -> -n
	}

		// We need two floats per line
	cudaError_t cudaStatus = cudaMemcpy(mDeviceInputLines, mHostInputLines, mInputPointsCount * 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!";
    }

	mInputIntersectionPointCount = mInputPointsCount * (mInputPointsCount - 1) / 2 ;

	this->compute();


	*lms_line_slope = mLMSLineSlope;
	*lms_line_intercept = mLMSLineIntercept;
	*min_bracelet = mLMSMinBracelet;
}



/*******************************************************************************************************/


void LMS2D::allocate()
{

	//allocate host memory

	mHostInputLines = new float[MAX_POINT_COUNT*2]; //2 floats per point

	// move input data to device

	
    cudaError_t cudaStatus = cudaMalloc((void**)&mDeviceInputLines, MAX_POINT_COUNT * 2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


	unsigned int max_intersection_point_count = MAX_POINT_COUNT * (MAX_POINT_COUNT - 1) / 2 ;


    cudaStatus = cudaMalloc((void**)&mDeviceIntersectionPoints , 2 * max_intersection_point_count * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }
	//gil
#if 0
	int total_y_coordinates_count = max_intersection_point_count * MAX_POINT_COUNT;

    cudaStatus = cudaMalloc((void**)&mDeviceYcoordinatesPerVerticalLine, total_y_coordinates_count * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }
#endif
	
    cudaStatus = cudaMalloc((void**)&mDeviceMinBraceletReductionOutput, MAX_POINT_COUNT * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


    cudaStatus = cudaMalloc((void**)&mDeviceMinBraceletPerIntersectionPoint, max_intersection_point_count * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }

    cudaStatus = cudaMalloc((void**)&mDeviceMinBraceletMidPintPerIntersectionPoint, max_intersection_point_count * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


	mHostMinBraceletMidPintPerIntersectionPoint = (float*)malloc(max_intersection_point_count * sizeof(float));

	mHostMinBraceletPerIntersectionPoint = (float*)malloc(max_intersection_point_count * sizeof(float));

	mHostIntersectionPoints = (float*)malloc(2 * max_intersection_point_count * sizeof(float));

	mHostMinBraceletReductionOutput = (float*)malloc(MAX_POINT_COUNT * sizeof(float));

}


/*******************************************************************************************************/

void LMS2D::readInputFromFile(const char* file_name)
{
	//1. Read input points from file

	ifstream input_point_file(file_name);
	
	//int points_count;

	input_point_file >> mInputPointsCount;

	int dim;

	input_point_file >> dim;

	//allocate host 

	mHostInputLines = new float[mInputPointsCount*2]; //2 floats per point
	for (int i = 0; i < mInputPointsCount; i++){
		input_point_file >> mHostInputLines[2*i]; //x -> m
		float temp;
		input_point_file >> temp;
		mHostInputLines[2*i + 1] = -temp; //y -> -n
	}

	// move input data to device

	
    cudaError_t cudaStatus = cudaMalloc((void**)&mDeviceInputLines, mInputPointsCount * 2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


	// We need two floats per line
	cudaStatus = cudaMemcpy(mDeviceInputLines, mHostInputLines, mInputPointsCount * 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!";
    }


	//Allocate data for intersection points and maximal y separation
	//We have n(n-1) / 2 instersection points and n intersections of every line with the vertical line going through each intersection point.
	//Total: n * (n - 1) / 2 intersection points. Each point hold 2 float (x and y). = n(n-1) floats.
	//in addition we need  n*n*(n-1)/2 floats. total = n(n-1) + n*n*(n-1) / 2


	mInputIntersectionPointCount = mInputPointsCount * (mInputPointsCount - 1) / 2 ;

    cudaStatus = cudaMalloc((void**)&mDeviceIntersectionPoints , 2 * mInputIntersectionPointCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


	//gil
#if 0
	int total_y_coordinates_count = mInputIntersectionPointCount * mInputPointsCount;
    cudaStatus = cudaMalloc((void**)&mDeviceYcoordinatesPerVerticalLine, total_y_coordinates_count * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }
#endif
	
    cudaStatus = cudaMalloc((void**)&mDeviceMinBraceletReductionOutput, MAX_POINT_COUNT * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


    cudaStatus = cudaMalloc((void**)&mDeviceMinBraceletPerIntersectionPoint, mInputIntersectionPointCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }

    cudaStatus = cudaMalloc((void**)&mDeviceMinBraceletMidPintPerIntersectionPoint, mInputIntersectionPointCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
    }


	mHostMinBraceletMidPintPerIntersectionPoint = (float*)malloc(mInputIntersectionPointCount * sizeof(float));

	mHostMinBraceletPerIntersectionPoint = (float*)malloc(mInputIntersectionPointCount * sizeof(float));

	mHostIntersectionPoints = (float*)malloc(2 * mInputIntersectionPointCount * sizeof(float));

	mHostMinBraceletReductionOutput = (float*)malloc(MAX_POINT_COUNT * sizeof(float));

}


