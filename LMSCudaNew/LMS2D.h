

//#include "device_launch_parameters.h"

class LMS2D
{
public:
	LMS2D();

	~LMS2D();

	// input - input_points - array of float points in the format :x0 y0 x1 y1 .... xn yn
	// input - input_points_count - number of points in the point input_points array
	// output - lms_line_slope - the slope of the LMS line
	// output - lms_line_intercept - the intercept of the LMS line with y axis
	// output - min_bracelet - the LMS itself 

	void compute(float* input_points, int input_points_count, float* lms_line_slope, float* lms_line_intercept, float* min_bracelet);

	void compute();

	// allocate resources for the computation (host and device memory). Call once before compute(...)
	void allocate();


	void dumpGIVFile();

private:


	void findGlobalMinBracelet(float* x_coord, float* min_bracelet, float* y_coord);




	void computeDualLines();

	void initCUDA();

	float* mDeviceInputLines;

	float* mHostInputLines;

	//float* mDeviceInteresectionPointsAndYSeparation;

	//float* mDeviceYcoordinatesPerVerticalLine;

	float* mDeviceIntersectionPoints;

	float* mDeviceMinBraceletPerIntersectionPoint;

	float* mDeviceMinBraceletReductionOutput;

	float* mDeviceMinBraceletMidPintPerIntersectionPoint;

	float* mHostMinBraceletReductionOutput;


	// Just for debug
	float* mHostMinBraceletMidPintPerIntersectionPoint;

	float* mHostMinBraceletPerIntersectionPoint;

	float* mHostIntersectionPoints;

	int mInputPointsCount;

	int mInputIntersectionPointCount;

	float mLMSLineSlope;
	float mLMSLineIntercept;
	float mLMSMinBracelet;

	//int mMaxPointCount;

//	__global__ void computeDualLinesKernel(float* input_points, float* output_lines);

};