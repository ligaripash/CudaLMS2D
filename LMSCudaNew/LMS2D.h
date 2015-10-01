

//#include "device_launch_parameters.h"

class LMS2D
{
public:
	LMS2D(int max_point_count);

	~LMS2D();

	void readInputFromFile(const char* file_name);

	void compute(float* input_points, int input_points_count, float* lms_line_slope, float* lms_line_intercept, float* min_bracelet);

	void compute();

	void allocate();

	// dump a giv file with lms slab and line. a giv file is an easy to use, open source utility for geomtry primitives representation.
	// you can download it here http://giv.sourceforge.net/giv/

	void dumpGIVFile();

private:


	void findGlobalMinBracelet(float* x_coord, float* min_bracelet, float* y_coord);




	void computeDualLines();

	void initCUDA();

	float* mDeviceInputLines;

	float* mHostInputLines;

	//float* mDeviceInteresectionPointsAndYSeparation;

	float* mDeviceYcoordinatesPerVerticalLine;

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

	int mMaxPointCount;

//	__global__ void computeDualLinesKernel(float* input_points, float* output_lines);

};