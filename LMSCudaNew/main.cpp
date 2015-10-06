

#include "LMS2D.h"
#include <windows.h>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

/********************************************************************************************************************************/

void readPointListFromFile(char* file_name, float*& points_list, int* point_count )
{
	ifstream input_point_file(file_name);
	
	//int points_count;
	
	input_point_file >> *point_count;

	int dim;

	input_point_file >> dim;

	//allocate host 

	points_list = new float[*point_count*2]; //2 floats per point
	for (int i = 0; i < *point_count; i++){
		input_point_file >> points_list[2*i]; //x -> m
		input_point_file >> points_list[2*i + 1];
	}

	// move input data to device

}



/********************************************************************************************************************************/


// dump a giv file with lms slab and line. a giv file is an easy to use, open source utility for geomtry primitives representation.
// you can download it here http://giv.sourceforge.net/giv/

void dumpOutputGIV(float lms_slope, float lms_intercept, float lms, float* point_list, int point_count)
{
	float x_coord = lms_slope;
	float y_coord = lms_intercept;
	float min_bracelet = lms;
	stringstream st;
	st << "dump_cuda_lms.giv";
	ofstream file(st.str());

	file << "$noline" << endl <<
	"$marks circle" << endl;

	float min_x = FLT_MAX;
	float min_y = FLT_MAX;
	float max_x = FLT_MIN;
	float max_y = FLT_MIN;


	for (int i = 0; i < point_count; i++){
		float x = point_list[2*i];
		float y = point_list[2*i + 1];

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


}



/********************************************************************************************************************************/

int main(int argc, char* argv[])
{


//	SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_TIME_CRITICAL);

	//0. init CUDA


	LMS2D lms2D;

	// the input file contains a list of 2D points. One point per line
	// currently only. curr
	float* points_list;
	int points_count;

	// allocate data - done once
	lms2D.allocate();

	// read point list from file
	readPointListFromFile(argv[1], points_list, &points_count );


	// compute 
	float lms_slope, lms_intercept, lms;
	//possibly many invocation of compute
#define PERF
#ifdef PERF
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;

	QueryPerformanceFrequency(&Frequency); 
	QueryPerformanceCounter(&StartingTime);
#endif

	lms2D.compute(points_list, points_count, &lms_slope, &lms_intercept, &lms); 
#ifdef PERF
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	
	std::cout << "computation time = " << ElapsedMicroseconds.QuadPart << endl;
#endif


	std::cout << endl << "LMS line slope = " << lms_slope << " Intercept = " << lms_intercept << " Min bracelet = " << lms << endl;

	// dump a giv file with lms slab and line. a giv file is an easy to use, open source utility for geomtry primitives representation.
	// you can download it here http://giv.sourceforge.net/giv/
	dumpOutputGIV(lms_slope, lms_intercept, lms, points_list, points_count);

	delete [] points_list;

	
}

