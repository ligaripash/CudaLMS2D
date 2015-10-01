

#include "LMS2D.h"
#include <windows.h>

using namespace std;





int main(int argc, char* argv[])
{




	SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_TIME_CRITICAL);

	//0. init CUDA

	const int MAX_POINT_COUNT = 512; // this must be a power of two value.

	LMS2D lms2D(MAX_POINT_COUNT);

	// the input file contains a list of 2D points. One point per line
	// currently only. curr
	lms2D.readInputFromFile(argv[1]);

	// compute 
	lms2D.compute();

	// dump a giv file with lms slab and line. a giv file is an easy to use, open source utility for geomtry primitives representation.
	// you can download it here http://giv.sourceforge.net/giv/
	lms2D.dumpGIVFile();
//	lms2D.writeOutput();

	
}

