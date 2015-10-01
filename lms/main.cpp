
#include <fstream>
#include <stdlib.h>

using namespace std;

int main(int argc, char* argv[])
{
	ofstream file("c:\\temp\\dat.txt");

	int count = atoi(argv[1]);

	file << count << " " << "2" << endl;

	for (int i = 0; i < count; i++){
		int x = rand() % 1000;
		int y = rand() % 1000;

		file << x << " " << y << endl;
	}

	
}