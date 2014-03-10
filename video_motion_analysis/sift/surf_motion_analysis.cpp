#include <iostream>

using namespace std;

int main ( int argc, char *argv[] ){
	if( argc != 2 ){
		cout << "Program requires a single argument," << endl <<
			"which is the location of a video file." << endl << endl;
		cout << "Press any key to close" << endl;
		getchar();
		exit(-1);
	}

}