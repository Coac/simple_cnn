default all:
	g++ -std=c++11 "examples/simple.cpp" -I . -o simple -O3
	g++ -std=c++11 "examples/alexnet.cpp" -I . -o alexnet -O3
