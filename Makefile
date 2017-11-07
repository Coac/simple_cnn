default all:
	g++ -std=c++11 Example\ MNIST/example1.cpp -I . -o example1 -O3
	g++ -std=c++11 Example\ MNIST/example2.cpp -I . -o example2 -O3
