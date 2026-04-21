CXX = nvcc
CXXFLAGS = -O3

all: gelu_project

gelu_project: gelu.cu
	$(CXX) $(CXXFLAGS) gelu.cu -o gelu_project

clean:
	rm -f gelu_project
