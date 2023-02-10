//nvcc SimpleJuliaSetGPU.cu -o SimpleJuliaSetGPU -lglut -lGL -lm
// This is a simple Julia set which is repeated iterations of 
// Znew = Zold + C whre Z and Care imaginary numbers.
// After so many tries if Zinitial escapes color it black if it stays around color it red.

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define A  -0.824  //real
#define B  -0.1711   //imaginary

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

float *pixelsGPU;
float *pixels;

dim3 BlockSize;
dim3 GridSize;

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&pixelsGPU, window_width*window_height*3*sizeof(float));
	//Allocate Host (CPU) Memory
	pixels = (float*)malloc(window_width*window_height*3*sizeof(float));
}

void SetUpCudaDevices()
{
	// 1024 apartment buildings, a each building is like a row.
	GridSize.x = 1024;
	GridSize.y = 1;
	GridSize.z = 1;
	// 1024 rooms per apartment, each room is like an individual pixel in a row.
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	cudaFree(pixelsGPU);
	free(pixels);
}

void errorCheck(const char *file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

__device__ float color(float x, float y) 
{
	float mag,maxMag,temp;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		// Zn = Zo*Zo + C
		// or xn + yni = (xo + yoi)*(xo + yoi) + A + Bi
		// xn = xo*xo - yo*yo + A (real Part) and yn = 2*xo*yo + B (imagenary part)
		temp = x; // We will be changing the x but weneed its old value to hind y.	
		x = x*x - y*y + A;
		y = (2.0 * temp * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

__global__ void doWork(float *myPixels, float x_min, float y_min, float stepX, float stepY)
{
	int id = (threadIdx.x + blockIdx.x * blockDim.x) * 3; // *3 is to account for pixels.
	float x = x_min + threadIdx.x*stepX;
	float y = y_min + blockIdx.x*stepY;

	myPixels[id] = color(x, y);	// Red on or off returned from color
	myPixels[id+1] = 0.0; 	    // Green off
	myPixels[id+2] = 0.0;	    // Blue  off
}

void display(void)
{ 
	doWork<<<GridSize, BlockSize>>>(pixelsGPU, xMin, yMin, stepSizeX, stepSizeY);
	errorCheck(__FILE__, __LINE__);
	printf("Calculated the pixels, now copying from GPU to CPU.\n");
	cudaMemcpyAsync(pixels, pixelsGPU, window_width*window_height*3*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck(__FILE__, __LINE__);
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

int main(int argc, char** argv)
{
	SetUpCudaDevices();
	AllocateMemory();
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
	CleanUp();
}
