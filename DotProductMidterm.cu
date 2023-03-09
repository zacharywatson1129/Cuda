// nvcc DotProductMidterm.cu -o temp
/*
	In this program, we have 4 blocks, and 10 threads/block. n = 1001
	We're calculating dot product. Passing a, b, single float dot, and n.
	We wanted to make this run using shared memeory.
*/

#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

//Length of vectors to be added.
#define N 1001
#define BLOCK_SIZE 10

//Function prototypes
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
__global__ void DotProductGPU(float *, float *, float *, int );

//Globals
float *A_CPU, *B_CPU; //CPU pointers
float *A_GPU, *B_GPU, *Dot; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = 4;
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&Dot,sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = 1.0;	
		B_CPU[i] = 1.0;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU);
	myCudaErrorCheck(__FILE__, __LINE__);
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void DotProductGPU(float *a, float *b, float *dot, int n)
{
	
	__shared__ float c_sh[BLOCK_SIZE];
	
	/* 
		Make sure memory is clear.
		If garbage values are present, which they sometimes are
		in new allocated memory, the answer won't be correct. 
	*/
	for (int i = 0; i < BLOCK_SIZE; i++)
	{
		c_sh[i] = 0.0f;
	}
	
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	// In our case, jump 40 ahead.
	int stepSize = gridDim.x * blockDim.x;
	int c_counter = threadIdx.x;
	for (int i = id; i < n; i+=stepSize)
	{
		if (i < n)
		{
			c_sh[c_counter] = a[i] * b[i];
		}
		c_counter++;
	}

	/*
		At this point, we are finished calculating the 
	*/
	__syncthreads();

	for (int i = id; i < n; i+=stepSize)
	{
		for (int i = 0; i < n; i++)
		{
			
		}
	}

    // In previous code, we started out with a fold the size of the vector.
	// Because we are only operating within our block, we set fold equal to
	// the block size.
	// However, if we are on the last block, make sure we don't do like normal.
	// There may be extra threads that do not need to do any work.
	int fold = blockDim.x;
	if (blockIdx.x == blockDim.x - 1)
	{
		// In this case, when we get to thread with threadNumber 65000 or higher, 
		// we only set fold to max value. In this case, for threadNumber 65000, we
		// jump 349 higher, which is very last index, and thats our fold.
		fold = threadIdx.x - 1; // remember threadnumber = threadidx.x
	}
	else {
    	fold = blockDim.x;
	}
    while (fold / 2 >= 1) // If fold == 2, we run one last time.
    {
        // If fold / 2 gives us an odd number, add the last element to the first element,
        // so that we have an even number of indices.
        if (fold % 2 != 0)
        {
            if (threadIdx.x == 0)
            {
				// If we get inside here, we are inside the first thread of some block,
				// but we still must access using our global id which is vectorNumber.
                c_sh[id] += c_sh[id + fold - 1];
            }
            fold -= 1;
        }
		__syncthreads();
        // So we don't add with indices that we don't need.
        if (threadIdx.x < fold / 2)
        {
            c_sh[id] += c_sh[id + fold/2];
        }
		fold /= 2;
        __syncthreads();   
    }

}

int main()
{
	float dotProduct; //, temp;
	timeval start, end;
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	//Starting the timer
	gettimeofday(&start, NULL);

	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	//Calling the Kernel (GPU) function.	
	DotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, Dot, N);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpy(&dotProduct, Dot, sizeof(float), cudaMemcpyDeviceToDevice);

	//Copy Memory from GPU to CPU	
	/*dotProduct = 0.0;
	printf("\n grid size = %d\n", GridSize.x);
	for(int k = 0; k < GridSize.x; k++)
	{
		cudaMemcpyAsync(&temp, &C_GPU[k*BlockSize.x], sizeof(float), cudaMemcpyDeviceToHost);
		printf("temp = %f\n", temp);
		myCudaErrorCheck(__FILE__, __LINE__);
		dotProduct += temp;
	}*/

	

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	//Displaying the dot product.
	printf("Dot product = %.15f\n", Dot);
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
