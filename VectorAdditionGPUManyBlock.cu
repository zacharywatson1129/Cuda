// Vector addition on the GPU with as many blocks as needed.
// nvcc VectorAdditionGPUManyBlock.cu -o VectorAdditionGPUManyBlock

#include <sys/time.h>
#include <stdio.h>

//Length of vectors to be added.
#define N 30

//Globals
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = 4;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	/* Grid size actually defines how many blocks we have. Each block is 4
	   threads, and our vector length N = 30 / 4 threads to get the number of blocks
	   is not a whole number, so we round up, and we need at least 8 blocks. That means
	   we have 32 threads, but as we saw last time, that's actually an easy problem to
	   solve--just add an if statement.
	*/
	GridSize.x = 8;
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)2*i;	
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void AdditionGPU(float *a, float *b, float *c, int n)
{
	/*Our check kind of fails because we don't have a thread id bigger than 3,
	  yet we need to access indices 4, 5, ..., up to 29. Let's calculate a way to
	  use the block id and thread id both to calculate its 'id' as if the threads
	  were all part of block 0. So for example, block 1, thread 0, is actually
	  thread 4, and block 2, thread 2, is actually thread 10, in our scheme.
	  Now, this isn't the actual id of the thread, but it doesn't matter 
	  what we call it. In a more generic case, we could replace 4 with the thread
	  per block count. Same type of concept used as how pointer arithmetic works or
	  how array indices work. */
	int id = threadIdx.x + (4 * blockIdx.x);
	
	// Now our normal check will work.
	if(id < N)
	{
		c[id] = a[id] + b[id];
	}
}

int main()
{
	int i;
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
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	//Calling the Kernel (GPU) function.
	// When you call a Cuda function (declared with __global__), you specify the gridsize first,
	// then the blocksize, and those are always of type dim3.
	AdditionGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

	// Displaying the vector. You will want to comment this out when the vector gets big.
	// This is just to make sure everything is running correctly.	
	for(i = 0; i < N; i++)		
	{		
		printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}

	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, A_CPU[N-1], N-1, B_CPU[N-1], N-1, C_CPU[N-1]);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));
	
	//You're done so cleanup your mess.
	CleanUp();
	
	return(0);
}