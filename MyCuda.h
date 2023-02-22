/// Defined in MyCuda.h: Gets the last CUDA error message and prints it out, along with file and line #.
void myCudaErrorCheck(const char *file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line-1);
		exit(0);
	}
}

/// Defined in MyCuda.h: Returns the number of blocks needed given the number of threads per block and an N value. 
int getNumberBlocks(int numberOfThreads, int n)
{
	return (n - 1)/numberOfThreads + 1;
}