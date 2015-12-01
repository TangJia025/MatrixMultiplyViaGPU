#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

const int threadsPerBlock_x = 16;
//naive
__global__ void MatrixMulKernel1(float *dev_a, float *dev_b, float *dev_c, int n)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
    
	float temp = 0;
	int k;
	for (k = 0; k < n; k++)
	{
		temp += dev_a[y * n + k] * dev_b[k * n + x];
	}
	dev_c[y * n + x] = temp; 
}         
//tiling
__global__ void MatrixMulKernel2(float *dev_a, float *dev_b, float *dev_c, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = 0;
	for (int i = 0; i < gridDim.x; i++)
	{
		__shared__ float row_a[threadsPerBlock_x][threadsPerBlock_x];
		__shared__ float col_b[threadsPerBlock_x][threadsPerBlock_x];

		row_a[ty][tx] = *(dev_a + (ty + by * blockDim.y) * n + tx + i * blockDim.x);
        col_b[ty][tx] = *(dev_b + (ty + i * blockDim.y) * n + tx + bx * blockDim.x);
        __syncthreads();
		for (int j = 0; j < blockDim.x; j++)
			Pvalue += row_a[ty][j] * col_b[j][tx];
		__syncthreads();
	}
	dev_c[(ty + by * blockDim.y) * n + tx + bx * blockDim.x] = Pvalue;   
}    
//matrix transpose with tiling
 __global__ void MatrixMulKernel3(float *dev_a, float *dev_b, float *dev_c, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

    __shared__ float A_tile[threadsPerBlock_x][threadsPerBlock_x];
	__shared__ float B_tile[threadsPerBlock_x][threadsPerBlock_x];
	float Pvalue = 0;
	for (int i = 0; i < gridDim.x; i++)
	{
		int i = ty + by * blockDim.y;
		int j = tx + i * blockDim.x;
		A_tile[ty][tx] = *(dev_a + i * n + j);
        B_tile[tx][ty] = *(dev_b + i * n + j);
        __syncthreads();
		for (int j = 0; j < blockDim.x; j++)
			Pvalue += A_tile[ty][j] * B_tile[tx][j];
		__syncthreads();
	}
	dev_c[(ty + by * blockDim.y) * n + tx + bx * blockDim.x] = Pvalue;   
}  
//transpose without tiling
__global__ void MatrixMulKernel31(float *dev_a, float *dev_b, float *dev_c, int n)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
    
	float temp = 0;
	int k;
	for (k = 0; k < n; k++)
	{
		temp += dev_a[y * n + k] * dev_b[x * n + k];
	}
	dev_c[y * n + x] = temp; 
}         
 
//loop unrolling
__global__ void MatrixMulKernel4(float* Md, float* Nd, float* Pd, int Width)
{
		int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		float Pvalue = 0;
				
		for (int m = 0; m < gridDim.x; ++m) {
			__shared__ float Mds[threadsPerBlock_x][threadsPerBlock_x];
			__shared__ float Nds[threadsPerBlock_x][threadsPerBlock_x];
			Mds[ty][tx] = *(Md + (by*blockDim.y + ty) * Width + m*blockDim.x + tx);
			Nds[ty][tx] = *(Nd + (m*blockDim.y + ty) * Width + bx*blockDim.x + tx);
			__syncthreads();
			Pvalue += Mds[ty][0] * Nds[0][tx] + Mds[ty][1] * Nds[1][tx] +
			Mds[ty][2] * Nds[2][tx] + Mds[ty][3] * Nds[3][tx] +
			Mds[ty][4] * Nds[4][tx] + Mds[ty][5] * Nds[5][tx] +
			Mds[ty][6] * Nds[6][tx] + Mds[ty][7] * Nds[7][tx] +
			Mds[ty][8] * Nds[8][tx] + Mds[ty][9] * Nds[9][tx] +
			Mds[ty][10] * Nds[10][tx] + Mds[ty][11] * Nds[11][tx] +
			Mds[ty][12] * Nds[12][tx] + Mds[ty][13] * Nds[13][tx] +
			Mds[ty][14] * Nds[14][tx] + Mds[ty][15] * Nds[15][tx];
			__syncthreads();
		}
		Pd[(by*blockDim.y+ty)*Width+bx*blockDim.x+tx] = Pvalue;
}

int main(int argc, char *argv[])
{
	long int N = 0;
	if (argc == 1)
	{
		N = 1024;
	}
	else 
	{
		N = atoi(argv[1]);
    }
    long int blocksPerGrid_x = N / threadsPerBlock_x;


    float *a, *b, *c;
	float *d_a, *d_b, *d_c;
    FILE *file1, *file2, *file3;
   
    a = (float *)malloc(sizeof(float) * N * N);
    b = (float *)malloc(sizeof(float) * N * N);
    c = (float *)malloc(sizeof(float) * N * N);
    
	srand((unsigned)time(NULL));
    if ((file1 = fopen("/home/tangjia/C++Parallel/MatrixMul/A_Matri.txt", "wt")) == NULL)
	{
        printf("Here is a mistake\n");
		return 0;
	}     
		
    file2 = fopen("/home/tangjia/C++Parallel/MatrixMul/B_Matri.txt", "wt");

	for (int i = 0; i < N*N; i++)
	{
		a[i] = rand()/(float)RAND_MAX;
		b[i] = rand()/(float)RAND_MAX;
//        printf("%f  %f\n",a[i], b[i]);
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			fprintf(file1, "%-8f  ", a[i*N+j]);
            fprintf(file2, "%-8f  ", b[i*N+j]);
		}
		fprintf(file1, "\n");
		fprintf(file2, "\n");
	}
	
    cudaMalloc((void **)&d_a, N * N * sizeof(float));
	cudaMalloc((void **)&d_b, N * N * sizeof(float));
	cudaMalloc((void **)&d_c, N * N * sizeof(float));

    cudaMemcpy(d_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(threadsPerBlock_x, threadsPerBlock_x);
	dim3 dimGrid(blocksPerGrid_x, blocksPerGrid_x);
   
    struct timeval start, end;
	float time_use, speed;
    gettimeofday(&start, NULL);
	cudaThreadSynchronize();
	MatrixMulKernel3<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
	cudaThreadSynchronize();
    gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    speed = N*N*N/time_use/1000;
    printf("N: %ld, Time consumed: %.2f us, Speed: %.2f Gflops\n",N, time_use, speed);
	  
	cudaMemcpy(c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
   
    printf("stage 5\n"); 
  
    file3 = fopen("/home/tangjia/C++Parallel/MatrixMul/C_Matri.txt", "wt");
    for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			fprintf(file3, "%-8.2f", c[i*N+j]);
		}
		fprintf(file3, "\n");
	}    

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(a);
	free(b);
	free(c);

	return 1;
}

