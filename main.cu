#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#define BLOCK_W 32
#define BLOCK_H 32
#define TESTY 25
#define TESTX 25
using namespace std;

void CPU_Med(unsigned short *d_in, unsigned short *d_out, int nx, int ny)
{

	for (int x = 0; x < nx; x++) 
	{
		for (int y = 0; y < ny; y++) 
		{
			int i = 0;
			unsigned short v[9] = { 0 };
			for (int xx = x - 1; xx <= x + 1; xx++) {
				for (int yy = y - 1; yy <= y + 1; yy++) {
					if (0 <= xx && xx < nx && 0 <= yy && yy < ny) //ตัดมุม
						v[i++] = d_in[yy*nx + xx];
				}
			}
			if (x == TESTX && y == TESTY)
			{
				for (int i = 0; i < 9; i++)
					printf("%d ", v[i]);
				printf("\n\n");
			}
			//B sort
			for (int i = 0; i < 9; i++) 
			{
				for (int j = i + 1; j < 9; j++) 
				{
					if (v[i] > v[j]) // swap(A,B)
					{
						unsigned short tmp = v[i];
						v[i] = v[j];
						v[j] = tmp;
					}
				}
			}
			if (x == TESTX && y == TESTY)
			{
				for (int i = 0; i < 9; i++)
					printf("%d ",v[i]);
				printf("\n\n");
			}
			d_out[y*nx + x] = v[4]; //4 mid
		}
	}
}
//GPU OLD
__global__ void GPU_MedianFilter(unsigned short *d_in , unsigned short *d_out , int nx , int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = 0;
	unsigned short v[9] = { 0 };
	for (int xx = x - 1; xx <= x + 1; xx++) 
	{
		for (int yy = y - 1; yy <= y + 1; yy++) 
		{
			if (0 <= xx && xx < nx && 0 <= yy && yy < ny) //ตัดมุม
				v[i++] = d_in[yy*nx + xx];
		}
	}
	//B Sort
	for (int i = 0; i < 9; i++) 
	{
		for (int j = i + 1; j < 9; j++) 
		{
			if (v[i] > v[j]) // swap(A,B)
			{ 
				unsigned short tmp = v[i];
				v[i] = v[j];
				v[j] = tmp;
			}
		}
	}
	d_out[y*nx + x] = v[4]; //4 mid
}

__global__ void Cuda_debug(unsigned short *Input_Image, unsigned short *Output_Image, int img_w, int img_h) {
	unsigned short ingpuArray[9];
	int count = 0;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	//check
	if ((x >= (img_w - 1)) || (y >= img_h - 1) || (x == 0) || (y == 0)) //กรอบ 0
		return;
	//IMG to ingpuArray
	for (int r = x - 1; r <= x + 1; r++) 
	{
		for (int c = y - 1; c <= y + 1; c++) 
		{
			ingpuArray[count++] = Input_Image[c*img_w + r];
		}
	}
	if (x == TESTX && y == TESTY)
	{
		for (int i = 0; i < 9; i++)
			printf("%d ", ingpuArray[i]);
		printf("\n\n");
	}
	//B Short
	for (int i = 0; i<9; ++i)
	{
		int minval = i;
		for (int l = i + 1; l<9; ++l)
			if (ingpuArray[l] < ingpuArray[minval])
				minval = l;
		//swap(a,b)
		unsigned short temp = ingpuArray[i];
		ingpuArray[i] = ingpuArray[minval];
		ingpuArray[minval] = temp;
	}
	if (x == TESTX && y == TESTY)
	{
		for (int i = 0; i < 9; i++)
			printf("%d ", ingpuArray[i]);
		printf("\n\n");
	}
	//printf("[%d %d] = %d\n",x,y, ingpuArray[4]);
	Output_Image[(y*img_w) + x] = ingpuArray[4]; // 4 mid
	if (x == TESTX && y == TESTY)
	{
		printf("1A %d %d\n", Output_Image[(y*img_w) + x], (y*img_w) + x);
	}
}

int Div0Up(int a, int b)//fix int/int=0
{
	return ((a % b) != 0) ? (1) : (a / b);
}

int main(int argc, const char** argv)
{
	int img_w = 1920; //IMG_Width
	int img_h = 1080; //IMG_Higth

	//Input File
	int dataLength = img_w*img_h; 
	unsigned short *Input_Image_Host = (unsigned short *)malloc(dataLength* sizeof(unsigned short));
	unsigned short *Output_Image_Host = (unsigned short *)malloc(dataLength* sizeof(unsigned short));
	FILE *op1;
	op1 = fopen("D:\\Datain.txt", "r");
	for (int i = 0; i < dataLength;i++)
		fscanf(op1, "%hu",&Input_Image_Host[i]);
	fclose(op1);

	//Cuda init
	unsigned short *Input_Image;
	cudaMalloc((void**)&Input_Image,dataLength * sizeof(unsigned short));
	unsigned short *Output_Image;
	cudaMalloc((void**)&Output_Image,dataLength * sizeof(unsigned short));
	cudaMemcpy(Input_Image, Input_Image_Host, 2*dataLength, cudaMemcpyHostToDevice);

	//Cuda main
	const dim3 grid(Div0Up(img_w, BLOCK_W), Div0Up(img_h, BLOCK_H));
	const dim3 block(BLOCK_W, BLOCK_H);
	GPU_MedianFilter << < grid, block >> >(Input_Image, Output_Image, img_w, img_h); //Cuda Median Filter
	cudaDeviceSynchronize();
	cudaMemcpy(Output_Image_Host, Output_Image, 2*dataLength, cudaMemcpyDeviceToHost);

	//CPU_Med(Input_Image_Host,Output_Image_Host,img_w, img_h); //CPU Median filter

	//Output file
	FILE *op2 = fopen("D:\\Dataout.txt", "w+");
	for (int i = 0; i < dataLength; i++)
	{
		if (i%img_w == 0) fprintf(op2, "\n");
		fprintf(op2, "%hu ", Output_Image_Host[i]);
	}
	fclose(op2);

	//Free Var
	cudaFree(Input_Image);
	free(Input_Image_Host);
	free(Output_Image_Host);
	return 0;
}