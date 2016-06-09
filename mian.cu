#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <time.h>
#define BLOCK_W 8
#define BLOCK_H 128
#define TESTY 30
#define TESTX 30
using namespace std;

void loadData(const char* filename, int size, double* readArray) {
	ifstream file(filename, ios::binary | ios::in);
	if (!file.is_open())
	{
		cout << "Cannot open file." << endl;
		return;
	}
	for (int i = 0; i<size && !file.eof(); i++)
	{
		try
		{
			double aDouble = 0;
			file.read((char*)(&aDouble), sizeof(double));
			readArray[i] = aDouble;
			//printf("%lf\n", aDouble);
		}
		catch (int e)
		{
			cout << "An exception occurred. Exception Nr. " << e << endl;
			break;
		}
	}

	file.close();
}

void writeFile(const char *filename, const int size, double* readArray)
{
	ofstream output(filename, std::ios::binary | std::ios::out);
	for (int i = 0; i < size; i++)
	{
		output.write((char *)&readArray[i], sizeof(double));
	}
	output.close();
}

void CPU_Med(float *d_in, float *d_out, int nx, int ny)
{

	for (int x = 0; x < nx; x++)
	{
		for (int y = 0; y < ny; y++)
		{

			if ((x >= (nx - 1)) || (y >= ny - 1) || (x == 0) || (y == 0)) //กรอบ 0
				return;

			int i = 0;
			double v[9] = { 0 };
			for (int xx = x - 1; xx <= x + 1; xx++)
			{
				for (int yy = y - 1; yy <= y + 1; yy++)
				{
					if (0 <= xx && xx < nx && 0 <= yy && yy < ny) //ตัดมุม
						v[i++] = d_in[yy*nx + xx];
				}
			}
			//B sort
			for (int i = 0; i < 5; i++)
			{
				for (int j = i + 1; j < 9; j++)
				{
					// swap(A,B)
					if (v[i] > v[j])
					{
						double tmp = v[i];
						v[i] = v[j];
						v[j] = tmp;
					}
				}
			}
			/*if (x == TESTX && y == TESTY)
			{
			for (int i = 0; i < 9; i++)
			printf("%d ", v[i]);
			printf("\n\n");
			}*/
			d_out[y*nx + x] = v[4]; //4 mid
		}
	}
}

__global__ void Cuda_debug(double *Input_Image, double *Output_Image, int img_h, int img_w) {
	double ingpuArray[9];
	int count = 0;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	//check
	if ((x >= (img_h - 1)) || (y >= img_w - 1) || (x == 0) || (y == 0)) //กรอบ 0
		return;
	//IMG to ingpuArray
	for (int r = x - 1; r <= x + 1; r++)
	{
		for (int c = y - 1; c <= y + 1; c++)
		{
			ingpuArray[count++] = Input_Image[c*img_h + r];
		}
	}
	//B Short
	for (int i = 0; i<5; ++i)
	{
		int min = i;
		for (int l = i + 1; l<9; ++l)
			if (ingpuArray[l] < ingpuArray[min])
				min = l;
		//swap(a,b)
		double temp = ingpuArray[i];
		ingpuArray[i] = ingpuArray[min];
		ingpuArray[min] = temp;
	}
	//printf("[%d %d] = %d\n",x,y, ingpuArray[4]);
	Output_Image[(y*img_h) + x] = ingpuArray[4]; // 4 mid
	/*if (x == TESTX && y == TESTY)
	{
	printf("1A %d %d", ingpuArray[(y*img_w) + x], (y*img_h) + x);
	}*/
}

int Div0Up(int a, int b)//fix int/int=0
{
	return ((a % b) != 0) ? (1) : (a / b);
}

int main(int argc, const char** argv)
{
	int img_h = 8192; //IMG_Higth
	int img_w = 81; //IMG_Width
	//Input File
	int dataLength = img_w*img_h;
	double *Input_Image_Host = new double[dataLength];
	double *Output_Image_Host = new double[dataLength];

	loadData("D://env_douuble_bin.dat", dataLength, Input_Image_Host);
	//Cuda init
	double *Input_Image;
	cudaMalloc((void**)&Input_Image, dataLength * sizeof(double));
	double *Output_Image;
	cudaMalloc((void**)&Output_Image, dataLength * sizeof(double));
	cudaMemcpy(Input_Image, Input_Image_Host, sizeof(double) * dataLength, cudaMemcpyHostToDevice);

	//Cuda main
	const dim3 blk(Div0Up(img_h, BLOCK_W), Div0Up(img_w, BLOCK_H));
	const dim3 tid(BLOCK_W, BLOCK_H);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start);

	Cuda_debug << < blk, tid >> >(Input_Image, Output_Image, img_h, img_w); //Cuda Median Filter
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time " << milliseconds << " ms\n";

	cudaMemcpy(Output_Image_Host, Output_Image, sizeof(double) * dataLength, cudaMemcpyDeviceToHost);

	//CPU_Med(Input_Image_Host,Output_Image_Host,img_w, img_h); //CPU Median filter

	//TEST print
	/*printf("\n\n");
	for (int i = 0; i < dataLength; i++)
	{
	if (i%img_w == 0) printf("\n");
	printf("%hu ", Output_Image_Host[i]);
	}*/

	//Output file
	writeFile("d://env_douuble_output.dat", dataLength, Output_Image_Host);

	//Free Var
	cudaFree(Input_Image);
	cudaFree(Output_Image);
	delete Input_Image_Host;
	delete Output_Image_Host;
	return 0;
}