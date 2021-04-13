#include "calculator.h"
#include <cuda_runtime.h>
#include "iostream"
 
bool InitCUDA()
{
	//used to count the device numbers
	int count; 
 
	// get the cuda device count
	cudaGetDeviceCount(&count);
	// print("%d\n", count);
	std::cout << "count: " <<count << std::endl;
	if (count == 0) 
	{
		return false;
	}
 
	// find the device >= 1.X
	int i;
	for (i = 0; i < count; ++i) 
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
		{
			if (prop.major >= 1) 
			{
				break;
			}
		}
	}
	// if can't find the device
	if (i == count) 
	{
		std::cout<<"count: "<<count<<" i:"<<i<<std::endl;
		return false;
	}
 
	// set cuda device
	cudaSetDevice(i);
 
	return true;
}
 
__global__ void add(const float *dev_a,const float *dev_b,float *dev_c)
{
    int i=threadIdx.x;
    float sub = dev_a[i]-dev_b[i];
    dev_c[i] = sub * sub;
}
 
bool calculate_distance_imp(const float* v1, const float* v2, const int length, float* sub)
{
	float *dev_v1, *dev_v2, *result;
	cudaError_t err = cudaSuccess;
	err=cudaMalloc((void **)&dev_v1, sizeof(float)*length);
	err=cudaMalloc((void **)&dev_v2, sizeof(float)*length);
	err=cudaMalloc((void **)&result, sizeof(float)*length);
	if(err!=cudaSuccess)
	{
	 std::cout<<"the cudaMalloc on GPU is failed"<<std::endl;
	 return false;
	}
	cudaMemcpy(dev_v1,v1,sizeof(float)*length,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2,v2,sizeof(float)*length,cudaMemcpyHostToDevice);
	add<<<1,length>>>(dev_v1,dev_v2,result);
	cudaMemcpy(sub,result,sizeof(float)*length,cudaMemcpyDeviceToHost);
	std::cout<<"("<<sub[0]<<","<<sub[1]<<")"<<std::endl;
	return true;
}
————————————————
版权声明：本文为CSDN博主「蒙特卡洛家的树」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/u011021773/article/details/83792418