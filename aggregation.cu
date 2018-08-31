#include <stdio.h>
#include <stdlib.h>
#include <SGM.h>
#include <unistd.h>
#include <cuda_profiler_api.h>


__global__
void computeAggKernel_x(float *costDevice, float *dirAccCUDA, const int HEIGHT,const int WIDTH, const int DEPTH, const int block){
 

	__shared__ float prior[DISP_RANGE];
	__shared__ float minVal;
	int tz = threadIdx.x;
	int by = blockIdx.x;
	int xstart;
	int xend;
	if(block == 1)
	{
		xstart = 1; 
		xend = WIDTH/2;
		prior[tz] = costDevice[by*WIDTH*DEPTH  + (xstart - 1)*DEPTH + tz];

	}
	if(block == 3)
	{
		xstart = 1; 
		xend = WIDTH/2; 
		by = by + HEIGHT/2;
		prior[tz] = costDevice[by*WIDTH*DEPTH  + (xstart - 1)*DEPTH + tz];
	}
	if(block == 2)
	{
		xstart = WIDTH/2;
 		xend = WIDTH;
		prior[tz] = dirAccCUDA[by*WIDTH*DEPTH  + (xstart - 1)*DEPTH + tz];
	}
	if(block == 4)
	{
		xstart = WIDTH/2; 
		xend = WIDTH; 
		by = by + HEIGHT/2;
		prior[tz] = dirAccCUDA[by*WIDTH*DEPTH + (xstart - 1)*DEPTH + tz];
	}


	float e_smooth;
	float check;

__syncthreads();

	for(int x = xstart; x < xend; x++){ //changed here
	
		e_smooth = prior[tz];

		for(int d = 0; d < DEPTH; d=d+8){
			
			//check= (fabsf(tz - d) == 1)? 400:6000;
			//e_smooth = fminf(e_smooth, check + prior[d]);

			check= (fabsf(tz - d) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d]);
			check= (fabsf(tz - (d+1)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+1]);
			check= (fabsf(tz - (d+2)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+2]);
			check= (fabsf(tz - (d+3)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+3]);
			check= (fabsf(tz - (d+4)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+4]);
			check= (fabsf(tz - (d+5)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+5]);
			check= (fabsf(tz - (d+6)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+6]);
			check= (fabsf(tz - (d+7)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+7]);
		}


__syncthreads();		
			if(tz<64)
			prior[tz] = fminf(prior[tz], prior[tz+64]);
__syncthreads();
			if(tz<32)
			prior[tz] = fminf(prior[tz], prior[tz+32]);
__syncthreads();
			if(tz<16) 
			prior[tz] = fminf(prior[tz], prior[tz+16]);
__syncthreads();
			if(tz<8) 
			prior[tz] = fminf(prior[tz], prior[tz+8]);
__syncthreads();
			if(tz<4) 
			prior[tz] = fminf(prior[tz], prior[tz+4]);
__syncthreads();
			if(tz<2) 
			prior[tz] = fminf(prior[tz], prior[tz+2]);
__syncthreads();
			if(tz<1) 
			prior[tz] = fminf(prior[tz], prior[tz+1]);

			

			if(tz>127)
			prior[tz] = fminf(prior[tz], prior[tz-32]);
__syncthreads();
			if(tz>143)
			prior[tz] = fminf(prior[tz], prior[tz-16]);
__syncthreads();
			if(tz>151)
			prior[tz] = fminf(prior[tz], prior[tz-8]);
__syncthreads();
			if(tz>155)
			prior[tz] = fminf(prior[tz], prior[tz-4]);
__syncthreads();
			if(tz>157)
			prior[tz] = fminf(prior[tz], prior[tz-2]);
__syncthreads();
			if(tz>158)
			prior[tz] = fminf(prior[tz], prior[tz-1]);
		

			check = costDevice[by*WIDTH*DEPTH  + x*DEPTH + tz];
		
//__syncthreads();
		minVal=fminf(prior[0], prior[159]);
__syncthreads();

		dirAccCUDA[ by*WIDTH*DEPTH + x*DEPTH + tz] = check + e_smooth - minVal;
		prior[tz] = check + e_smooth - minVal;
		
__syncthreads();		
	}


}


__global__
void computeAggKernel_y(float *costDevice, float *dirAccCUDA, const int HEIGHT, const int WIDTH, const int DEPTH, const int block){
 

	__shared__ float prior[DISP_RANGE];
	__shared__ float minVal;
	int tz = threadIdx.x;
	int bx = blockIdx.x;
	int ystart;
	int yend;
	if(block == 1)
	{
		ystart = 1; 
		yend = HEIGHT/2;
		prior[tz] = costDevice[(ystart - 1)*WIDTH*DEPTH + bx*DEPTH + tz];

	}
	if(block == 3)
	{
		ystart = HEIGHT/2; 
		yend = HEIGHT; 
		prior[tz] = dirAccCUDA[(ystart - 1)*WIDTH*DEPTH + bx*DEPTH + tz];
	}
	if(block == 2)
	{
		ystart = 1; 
		yend = HEIGHT/2;
		bx = bx + WIDTH/2;
		prior[tz] = costDevice[(ystart - 1)*WIDTH*DEPTH + bx*DEPTH + tz];
	}
	if(block == 4)
	{
		ystart = HEIGHT/2; 
		yend = HEIGHT; 
		bx = bx + WIDTH/2;
		prior[tz] = dirAccCUDA[(ystart - 1)*WIDTH*DEPTH + bx*DEPTH + tz];
	}


	float e_smooth;
	float check;
	//prior[tz] = costDevice[(ystart - 1)*WIDTH*DEPTH + bx*DEPTH + tz];
__syncthreads();

	for(int y = ystart; y < yend; y++){
	
		e_smooth = prior[tz];

		for(int d = 0; d < DEPTH; d=d+8){
			
			//check= (fabsf(tz - d) == 1)? 400:6000;
			//e_smooth = fminf(e_smooth, check + prior[d]);
			check= (fabsf(tz - d) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d]);
			check= (fabsf(tz - (d+1)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+1]);
			check= (fabsf(tz - (d+2)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+2]);
			check= (fabsf(tz - (d+3)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+3]);
			check= (fabsf(tz - (d+4)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+4]);
			check= (fabsf(tz - (d+5)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+5]);
			check= (fabsf(tz - (d+6)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+6]);
			check= (fabsf(tz - (d+7)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+7]);
		}
		

__syncthreads();		
			if(tz<64)
			prior[tz] = fminf(prior[tz], prior[tz+64]);
__syncthreads();
			if(tz<32)
			prior[tz] = fminf(prior[tz], prior[tz+32]);
__syncthreads();
			if(tz<16) 
			prior[tz] = fminf(prior[tz], prior[tz+16]);
__syncthreads();
			if(tz<8) 
			prior[tz] = fminf(prior[tz], prior[tz+8]);
__syncthreads();
			if(tz<4) 
			prior[tz] = fminf(prior[tz], prior[tz+4]);
__syncthreads();
			if(tz<2) 
			prior[tz] = fminf(prior[tz], prior[tz+2]);
__syncthreads();
			if(tz<1) 
			prior[tz] = fminf(prior[tz], prior[tz+1]);

			

			if(tz>127)
			prior[tz] = fminf(prior[tz], prior[tz-32]);
__syncthreads();
			if(tz>143)
			prior[tz] = fminf(prior[tz], prior[tz-16]);
__syncthreads();
			if(tz>151)
			prior[tz] = fminf(prior[tz], prior[tz-8]);
__syncthreads();
			if(tz>155)
			prior[tz] = fminf(prior[tz], prior[tz-4]);
__syncthreads();
			if(tz>157)
			prior[tz] = fminf(prior[tz], prior[tz-2]);
__syncthreads();
			if(tz>158)
			prior[tz] = fminf(prior[tz], prior[tz-1]);
		

			check = costDevice[y*WIDTH*DEPTH  + bx*DEPTH + tz];
		

		minVal=fminf(prior[0], prior[159]);
__syncthreads();	
		dirAccCUDA[y*WIDTH*DEPTH + bx*DEPTH + tz] = check + e_smooth - minVal;
		prior[tz] = check + e_smooth - minVal;
		
__syncthreads();		
	}


}

__global__
void computeAggKernel_Oppsitex(float *costDevice, float *dirAccCUDA, float *innerBoundary_y,const int HEIGHT,const int WIDTH, const int DEPTH, const int block){
 

	__shared__ float prior[DISP_RANGE];
	__shared__ float minVal;
	int tz = threadIdx.x;
	int by = blockIdx.x;
	int xstart;
	int xend;
	if(block == 1)
	{
		xstart = WIDTH/2; xend = -1;
		//prior[tz] = dirAccCUDA[ by*WIDTH*DEPTH + (xstart + 1)*DEPTH + tz];
		prior[tz] = innerBoundary_y[by*DEPTH + tz];
	}
	if(block == 3)
	{
		xstart = WIDTH/2; xend = -1; 
		by = by + HEIGHT/2;
		//prior[tz] = dirAccCUDA[  by*WIDTH*DEPTH + (xstart + 1)*DEPTH + tz];
		prior[tz] = innerBoundary_y[by*DEPTH + tz];
	}
	if(block == 2)
	{
		xstart = WIDTH - 2;
 		xend = WIDTH/2;
		prior[tz] = costDevice[ by*WIDTH*DEPTH + (xstart + 1)*DEPTH + tz];
	}
	if(block == 4)
	{
		xstart = WIDTH - 2; 
		xend = WIDTH/2; 
		by = by + HEIGHT/2;
		prior[tz] = costDevice[ by*WIDTH*DEPTH + (xstart + 1)*DEPTH + tz];
	}


	float e_smooth;
	float check;

__syncthreads();

	for(int x = xstart; x > xend; x--){ //changed here
	
		e_smooth = prior[tz];

		for(int d = 0; d < DEPTH; d=d+8){
			
			//check= (fabsf(tz - d) == 1)? 400:6000;
			//e_smooth = fminf(e_smooth, check + prior[d]);

			check= (fabsf(tz - d) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d]);
			check= (fabsf(tz - (d+1)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+1]);
			check= (fabsf(tz - (d+2)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+2]);
			check= (fabsf(tz - (d+3)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+3]);
			check= (fabsf(tz - (d+4)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+4]);
			check= (fabsf(tz - (d+5)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+5]);
			check= (fabsf(tz - (d+6)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+6]);
			check= (fabsf(tz - (d+7)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+7]);
		}


__syncthreads();		
			if(tz<64)
			prior[tz] = fminf(prior[tz], prior[tz+64]);
__syncthreads();
			if(tz<32)
			prior[tz] = fminf(prior[tz], prior[tz+32]);
__syncthreads();
			if(tz<16) 
			prior[tz] = fminf(prior[tz], prior[tz+16]);
__syncthreads();
			if(tz<8) 
			prior[tz] = fminf(prior[tz], prior[tz+8]);
__syncthreads();
			if(tz<4) 
			prior[tz] = fminf(prior[tz], prior[tz+4]);
__syncthreads();
			if(tz<2) 
			prior[tz] = fminf(prior[tz], prior[tz+2]);
__syncthreads();
			if(tz<1) 
			prior[tz] = fminf(prior[tz], prior[tz+1]);

			

			if(tz>127)
			prior[tz] = fminf(prior[tz], prior[tz-32]);
__syncthreads();
			if(tz>143)
			prior[tz] = fminf(prior[tz], prior[tz-16]);
__syncthreads();
			if(tz>151)
			prior[tz] = fminf(prior[tz], prior[tz-8]);
__syncthreads();
			if(tz>155)
			prior[tz] = fminf(prior[tz], prior[tz-4]);
__syncthreads();
			if(tz>157)
			prior[tz] = fminf(prior[tz], prior[tz-2]);
__syncthreads();
			if(tz>158)
			prior[tz] = fminf(prior[tz], prior[tz-1]);
		

			check = costDevice[by*WIDTH*DEPTH  + x*DEPTH + tz];
		
		minVal=fminf(prior[0], prior[159]);
__syncthreads();

		dirAccCUDA[ by*WIDTH*DEPTH + x*DEPTH + tz] += check + e_smooth - minVal;
		prior[tz] = check + e_smooth - minVal;
		
__syncthreads();		
	}


	if(block == 2)
	{
		innerBoundary_y[by*DEPTH + tz] = prior[tz];
	}
	if(block == 4)
	{
		innerBoundary_y [by*DEPTH + tz] = prior[tz];
	}



}

__global__
void computeAggKernel_Oppsitey(float *costDevice, float *dirAccCUDA, float *innerBoundary_x, const int HEIGHT, const int WIDTH, const int DEPTH, const int block){
 

	__shared__ float prior[DISP_RANGE];
	__shared__ float minVal;
	int tz = threadIdx.x;
	int bx = blockIdx.x;
	int ystart;
	int yend;
	if(block == 1)
	{
		ystart = HEIGHT/2; yend = -1;
		//prior[tz] = dirAccCUDA[(ystart + 1)*WIDTH*DEPTH + bx*DEPTH + tz];
		prior[tz] = innerBoundary_x[bx*DEPTH + tz];

	}
	if(block == 3)
	{
		ystart = HEIGHT - 2; yend = HEIGHT/2; 
		prior[tz] = costDevice[(ystart + 1)*WIDTH*DEPTH + bx*DEPTH + tz];
	}
	if(block == 2)
	{
		ystart = HEIGHT/2; yend = -1;
		bx = bx + WIDTH/2;
		prior[tz] = innerBoundary_x[bx*DEPTH + tz];
	}
	if(block == 4)
	{
		ystart = HEIGHT - 2;  yend = HEIGHT/2; 
		bx = bx + WIDTH/2;
		prior[tz] = costDevice[(ystart + 1)*WIDTH*DEPTH + bx*DEPTH + tz];
	}


	float e_smooth;
	float check;
__syncthreads();

	for(int y = ystart; y > yend; y--){
	
		e_smooth = prior[tz];

		for(int d = 0; d < DEPTH; d=d+8){
			
			//check= (fabsf(tz - d) == 1)? 400:6000;
			//e_smooth = fminf(e_smooth, check + prior[d]);
			check= (fabsf(tz - d) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d]);
			check= (fabsf(tz - (d+1)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+1]);
			check= (fabsf(tz - (d+2)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+2]);
			check= (fabsf(tz - (d+3)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+3]);
			check= (fabsf(tz - (d+4)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+4]);
			check= (fabsf(tz - (d+5)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+5]);
			check= (fabsf(tz - (d+6)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+6]);
			check= (fabsf(tz - (d+7)) > 1);
			e_smooth = fminf(e_smooth, check*5600 + 400 + prior[d+7]);
		}
		

__syncthreads();		
			if(tz<64)
			prior[tz] = fminf(prior[tz], prior[tz+64]);
__syncthreads();
			if(tz<32)
			prior[tz] = fminf(prior[tz], prior[tz+32]);
__syncthreads();
			if(tz<16) 
			prior[tz] = fminf(prior[tz], prior[tz+16]);
__syncthreads();
			if(tz<8) 
			prior[tz] = fminf(prior[tz], prior[tz+8]);
__syncthreads();
			if(tz<4) 
			prior[tz] = fminf(prior[tz], prior[tz+4]);
__syncthreads();
			if(tz<2) 
			prior[tz] = fminf(prior[tz], prior[tz+2]);
__syncthreads();
			if(tz<1) 
			prior[tz] = fminf(prior[tz], prior[tz+1]);

			

			if(tz>127)
			prior[tz] = fminf(prior[tz], prior[tz-32]);
__syncthreads();
			if(tz>143)
			prior[tz] = fminf(prior[tz], prior[tz-16]);
__syncthreads();
			if(tz>151)
			prior[tz] = fminf(prior[tz], prior[tz-8]);
__syncthreads();
			if(tz>155)
			prior[tz] = fminf(prior[tz], prior[tz-4]);
__syncthreads();
			if(tz>157)
			prior[tz] = fminf(prior[tz], prior[tz-2]);
__syncthreads();
			if(tz>158)
			prior[tz] = fminf(prior[tz], prior[tz-1]);
		

			check = costDevice[y*WIDTH*DEPTH  + bx*DEPTH + tz];
		

		minVal=fminf(prior[0], prior[159]);
__syncthreads();	
		dirAccCUDA[y*WIDTH*DEPTH + bx*DEPTH + tz] += check + e_smooth - minVal;
		prior[tz] = check + e_smooth - minVal;
		
__syncthreads();		
	}

	if(block == 3)
	{
		innerBoundary_x[bx*DEPTH + tz] = prior[tz];
	}
	if(block == 4)
	{
		innerBoundary_x[bx*DEPTH + tz] = prior[tz];
	}
}

__global__
void sum(float* acc1, float* acc2){


}



void SGM::aggregationCUDA(cv::Mat &accumulatedCost, cv::Mat &cost){

	const int WIDTH  = cost.cols;
	const int HEIGHT = cost.rows;
	const int DEPTH = DISP_RANGE;

	const int size = WIDTH*HEIGHT*DEPTH;
	float *costHost = (float*)calloc(size, sizeof(float));          //host pageable
	float *dirAccHost_0 = (float*)calloc(size, sizeof(float));	//host pageable
	float *dirAccHost_1 = (float*)calloc(size, sizeof(float));	//host pageable


	float *costHostPin;
	float *dirAccHostPin_0;
	float *dirAccHostPin_1;
	cudaError_t status = cudaMallocHost ( (void**)&costHostPin, size * sizeof(float) );	//host pinned
	if (status != cudaSuccess) printf("Error allocating pinned host memoryn");
		    status = cudaMallocHost( (void**)&dirAccHostPin_0, size * sizeof(float) );	//host pinned
	if (status != cudaSuccess) printf("Error allocating pinned host memoryn");
		    status = cudaMallocHost( (void**)&dirAccHostPin_1, size * sizeof(float) );	//host pinned
	if (status != cudaSuccess) printf("Error allocating pinned host memoryn");

	memset(costHost, 0, size * sizeof(float));
	memset(dirAccHost_0, 0, size * sizeof(float));
	memset(dirAccHost_1, 0, size * sizeof(float));

	//row major copy
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			for(int d = 0; d < DEPTH; d++){
				costHost[y*WIDTH*DEPTH + x*DEPTH + d] = cost.at<SGM::VecDf>(y,x)[d];
				//std::cout<<costHost[y*WIDTH*DEPTH + x*DEPTH + d]<<std::endl;sleep(1);
			}
		}
	}

	for (int y = 0 ; y < HEIGHT ; y++ ) {
		for(int d = 0; d < DEPTH; d++){
        		dirAccHost_0[y*WIDTH*DEPTH + 0*DEPTH + d] = directCost.at<SGM::VecDf>(y, 0)[d];
		}				
    	}

	for (int x = 0 ; x < WIDTH ; x++ ) {
		for(int d = 0; d < DEPTH; d++){
        		dirAccHost_1[0*WIDTH*DEPTH + x*DEPTH + d] = directCost.at<SGM::VecDf>(0, x)[d];
		}				
    	}

	memcpy(costHostPin, costHost, size * sizeof(float));
	memcpy(dirAccHostPin_0, dirAccHost_0, size * sizeof(float));
	memcpy(dirAccHostPin_1, dirAccHost_1, size * sizeof(float));

	float *costDevice;
	float *dirAccDevice_0;
	float *dirAccDevice_1;
	float *innerBoundary_x;
	float *innerBoundary_y;

	cudaMalloc ( (void**)&costDevice, size * sizeof(float) );	//device
	cudaMalloc( (void**)&dirAccDevice_0, size * sizeof(float) );	//device
	cudaMalloc( (void**)&dirAccDevice_1, size * sizeof(float) );	//device
	cudaMalloc( (void**)&innerBoundary_x, WIDTH * DEPTH * sizeof(float) );	//device
	cudaMalloc( (void**)&innerBoundary_y, HEIGHT * DEPTH * sizeof(float) );	//device

	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate ( &stream1);	
	cudaStreamCreate ( &stream2);
	cudaStreamCreate ( &stream3);
	cudaStreamCreate ( &stream4);

	dim3 dimBlock(DEPTH, 1, 1);
	dim3 dimGrid_xAgg(HEIGHT/2, 1, 1);
	dim3 dimGrid_yAgg(WIDTH/2, 1, 1);

cudaProfilerStart();
	for(int y = 0; y < HEIGHT/2; y=y+1){
		cudaMemcpyAsync(&costDevice[y*WIDTH*DEPTH], &costHostPin[y*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyHostToDevice, stream1);
		//cudaMemcpyAsync(&costDevice[(y+1)*WIDTH*DEPTH], &costHostPin[(y+1)*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyHostToDevice, stream3);
	}
	cudaThreadSynchronize();
	computeAggKernel_x<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, HEIGHT, WIDTH, DEPTH,1);
	computeAggKernel_y<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, HEIGHT, WIDTH, DEPTH,1);

	for(int y = 0; y < HEIGHT/2; y=y+1){
		cudaMemcpyAsync(&costDevice[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], &costHostPin[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyHostToDevice, stream1);
	}
	cudaThreadSynchronize();

	computeAggKernel_x<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, HEIGHT, WIDTH, DEPTH,2);
	computeAggKernel_y<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, HEIGHT, WIDTH, DEPTH,2);



	for(int y = HEIGHT/2; y < HEIGHT; y=y+1){
		cudaMemcpyAsync(&costDevice[y*WIDTH*DEPTH], &costHostPin[y*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyHostToDevice, stream1);
	}
	cudaThreadSynchronize();
	computeAggKernel_x<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, HEIGHT, WIDTH, DEPTH,3);
	computeAggKernel_y<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, HEIGHT, WIDTH, DEPTH,3);

	for(int y = HEIGHT/2; y < HEIGHT; y=y+1){
		cudaMemcpyAsync(&costDevice[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], &costHostPin[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyHostToDevice, stream1);
	}
	cudaThreadSynchronize();


	computeAggKernel_x<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, HEIGHT, WIDTH, DEPTH,4);
	computeAggKernel_y<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, HEIGHT, WIDTH, DEPTH,4);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	computeAggKernel_Oppsitex<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, innerBoundary_y, HEIGHT, WIDTH, DEPTH,4);
	computeAggKernel_Oppsitey<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, innerBoundary_x, HEIGHT, WIDTH, DEPTH,4);
	cudaThreadSynchronize();

	for(int y = HEIGHT/2; y < HEIGHT; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_0[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], &dirAccDevice_0[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream4);
	}

	for(int y = HEIGHT/2; y < HEIGHT; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_1[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], &dirAccDevice_1[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream3);
	}

	computeAggKernel_Oppsitex<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, innerBoundary_y, HEIGHT, WIDTH, DEPTH,3);
	computeAggKernel_Oppsitey<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, innerBoundary_x, HEIGHT, WIDTH, DEPTH,3);
	cudaThreadSynchronize();

	for(int y = HEIGHT/2; y < HEIGHT; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_0[y*WIDTH*DEPTH], &dirAccDevice_0[y*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream4);
	}

	for(int y = HEIGHT/2; y < HEIGHT; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_1[y*WIDTH*DEPTH], &dirAccDevice_1[y*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream3);
	}

	computeAggKernel_Oppsitex<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, innerBoundary_y, HEIGHT, WIDTH, DEPTH,2);
	computeAggKernel_Oppsitey<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, innerBoundary_x, HEIGHT, WIDTH, DEPTH,2);
	cudaThreadSynchronize();
	for(int y = 0; y < HEIGHT/2; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_0[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], &dirAccDevice_0[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream4);
	}

	for(int y = 0; y < HEIGHT/2; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_1[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], &dirAccDevice_1[y*WIDTH*DEPTH + (WIDTH/2)*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream3);
	}

	computeAggKernel_Oppsitex<<<dimGrid_xAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_0, innerBoundary_y, HEIGHT, WIDTH, DEPTH,1);
	computeAggKernel_Oppsitey<<<dimGrid_yAgg, dimBlock, 0, stream2>>>(costDevice, dirAccDevice_1, innerBoundary_x, HEIGHT, WIDTH, DEPTH,1);
	cudaThreadSynchronize();
	for(int y = 0; y < HEIGHT/2; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_0[y*WIDTH*DEPTH], &dirAccDevice_0[y*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream4);
	}

	for(int y = 0; y < HEIGHT/2; y=y+1){
		cudaMemcpyAsync(&dirAccHostPin_1[y*WIDTH*DEPTH], &dirAccDevice_1[y*WIDTH*DEPTH], WIDTH*DEPTH/2 * sizeof(float), cudaMemcpyDeviceToHost, stream3);
	}

	cudaThreadSynchronize();

cudaProfilerStop();

cudaThreadSynchronize();
cudaStreamDestroy(stream3);
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaStreamDestroy(stream4);



cudaFree(innerBoundary_y);
cudaFree(innerBoundary_x);
cudaFree(costDevice);
cudaFree(dirAccDevice_0);
cudaFree(dirAccDevice_1);

memcpy(dirAccHost_0, dirAccHostPin_0, size * sizeof(float));
memcpy(dirAccHost_1, dirAccHostPin_1, size * sizeof(float));
cudaFreeHost(costHostPin);
cudaFreeHost(dirAccHostPin_0);
cudaFreeHost(dirAccHostPin_1);


	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			for(int d = 0; d < DEPTH; d++){
				accumulatedCost.at<SGM::VecDf>(y,x)[d] = dirAccHost_0[y*WIDTH*DEPTH + x*DEPTH + d] + dirAccHost_1[y*WIDTH*DEPTH + x*DEPTH + d];
			}
		}
	}

}
