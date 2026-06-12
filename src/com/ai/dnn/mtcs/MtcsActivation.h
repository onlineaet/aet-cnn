#ifndef __COM_AI_DNN_MTCS_ACTIVATION_H__
#define __COM_AI_DNN_MTCS_ACTIVATION_H__

#include "../Activation.h"

package$ com.ai.dnn.mtcs;

public$ class$ MtcsActivation{
   public$ static __device__ float lhtan(float x);
   public$ static __device__ float hardtan(float x);
   public$ static __device__ float linear(float x);
   public$ static __device__ float logistic(float x);
   public$ static __device__ float loggy(float x);
   public$ static __device__ float relu(float x);
   public$ static __device__ float relu6(float x);
   public$ static __device__ float elu(float x);
   public$ static __device__ float selu(float x);
   public$ static __device__ float relie(float x);
   public$ static __device__ float ramp(float x);
   public$ static __device__ float leaky(float x);
   public$ static __device__ float tanh(float x);
   public$ static __device__ float gelu(float x);
   public$ static __device__ float softplus(float x);
   public$ static __device__ float softplus(float x, float threshold);
   public$ static __device__ float plse(float x);
   public$ static __device__ float stair(float x);
   public$ static __device__ float activate(float x, ActivationType a);
   public$ static __device__ float softplus_kernel(float x, float threshold);

   public$ static __global__ void  logistic(float *x, int totalSize);
   public$ static __global__ void  leaky(float *x, int totalSize);
   public$ static __global__ void  tanh(float *x, int totalSize);
   public$ static __global__ void  hardtan(float *x, int totalSize);
   public$ static __global__ void  relu(float *x, int totalSize);
   public$ static __global__ void  relu6(float *x, int totalSize);
   public$ static __global__ void  selu(float *x, int totalSize);
   public$ static __global__ void  gelu(float *x, int totalSize);
   public$ static __global__ void  activateArray(float *x, int totalSize, ActivationType type);
   public$ static void activate(int totalSize,float *dataArray,const ActivationType type);

   public$ static __global__ void gradientSwish(float *outputs, int totalSize, float *sigmoids, float *deltas);
   public$ static __global__ void gradientMish(int totalSize,float *activation_input_gpu, float *deltas);
   public$ static __device__ float hard_mish_yashas_grad(float x);
   public$ static __global__ void gradientHardMish(int n, float *activation_input_gpu, float *delta);
   private$ static __global__ void gradientNormlizeChannelsSoftMaxKernel(float *x, int size, int batch,
           int channels, int wh_step, float *delta_gpu);
   public$ static void gradientNormlizeChannelsSoftMax(float *output_gpu, int n, int batch,
         int channels, int wh_step, float *delta_gpu);
   private$ static __global__ void gradientNormlizeChannelsKernel(float *x, int size, int batch,
         int channels, int wh_step, float *delta_gpu);
   public$ static void gradientNormlizeChannels(float *output_gpu, int n, int batch,
         int channels, int wh_step, float *delta_gpu);

   public$ static __device__ float gradientLinear(float x);
   public$ static __device__ float gradientLogistic(float x);
   public$ static __device__ float gradientLeaky(float x);
   public$ static __device__ float gradientTanh(float x);
   public$ static __device__ float gradientHardtan(float x);
   public$ static __device__ float gradientRelu(float x);
   public$ static __device__ float gradientRelu6(float x);
   public$ static __device__ float gradientSelu(float x);
   public$ static __device__ float gradientGelu(float x);
   public$ static __device__ float gradient_kernel(float x, ActivationType a);
   public$ static __device__ float gradientLoggy(float x);
   public$ static __device__ float gradientElu(float x);
   public$ static __device__ float gradientRelie(float x);
   public$ static __device__ float gradientRamp(float x);
   public$ static __device__ float gradientPlse(float x);
   public$ static __device__ float gradientStair(float x);
   public$ static __device__ float gradientLhtan(float x);
   public$ static __device__ float gradient(float x, ActivationType a);
   private$ static __device__ float sech(float x);

   public$ static __global__ void gradientLeaky(float *x, int size, float *delta);
   public$ static __global__ void gradientRevleaky(float *x, int size, float *delta);
   public$ static __global__ void gradientLogistic(float *x, int size, float *delta);
   public$ static __global__ void gradientTanh(float *x, int size, float *delta);
   public$ static __global__ void gradientHardtan(float *x, int size, float *delta);
   public$ static __global__ void gradientRelu(float *x, int size, float *delta);
   public$ static __global__ void gradientRelu6(float *x, int size, float *delta);
   public$ static __global__ void gradientSelu(float *x, int size, float *delta);
   public$ static __global__ void gradientGelu(float *x, int size, float *delta);
   public$ static __global__ void gradientKernel(float *x, int size, ActivationType a, float *delta);

   public$ static void gradient(float *output, int size,ActivationType a, float *delta);


};


#endif

