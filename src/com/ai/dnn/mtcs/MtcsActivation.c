#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <aet/lang/System.h>
#include <aet/lang/AAssert.h>
#include <aet/mtcs/MtcsSystem.h>

#include "../cnnmicro.h"
#include "../DnnUtils.h"

#include "MtcsActivation.h"
#include "MtcsTool.h"

impl$ MtcsActivation{

   static __device__ float lhtan(float x){
      if(x < 0) return .001*x;
      if(x > 1) return .001*(x-1) + 1;
      return x;
   }

   static __device__ float hardtan(float x){
      if (x < -1) return -1;
      if (x > 1) return 1;
      return x;
   }

   static __device__ float linear(float x){
      return x;
   }

   static __device__ float logistic(float x){
      return 1.f/(1.f + expf(-x));
   }

   static __device__ float loggy(float x){
      return 2.f/(1.f + expf(-x)) - 1;
   }

   static __device__ float relu(float x){
      return x*(x>0);
   }

   static  __device__ float relu6(float x) {
      return min_val_cmp(max_val_cmp(x, 0), 6);
   }

   static __device__ float elu(float x){
      return (x >= 0)*x + (x < 0)*(expf(x)-1);
   }

   static __device__ float selu(float x) {
      return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x) - 1);
   }

   static __device__ float relie(float x){
      return (x>0) ? x : .01f*x;
   }

   static __device__ float ramp(float x){
      return x*(x>0)+.1f*x;
   }


   static __device__ float leaky(float x){
      return (x>0) ? x : .1f*x;
   }

   static __device__ float tanh(float x){
      return (2/(1 + expf(-2*x)) - 1);
   }

   static __device__ float gelu(float x){
      return (0.5*x*(1 + tanhf(0.797885*x + 0.035677*powf(x, 3))));
   }

   static __device__ float softplus(float x, float threshold) {
      if (x > threshold)
         return x;                // too large
      else if (x < -threshold)
         return expf(x);    // too small
      return log1pf(expf(x));
      //return logf(expf(x) + 1);
   }

   public$ static __device__ float softplus(float x){
      return softplus(x,20.0);
   }

   static __device__ float plse(float x){
      if(x < -4) return .01f * (x + 4);
      if(x > 4)  return .01f * (x - 4) + 1;
      return .125f*x + .5f;
   }

   static __device__ float stair(float x){
      int n = floorf(x);
      if (n%2 == 0)
         return floorf(x/2.f);
      else
         return (x - n) + floorf(x/2.f);
   }

   static __global__ void logistic(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = logistic(x[index]);
      }
   }

   static __global__ void leaky(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
        x[index] = leaky(x[index]);
      }
   }

   static __global__ void tanh(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = tanh(x[index]);
      }
   }

   static __global__ void hardtan(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = hardtan(x[index]);
      }
   }

   static __global__ void relu(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = relu(x[index]);
      }
   }

   static __global__ void relu6(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = relu6(x[index]);
      }
   }

   static __global__ void selu(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = selu(x[index]);
      }
   }

   static __global__ void gelu(float *x, int totalSize){
      int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = gelu(x[index]);
      }
   }


   static __device__ float activate(float x, ActivationType a){
      switch(a){
         case ActivationType.LINEAR:
            return linear(x);
         case ActivationType.LOGISTIC:
            return logistic(x);
         case ActivationType.LOGGY:
            return loggy(x);
         case ActivationType.RELU:
            return relu(x);
         case ActivationType.RELU6:
            return relu6(x);
         case ActivationType.ELU:
            return elu(x);
         case ActivationType.SELU:
            return selu(x);
         case ActivationType.GELU:
            return gelu(x);
         case ActivationType.RELIE:
            return relie(x);
         case ActivationType.RAMP:
            return ramp(x);
         case ActivationType.LEAKY:
            return leaky(x);
         case ActivationType.TANH:
            return tanh(x);
         case ActivationType.PLSE:
            return plse(x);
         case ActivationType.STAIR:
            return stair(x);
         case ActivationType.HARDTAN:
            return hardtan(x);
         case ActivationType.LHTAN:
            return lhtan(x);
      }
      return 0;
   }

   static __global__ void activateArray(float *x, int totalSize, ActivationType type){
      int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (index < totalSize) {
         x[index] = activate(x[index],type);
      }
   }


   static void activate(int totalSize,float *dataArray,const ActivationType type){
      const int num_blocks = MtcsTool.getNumberOfBlocks/*!get_number_of_blocks*/(totalSize, MTCS_BLOCK);
      switch(type){
         case ActivationType.LINEAR:
            return;
         case ActivationType.LEAKY:
            leaky <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;
         case ActivationType.LOGISTIC:
            logistic <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;

         case ActivationType.TANH:
            tanh <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;
         case ActivationType.HARDTAN:
            hardtan <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;
         case ActivationType.RELU:
            relu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;
         case ActivationType.RELU6:
            relu6 <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;
         case ActivationType.SELU:
            selu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;
         case ActivationType.GELU:
            gelu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream()>>>(dataArray, totalSize);
            break;

         default:
            activateArray<<<MtcsTool.gridSize(totalSize)/*!cuda_gridsize(n)*/, MTCS_BLOCK, 0, MtcsTool.getStream()>>>
                  (dataArray, totalSize,type);
            break;
      }
   }


   // https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cu#L28-L30
   //原型 gradient_array_swish_kernel activations.h activation_kernels.cu
   static __global__ void gradientSwish(float *x, int n, float *sigmoid_gpu, float *delta){
      int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (i < n) {
         float swish = x[i];
         delta[i] *= swish + sigmoid_gpu[i] * (1 - swish); // gradient_kernel(x[i], a);
      }
   }

   static __device__ float softplus_kernel(float x, float threshold) {
      if (x > threshold)
         return x;                // too large
      else if (x < -threshold)
         return expf(x);    // too small
      return log1pf(expf(x));
   }


   // https://github.com/digantamisra98/Mish
  //原型 gradient_array_mish_kernel activations.h activation_kernels.cu
   static __global__ void gradientMish(int n, float *activation_input_gpu, float *delta){
      int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (i < n) {
         const float MISH_THRESHOLD = 20.0f;

         // implementation from TensorFlow: https://github.com/tensorflow/addons/blob/093cdfa85d334cbe19a37624c33198f3140109ed/tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h#L66-L80
         // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
         // log1p(x) == log(x + 1)
         const float inp = activation_input_gpu[i];
         const float sp = softplus_kernel(inp, MISH_THRESHOLD);
         const float grad_sp = -expm1f(-sp);
         //const float grad_sp = 1 - expf(-sp);
         const float tsp = tanh(sp);
         const float grad_tsp = (1 - tsp*tsp) * grad_sp;
         const float grad = inp * grad_tsp + tsp;
         delta[i] *= grad;

         //float x = activation_input[i];
         //float d = 2 * expf(x) + expf(2 * x) + 2;
         //float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
         //float derivative = expf(x) * w / (d * d);
         //delta[i] *= derivative;
      }
   }

   static __device__ float hard_mish_yashas_grad(float x){
      if (x > 0)
      return 1;
      if (x > -2)
      return x + 1;
      return 0;
   }

   //原型 gradient_array_hard_mish_kernel activations.h activation_kernels.cu
   static __global__ void gradientHardMish(int n, float *activation_input_gpu, float *delta){
      int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (i < n) {
         const float x = activation_input_gpu[i];
         delta[i] *= hard_mish_yashas_grad(x);
      }
   }

   //原型 gradient_array_normalize_channels_softmax_kernel activations.h activation_kernels.cu
   static __global__ void gradientNormlizeChannelsSoftMaxKernel(float *x, int size, int batch,
         int channels, int wh_step, float *delta_gpu){
       int i = blockIdx.x * blockDim.x + threadIdx.x;

       int wh_i = i % wh_step;
       int b = i / wh_step;

       if (i < size) {
           int k;
           /*
           float grad = 0;
           for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               float out = x[index];
               float delta = delta_gpu[index];
               grad += out*fabs(delta);
           }
           */
           for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               float delta = delta_gpu[index];
               float grad = x[index] * (1 - x[index]);
               delta = delta * grad;
               if (isnan(delta) || isinf(delta)) delta = 0;
               delta_gpu[index] = delta;
           }
       }
   }

   //原型 gradient_array_normalize_channels_softmax_ongpu activation_kernels.cu
   static void gradientNormlizeChannelsSoftMax(float *output_gpu, int n, int batch,
         int channels, int wh_step, float *delta_gpu){
       // n = w*h*c*batch
       // size = w*h*batch
       int size = n / channels;
       const int num_blocks =MtcsTool.getNumberOfBlocks(size, MTCS_BLOCK);
       gradientNormlizeChannelsSoftMaxKernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
             (output_gpu, size, batch, channels, wh_step, delta_gpu);
   }

   //原型    gradient_array_normalize_channels_kernel
   static __global__ void gradientNormlizeChannelsKernel(float *x, int size, int batch,
         int channels, int wh_step, float *delta_gpu){
       int i = blockIdx.x * blockDim.x + threadIdx.x;

       int wh_i = i % wh_step;
       int b = i / wh_step;

       if (i < size) {
           int k;
           /*
           float grad = 0;
           for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               float out = x[index];
               float delta = delta_gpu[index];
               grad += out*fabs(delta);
           }
           */
           for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               if (x[index] > 0) {
                   float delta = delta_gpu[index];
                   float grad = x[index];
                   delta = delta * grad;
                   delta_gpu[index] = delta;
               }
           }
       }
   }

   //原型 gradient_array_normalize_channels_ongpu activations.h activation_kernels.cu
   static void gradientNormlizeChannels(float *output_gpu, int n, int batch,
         int channels, int wh_step, float *delta_gpu){
       // n = w*h*c*batch
       // size = w*h*batch
       int size = n / channels;
       const int num_blocks =MtcsTool.getNumberOfBlocks(size, MTCS_BLOCK);
       gradientNormlizeChannelsKernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
             (output_gpu, size, batch, channels, wh_step, delta_gpu);
   }

   static __device__ float gradientLinear(float x){
      return x;
   }

   static __device__ float gradientLeaky(float x){
      return (x>0) ? 1 : .1f;
   }

   static __device__ float gradientLogistic(float x){
      return (1-x)*x;
   }

   static __device__ float gradientTanh(float x){
      return 1-x*x;
   }

   static __device__ float gradientHardtan(float x){
      if (x > -1 && x < 1)
         return 1;
      return 0;
   }

   static __device__ float gradientLhtan(float x){
      if(x > 0 && x < 1)
         return 1;
      return .001;
   }

   static __device__ float gradientRelu(float x){
      return (x>0);
   }

   static __device__ float gradientRelu6(float x) {
      return (x > 0 && x < 6);
   }

   static __device__ float gradientSelu(float x) {
      return (x >= 0)*1.0507f + (x < 0)*(x + 1.0507f*1.6732f);
   }

   static __device__ float sech(float x) {
      return 2 / (expf(x) + expf(-x));
   }


   static __device__ float gradientGelu(float x) {
      const float x3 = powf(x, 3);
      return 0.5*tanhf(0.0356774*x3 + 0.797885*x)
      + (0.0535161*x3 + 0.398942*x) * powf(sech(0.0356774*x3 + 0.797885*x), 2) + 0.5;
   }

   static __device__ float gradientLoggy(float x){
       float y = (x+1.F)/2.F;
       return 2*(1-y)*y;
   }

   static __device__ float gradientElu(float x){
      return (x >= 0) + (x < 0)*(x + 1);
   }

   static __device__ float gradientRelie(float x){
      return (x>0) ? 1 : .01f;
   }

   static __device__ float gradientRamp(float x){
      return (x>0)+.1f;
   }

   static __device__ float gradientPlse(float x){
      return (x < 0 || x > 1) ? .01f : .125f;
   }

   static  __device__ float gradientStair(float x){
       if (floorf(x) == x)
          return 0;
       return 1;
   }

   static __global__ void gradientLeaky(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientLeaky(x[i]);
      }
   }

   static __global__ void gradientRevleaky(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] /= gradientLeaky(x[i]);
      }
   }


   static __global__ void gradientLogistic(float *x, int size, float *delta){
       int i = blockIdx.x*blockDim.x + threadIdx.x;
       if (i < size) {
          delta[i] *= gradientLogistic(x[i]);
       }
   }

   static __global__ void gradientTanh(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientTanh(x[i]);
      }
   }


   static __global__ void gradientHardtan(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientHardtan(x[i]);
      }
   }

   static __global__ void gradientRelu(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientRelu(x[i]);
      }
   }

   static __global__ void gradientRelu6(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientRelu6(x[i]);
      }
   }


   static __global__ void gradientSelu(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientSelu(x[i]);
      }
   }

   static __global__ void gradientGelu(float *x, int size, float *delta){
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < size) {
         delta[i] *= gradientGelu(x[i]);
      }
   }

   static __device__ float gradient(float x, ActivationType a){
      switch (a) {
         case ActivationType.LINEAR:
            return gradientLinear(x);
         case ActivationType.LOGISTIC:
            return gradientLogistic(x);
         case ActivationType.LOGGY:
            return gradientLoggy(x);
         case ActivationType.RELU:
            return gradientRelu(x);
         case ActivationType.RELU6:
            return gradientRelu6(x);
         case ActivationType.NORM_CHAN:
            return gradientRelu(x);
         case ActivationType.ELU:
            return gradientElu(x);
         case ActivationType.SELU:
            return gradientSelu(x);
         case ActivationType.GELU:
            return gradientGelu(x);
         case ActivationType.RELIE:
            return gradientRelie(x);
         case ActivationType.RAMP:
            return gradientRamp(x);
         case ActivationType.LEAKY:
            return gradientLeaky(x);
         case ActivationType.TANH:
            return gradientTanh(x);
         case ActivationType.PLSE:
            return gradientPlse(x);
         case ActivationType.STAIR:
            return gradientStair(x);
         case ActivationType.HARDTAN:
            return gradientHardtan(x);
         case ActivationType.LHTAN:
            return gradientLhtan(x);
      }
      return 0;
   }

   static __global__ void gradientKernel(float *x, int size,ActivationType a, float *delta){
      int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if(i < size) {
         delta[i] *= gradient(x[i], a);
      }
   }

   static void gradient(float *output, int size,ActivationType a, float *delta){
       const int num_blocks =MtcsTool.getNumberOfBlocks(size, MTCS_BLOCK);
       if (a == ActivationType.LINEAR)
          return;
       else if (a == ActivationType.LEAKY){
          gradientLeaky <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       }else if (a == ActivationType.REVLEAKY)
          gradientRevleaky <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       else if (a == ActivationType.LOGISTIC)
          gradientLogistic <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       else if (a == ActivationType.TANH)
          gradientTanh <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       else if (a == ActivationType.HARDTAN)
          gradientHardtan <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       else if (a == ActivationType.RELU)
          gradientRelu <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       else if (a == ActivationType.RELU6)
          gradientRelu6 <<< num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output,size,delta);
       else if (a == ActivationType.NORM_CHAN_SOFTMAX || a == ActivationType.NORM_CHAN) {
           a_error("Error: should be used custom NORM_CHAN_SOFTMAX-function for gradient");
       }else if (a == ActivationType.SELU)
          gradientSelu <<<num_blocks,MTCS_BLOCK, 0, MtcsTool.getStream() >>>(output,size,delta);
       else if (a ==ActivationType.GELU)
          gradientGelu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(output,size,delta);
       else
          gradientKernel <<<MtcsTool.gridSize(size), MTCS_BLOCK, 0, MtcsTool.getStream() >>> (output, size, a, delta);
   }



};


