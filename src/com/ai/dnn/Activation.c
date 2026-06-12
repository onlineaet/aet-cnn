#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <aet/lang/System.h>
#include <aet/lang/AAssert.h>

#include <immintrin.h>

#include "Activation.h"

//最快的leak
static   void leaky_fast(int batch,int size,int channels,float **dataArray,int needThreads)
{
   int b,i;
   int spatial=size*channels;
   #pragma omp parallel for num_threads( needThreads ) private( b,i )
   for(b = 0; b < batch; ++b){
      float *data=dataArray[b];
      for(i = 0; i < spatial; ++i)
         if(data[i]<0)
            data[i]*=0.1;
   }
}


static void leaky_gradient_new(float *outputData,float *src,int n)
{
   int i;
   for(i=0;i<n;i++){
      if(outputData[i]<=0)
         src[i]*=0.1;
   }
}

static void leaky_gradient_avx(float *src,int n)
{
   __m256 all256_1 = _mm256_set1_ps(1.0f);
   __m256 all256_01 = _mm256_set1_ps(0.1f);
   __m256 src256,vCondition,result256;
   int i;
   __m256 condition = _mm256_set1_ps(0);

   for (i = 0; i < n - 8; i += 8) {
      src256 = _mm256_loadu_ps(&src[i]);
      vCondition= _mm256_cmp_ps(src256, condition, _CMP_LE_OS);
      result256 = _mm256_blendv_ps(all256_1, all256_01, vCondition); // (sign>0) ? src : mult;
      _mm256_storeu_ps(&src[i], result256);
   }
   for (; i < n; ++i)
      src[i]=src[i]>0?1:0.1;
}

static float hard_mish_yashas_grad(float x)
{
   if (x > 0)
      return 1;
   if (x > -2)
      return x + 1;
   return 0;
}

impl$ Activation{

   char *getString(ActivationType type){
      switch(type){
         case ActivationType.LOGISTIC:
            return "logistic";
         case ActivationType.LOGGY:
            return "loggy";
         case ActivationType.RELU:
            return "relu";
         case ActivationType.ELU:
            return "elu";
         case ActivationType.SELU:
            return "selu";
         case ActivationType.RELIE:
            return "relie";
         case ActivationType.RAMP:
            return "ramp";
         case ActivationType.LINEAR:
            return "linear";
         case ActivationType.TANH:
            return "tanh";
         case ActivationType.PLSE:
            return "plse";
         case ActivationType.LEAKY:
            return "leaky";
         case ActivationType.STAIR:
            return "stair";
         case ActivationType.HARDTAN:
            return "hardtan";
         case ActivationType.LHTAN:
            return "lhtan";
         case ActivationType.SWISH:
            return "swish";
         case ActivationType.MISH:
            return "mish";
         default:
            break;
      }
      return "relu";
   }

   ActivationType getType(char *str){
      if (strcmp(str, "logistic")==0) return ActivationType.LOGISTIC;
      if (strcmp(str, "loggy")==0) return ActivationType.LOGGY;
      if (strcmp(str, "relu")==0) return ActivationType.RELU;
      if (strcmp(str, "elu")==0) return ActivationType.ELU;
      if (strcmp(str, "selu")==0) return ActivationType.SELU;
      if (strcmp(str, "relie")==0) return ActivationType.RELIE;
      if (strcmp(str, "plse")==0) return ActivationType.PLSE;
      if (strcmp(str, "hardtan")==0) return ActivationType.HARDTAN;
      if (strcmp(str, "lhtan")==0) return ActivationType.LHTAN;
      if (strcmp(str, "linear")==0) return ActivationType.LINEAR;
      if (strcmp(str, "ramp")==0) return ActivationType.RAMP;
      if (strcmp(str, "leaky")==0) return ActivationType.LEAKY;
      if (strcmp(str, "tanh")==0) return ActivationType.TANH;
      if (strcmp(str, "stair")==0) return ActivationType.STAIR;
      if (strcmp(str, "swish")==0) return ActivationType.SWISH;
      if (strcmp(str, "mish")==0) return ActivationType.MISH;
      fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n",str);
      return ActivationType.RELU;
   }

   float activate(float x, ActivationType type){
      switch(type){
         case ActivationType.LINEAR:
            return linear_activate(x);
         case ActivationType.LOGISTIC:
            return logistic_activate(x);
         case ActivationType.LOGGY:
            return loggy_activate(x);
         case ActivationType.RELU:
            return relu_activate(x);
         case ActivationType.ELU:
            return elu_activate(x);
         case ActivationType.SELU:
            return selu_activate(x);
         case ActivationType.RELIE:
            return relie_activate(x);
         case ActivationType.RAMP:
            return ramp_activate(x);
         case ActivationType.LEAKY:
            return leaky_activate(x);
         case ActivationType.TANH:
            return tanh_activate(x);
         case ActivationType.PLSE:
            return plse_activate(x);
         case ActivationType.STAIR:
            return stair_activate(x);
         case ActivationType.HARDTAN:
            return hardtan_activate(x);
         case ActivationType.LHTAN:
            return lhtan_activate(x);
      }
      return 0;
   }

   void activate(float *x,int n,const ActivationType type){
      int i;
      if (type == ActivationType.LINEAR) {

      }else if (type == ActivationType.LEAKY) {
         #pragma omp parallel for
         for (i = 0; i < n; ++i) {
            x[i] = leaky_activate(x[i]);
         }
      }else if (type == ActivationType.LOGISTIC) {
         #pragma omp parallel for
         for (i = 0; i < n; ++i) {
            x[i] = logistic_activate(x[i]);
         }
      }else {
         for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], type);
         }
      }
   }



   /*
   ** 根据不同的激活函数求输入的梯度（导数）
   ** 输入： x    激活函数接收的输入值
   **       a    激活函数类型，包括的激活函数类型见activations.h中枚举类型ACTIVATION的定义
   ** 输出： 激活函数关于输入x的导数值
   */
   float gradient(float x, ActivationType type){
      switch(type){
         case ActivationType.LINEAR:
            return linear_gradient(x);
         case ActivationType.LOGISTIC:
            return logistic_gradient(x);
         case ActivationType.LOGGY:
            return loggy_gradient(x);
         case ActivationType.RELU:
            return relu_gradient(x);
         case ActivationType.ELU:
            return elu_gradient(x);
         case ActivationType.SELU:
            return selu_gradient(x);
         case ActivationType.RELIE:
            return relie_gradient(x);
         case ActivationType.RAMP:
            return ramp_gradient(x);
         case ActivationType.LEAKY:
            return leaky_gradient(x);
         case ActivationType.TANH:
            return tanh_gradient(x);
         case ActivationType.PLSE:
            return plse_gradient(x);
         case ActivationType.STAIR:
            return stair_gradient(x);
         case ActivationType.HARDTAN:
            return hardtan_gradient(x);
         case ActivationType.LHTAN:
            return lhtan_gradient(x);
      }
      return 0;
   }

   /**
   * 计算激活函数的导数
   * y=σ′(z^l) 即 (∂y/∂z)
   * x： σ(z^l)的函数值由一维数组组成。
   * n：数组长度
   * type:激活函数类型
   * delta:误差
   *这里直接利用输出值求激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，其关于输入的导数值都可以描述为输出值的函数表达式，
   *比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
   */
   void gradientArray(const float *x, const int n, const ActivationType a, float *delta){
      int i;
      #pragma omp parallel for
      for(i = 0; i < n; ++i){
         delta[i] *= gradient(x[i], a);
      }
   }

   //原型 gradient_array_normalize_channels_softmax activation.h activation.c
   void gradientArrayNormalizeChannelsSoftmax(float *x, const int n, int batch, int channels, int wh_step, float *delta){
      int size = n / channels;

      int i;
      #pragma omp parallel for
      for (i = 0; i < size; ++i) {
         int wh_i = i % wh_step;
         int b = i / wh_step;

         if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               float out = x[index];
               float d = delta[index];
               grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               float d = delta[index];
               d = d * grad;
               delta[index] = d;
            }
         }
      }
   }

   //原型 gradient_array_normalize_channels activation.h activation.c
   void gradientArrayNormalizeChannels(float *x, const int n, int batch, int channels, int wh_step, float *delta){
      int size = n / channels;

      int i;
      #pragma omp parallel for
      for (i = 0; i < size; ++i) {
         int wh_i = i % wh_step;
         int b = i / wh_step;

         if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               float out = x[index];
               float d = delta[index];
               grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
               const int index = wh_i + k * wh_step + b*wh_step*channels;
               if (x[index] > 0) {
                  float d = delta[index];
                  d = d * grad;
                  delta[index] = d;
               }
            }
         }
      }
   }

   // https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
   //原型 gradient_array_swish activations.h activations.c
   void gradientArraySwish(const float *x, const int n, const float * sigmoid, float * delta){
      int i;
      #pragma omp parallel for
      for (i = 0; i < n; ++i) {
         float swish = x[i];
         delta[i] *= swish + sigmoid[i]*(1 - swish);
      }
   }

   // https://github.com/digantamisra98/Mish
   //原型 gradient_array_mish activations.h activations.c
   void gradientArrayMish(const int n, const float * activation_input, float * delta){
      int i;
      #pragma omp parallel for
      for (i = 0; i < n; ++i) {
         const float MISH_THRESHOLD = 20.0f;

         // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
         // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
         float inp = activation_input[i];
         const float sp = softplus_activate(inp, MISH_THRESHOLD);
         const float grad_sp = 1 - exp(-sp);
         const float tsp = tanh(sp);
         const float grad_tsp = (1 - tsp*tsp) * grad_sp;
         const float grad = inp * grad_tsp + tsp;
         delta[i] *= grad;
      }
   }

   //原型 gradient_array_hard_mish activation.h activation.c
   void gradientArrayHardMish(const int n, const float * activation_input, float * delta){
       int i;
       #pragma omp parallel for
       for (i = 0; i < n; ++i) {
           float inp = activation_input[i];
           delta[i] *= hard_mish_yashas_grad(inp);
       }
   }


};


