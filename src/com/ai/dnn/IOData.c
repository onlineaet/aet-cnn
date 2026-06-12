#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <aet/time/Time.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/System.h>
#include "IOData.h"
#include "DnnUtils.h"

impl$ IOData{

   public$ IOData(int w,int h,int channels,int batch){
      super$(w,h,channels,batch);
   }

   public$ IOData(int size,int batch){
      super$(size,batch);
   }
   /**
    * 实现InputData接口
    */
   public$ void setImageData(float *imageData){
      memcpy((char*)dataArray,imageData,batch*size*sizeof(float));
   }


   /**
   * 把自身的数据归一化。
   * 归一化公式来自z-score标准化
   * A=v1,v2,v3,...,vn
   * 公式 v'=(vi-A的平均值)/σa(A的均方差)
   * 调用sqrt运行时间是70ms,如是提前做好sqrt的数组，变成20ms 提升3倍多速度。
   * avx的开平方根函数：_mm256_sqrt_ps
   */

   /**
   * 加偏置到每个输出数据
   */
   //原型 add_bias blas.h convolutional_layer.c
   void addBias(float *biases){
      float *output=dataArray;
      int spital = w*h;
      int n=channels;
      int i,j,b;
      for(b = 0; b < batch; ++b){
         for(i = 0; i < n; ++i){
            for(j = 0; j < spital; ++j){
               output[(b*n + i)*spital + j] += biases[i];
            }
         }
      }
   }

   void testprint(){
      int b,i;
      for(b = 0; b < batch; ++b){
         float *data=dataArray+b*size;
         for(i = 0; i < size; ++i)
            printf("输出数据的值 batch:%d i:%d size:%d %f\n",b,i,size,data[i]);
      }
   }

   //原型 activate_array_cpu_custom gemm.h gemm.c
   void activate(const ActivationType type){
      if (type == ActivationType.LINEAR){
         ;
      }else
         Activation.activate(dataArray,batch*size,type);
   }

   //原型 activate_array_swish activations.h activations.c
   void activateArraySwish(OutputData * activation_input){
       int i;
       int n=batch*size;
       float *x=dataArray;
       float *output=dataArray;
       float *output_sigmoid =  activation_input->getDataArray();
       #pragma omp parallel for
       for (i = 0; i < n; ++i) {
           float x_val = x[i];
           float sigmoid = logistic_activate(x_val);
           output_sigmoid[i] = sigmoid;
           output[i] = x_val * sigmoid;
       }
   }

   // https://github.com/digantamisra98/Mish
   //原型 activate_array_mish activations.h activations.c
   void activateArrayMish(OutputData * activation_input){
      int i;
      int n=batch*size;
      float *x=dataArray;
      float *output=dataArray;
      float *activationInput =  activation_input->getDataArray();
      const float MISH_THRESHOLD = 20;
      #pragma omp parallel for
      for (i = 0; i < n; ++i) {
         float x_val = x[i];
         activationInput[i] = x_val;    // store value before activation
         output[i] = x_val * tanh_activate( softplus_activate(x_val, MISH_THRESHOLD) );
      }
   }

   private$ inline float hard_mish_yashas(float x){
       if (x > 0)
           return x;
       if (x > -2)
           return x * x / 2 + x;
       return 0;
   }

   //原型 activate_array_hard_mish activations.h activations.c
   void activateArrayHardMish(OutputData *activation_input){
      int i;
      int n=batch*size;
      float *x=dataArray;
      float *output=dataArray;
      float *activationInput =  activation_input->getDataArray();
      #pragma omp parallel for
      for (i = 0; i < n; ++i) {
         float x_val = x[i];
         activationInput[i] = x_val;    // store value before activation
         output[i] = hard_mish_yashas(x_val);
      }
   }

   //原型 activate_array_normalize_channels activations.h activations.c
   void activateArrayNormalizeChannels(){
      float *output=dataArray;
      float *x=dataArray;
      int n=batch*size;
      int spital=n/channels;
      int wh_step = w*h;
      int i;
      #pragma omp parallel for
      for (i = 0; i < spital; ++i) {
         int wh_i = i % wh_step;
         int b = i / wh_step;

         const float eps = 0.0001;
         if (i < spital) {
            float sum = eps;
            int k;
            for (k = 0; k < channels; ++k) {
               float val = x[wh_i + k * wh_step + b*wh_step*channels];
               if (val > 0)
                  sum += val;
            }
            for (k = 0; k < channels; ++k) {
               float val = x[wh_i + k * wh_step + b*wh_step*channels];
               if (val > 0)
                  val = val / sum;
               else
                  val = 0;
               output[wh_i + k * wh_step + b*wh_step*channels] = val;
            }
         }
      }
   }


   //原型 activate_array_normalize_channels_softmax activations.h activations.c
   void activateArrayNormalizeChannelsSoftmax(int use_max_val){
      float *output=dataArray;
      float *x=dataArray;
      int n=batch*size;
      int spital=n/channels;
      int wh_step = w*h;
      int i;
      #pragma omp parallel for
      for (i = 0; i < spital; ++i) {
         int wh_i = i % wh_step;
         int b = i / wh_step;

         const float eps = 0.0001;
         if (i < spital) {
            float sum = eps;
            float max_val = -FLT_MAX;
            int k;
            if (use_max_val) {
               for (k = 0; k < channels; ++k) {
                  float val = x[wh_i + k * wh_step + b*wh_step*channels];
                  if (val > max_val || k == 0)
                     max_val = val;
               }
            } else
               max_val = 0;

            for (k = 0; k < channels; ++k) {
               float val = x[wh_i + k * wh_step + b*wh_step*channels];
               sum += expf(val - max_val);
            }
            for (k = 0; k < channels; ++k) {
               float val = x[wh_i + k * wh_step + b*wh_step*channels];
               val = expf(val - max_val) / sum;
               output[wh_i + k * wh_step + b*wh_step*channels] = val;
            }
         }
      }
   }

   //原型 binarize_cpu convolutional_layer.c
   //实现InputDta接口
   void binarize(InputData *binary){
      if(size!=((IOData*)binary)->getSize()){
         a_error("binarize 大小不符:%d %d\n",size,((IOData*)binary)->getSize());
      }
      if(size!=((IOData*)binary)->getSize()){
         a_error("binarize batch不符:%d %d\n",batch,((IOData*)binary)->getBatch());
      }

      int i;
      int n=batch*size;
      float *dest=binary->getDataArray();
      for(i = 0; i < n; ++i){
         dest[i] = (dataArray[i] > 0) ? 1 : -1;
      }
   }

};

