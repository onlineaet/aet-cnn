#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <bits/types.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include <aet/lang/AAssert.h>

#include <aet/lang/System.h>
#include <aet/time/Time.h>
#include "MtcsNormData.h"
#include "../cnnmicro.h"
#include "../DnnUtils.h"
#include "MtcsTool.h"

#define IS_ALIGNED(x, a)        (((x) & ((typeof(x))(a) - 1)) == 0)

#define TOLERANCE 1E-6
static int calcCount=0;
static aint64 scaletime=0;

typedef struct _TestData{
   float *outputsData;
   float *xs;
   float *xnorms;
}TestData;

impl$ MtcsNormData{

   MtcsNormData(int w,int h,int channels,int batch){
      self->w=w;
      self->h=h;
      self->channels=channels;
      self->batch=batch;
      self->mean = NULL;   //用于保存每个通道元素的平均值
      self->variance =NULL;//用于保存每个通道元素的方差
      self->rolling_mean = MtcsMem.calloc(channels*sizeof(float),TRUE);
      self->rolling_variance = MtcsMem.calloc(channels*sizeof(float),TRUE);
      self->x_norm=NULL;
      self->x=NULL;
   }

   public$ void createMean(){
      if(!mean){
         mean= MtcsMem.calloc(channels*sizeof(float),TRUE);
      }
   }

   //原型 fast_mean_kernel blas_kernels.cu
   __global__ void  calcMean(float *x, int batch, int filters, int spatial, float *mean){
       const int threads = MTCS_BLOCK;
       __shared__ float local[threads];

       int id = threadIdx.x;
       local[id] = 0;

       int filter = blockIdx.x;

       int i, j;
       for(j = 0; j < batch; ++j){
           for(i = 0; i < spatial; i += threads){
               int index = j*spatial*filters + filter*spatial + i + id;
               local[id] += (i+id < spatial) ? x[index] : 0;
           }
       }
       __syncthreads();
       //printf("fast_mean_kernel batch:%d filters:%d spatial:%d filter:%d id:%d threads:%d\n",batch,filters,spatial,filter,id,threads);
       if(id == 0){
           float mean_tmp = 0;
           for(i = 0; i < threads; ++i){
               mean_tmp += local[i];
           }
           mean_tmp /= spatial * batch;
           mean[filter] = mean_tmp;
       }
   }



   /**
   * 计算outputData的平均值
   */
   //原型 mean_cpu blas.h blas.c
   void calcMeanCPU(float *x, float *mean){
       int spatial=h*w;
       int filters = channels;
       float scale = 1./(batch * spatial);
       int i,j,k;
       for(i = 0; i < filters; ++i){
           mean[i] = 0;
           for(j = 0; j < batch; ++j){
               for(k = 0; k < spatial; ++k){
                   int index = j*filters*spatial + i*spatial + k;
                   mean[i] += x[index];
               }
           }
           mean[i] *= scale;
       }
   }


   //原型 fast_variance_kernel blas_kernels.cu
   __global__ void  calcVariance(float *x, float *mean, int batch, int filters, int spatial, float *variance){
      const int threads = MTCS_BLOCK;
      __shared__ float local[threads];

      int id = threadIdx.x;
      local[id] = 0;

      int filter = blockIdx.x;

      int i, j;
      for(j = 0; j < batch; ++j){
         for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
         }
      }
      __syncthreads();

      if(id == 0){
         float variance_tmp = 0;
         for(i = 0; i < threads; ++i){
            variance_tmp += local[i];
         }
         variance_tmp /= (spatial * batch);// -1);
         variance[filter] = variance_tmp;
      }
   }

   //原型 variance_cpu blas.h blas.c
   void calcVarianceCPU(float *x, float *mean,  float *variance){
      int spatial=h*w;
      int filters = channels;
      float scale = 1./(batch * spatial - 1);
      int i,j,k;
      for(i = 0; i < filters; ++i){
         variance[i] = 0;
         for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + i*spatial + k;
               variance[i] += pow((x[index] - mean[i]), 2);
            }
         }
         variance[i] *= scale;
      }
   }


   public$ void createVariance(){
      if(!variance){
         variance=MtcsMem.calloc(channels*sizeof(float),TRUE);
      }
   }

   public$ void createX(){
      if(!x){
         x=createData(w*h*channels);
         x_norm=createData(w*h*channels);
      }
   }

   float *createData(int elementSize){
      float *ret =(float*) MtcsMem.malloc(batch*elementSize*sizeof(float),TRUE);
      return ret;
   }

   void scal_cpus(int N, float ALPHA, float *X){
      int i;
      for(i = 0; i < N; ++i)
         X[i] *= ALPHA;
   }

   void axpy_cpus(int N, float ALPHA, float *X, float *Y){
      int i;
      for(i = 0; i < N; ++i)
         Y[i] += ALPHA*X[i];
   }

   /**
    * CPU用 0.9和0.1
    */
   void scalAxpyRollMeanAndRollVariance(){
      /* compare
      float *rollingMeanCPU = calloc(channels,sizeof(float));
      MtcsMem.memcpy(rollingMeanCPU,rolling_mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      */

      MtcsTool.scal(channels/*!N*/, .99/*!ALPHA*/,rolling_mean);

      /* compare
      printf("MtcsNormData scalAxpyRollMeanAndRollVariance 比较滚动均值\n");
      scal_cpus(channels,0.99,rollingMeanCPU);
      float *compareCPU = calloc(channels,sizeof(float));
      MtcsMem.memcpy(compareCPU,rolling_mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      DnnUtils.compare(channels,rollingMeanCPU,compareCPU);
      */

      MtcsTool.axpy(channels,0.01,mean,rolling_mean);

      /* compare
      float *cpuMean = calloc(channels,sizeof(float));
      MtcsMem.memcpy(cpuMean,mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      axpy_cpus(channels,0.01,cpuMean,rollingMeanCPU);
      MtcsMem.memcpy(compareCPU,rolling_mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      DnnUtils.compare(channels,rollingMeanCPU,compareCPU);
      free(rollingMeanCPU);
      free(compareCPU);
      free(cpuMean);
      */

      /* compare
      float *rollingVarianceCPU = calloc(channels,sizeof(float));
      MtcsMem.memcpy(rollingVarianceCPU,rolling_variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      */

      MtcsTool.scal(channels/*!N*/, .99/*!ALPHA*/,rolling_variance/*!X*/);

      /* compare
      scal_cpus(channels,0.99,rollingVarianceCPU);
      float *compareCPU1 = calloc(channels,sizeof(float));
      MtcsMem.memcpy(compareCPU1,rolling_variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsNormData scalAxpyRollMeanAndRollVariance 比较滚动方差\n");
      DnnUtils.compare(channels,compareCPU1,rollingVarianceCPU);
      */

      //采用乘加方式复制variance到rolling_variance
      MtcsTool.axpy(channels,0.01,variance,rolling_variance);

      /* compare
      float *cpuVariance = calloc(channels,sizeof(float));
      MtcsMem.memcpy(cpuVariance,variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      axpy_cpus(channels,0.01,cpuVariance,rollingVarianceCPU);
      MtcsMem.memcpy(compareCPU1,rolling_variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      DnnUtils.compare(channels,compareCPU1,rollingVarianceCPU);
      free(rollingVarianceCPU);
      free(cpuVariance);
      free(compareCPU1);
      */
   }


   /**
   * 计算outputData的均值和方差
   */
   void calcMeanAndVariance(NData *outputData,aboolean merge){
      float *outputDataArray=outputData->getDataArray();
      int outputs=outputData->getSize();
      int batch=outputData->getBatch();
      int width=outputData->getWidth();
      int height=outputData->getHeight();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      a_assert(outputs==width*height*channels);
      int spatial=height*width;
      calcMean <<<channels, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(outputDataArray, batch, channels, spatial, mean);

      /* compare
      float *cpumean= calloc(channels,sizeof(float));
      float *cpuOut = malloc(batch*outputs*sizeof(float));
      MtcsMem.memcpy(cpuOut,outputDataArray,batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
      calcMeanCPU(cpuOut,cpumean);
      float *compareCPU= calloc(channels,sizeof(float));
      MtcsMem.memcpy(compareCPU,mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsNormData 比较mean\n");
      DnnUtils.compare(channels,cpumean,compareCPU);
      free(cpumean);
      free(cpuOut);
      free(compareCPU);
      */

      calcVariance<<<channels, MTCS_BLOCK, 0,  MtcsTool.getStream()>>>(outputDataArray, mean, batch, channels, spatial, variance);

      /* compare
      float *cpuvariance= calloc(channels,sizeof(float));
      float *cpuOut1 = malloc(batch*outputs*sizeof(float));
      MtcsMem.memcpy(cpuOut1,outputDataArray,batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpumean1 = calloc(channels,sizeof(float));
      MtcsMem.memcpy(cpumean1,mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *compareVariance = calloc(channels,sizeof(float));
      MtcsMem.memcpy(compareVariance,variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      calcVarianceCPU(cpuOut1,cpumean1,cpuvariance);
      DnnUtils.compare(channels,cpuvariance,compareVariance);
      printf("MtcsNormData 比较方差 variance\n");
      free(cpuvariance);
      free(cpuOut1);
      free(cpumean1);
      free(compareVariance);
      */
   }

   /**
    * 复制ouputData的数据到x
    * 覆盖父类的方法
    */
   void copyToX(NData *outputData){
      if(self->x==NULL){
         self->x=createData(w*h*channels);
      }
      float *outputDataArray=outputData->getDataArray();
      //复制outputDataArray到x
      int outputs=outputData->getSize();
      MtcsTool.simpleCopy(outputs*batch,outputDataArray,x);
      /* compare
      printf("MtcsNormData copyToX 比较x与outputData是否相同\n");
      DnnUtils.compare(batch,outputs,outputDataArray,x);
      */
   }

   void copyToXNorm(NData *outputData){
      if(self->x_norm==NULL){
         self->x_norm=createData(w*h*channels);
      }
      float *outputDataArray=outputData->getDataArray();
      //复制outputDataArray到x
      int outputs=outputData->getSize();
      MtcsTool.simpleCopy(outputs*batch,outputDataArray,x_norm);

      /*
      printf("比较x_norm与outputData是否相同\n");
      DnnUtils.compare(batch,outputs,outputDataArray,x_norm);
      */
   }

    /**
     * 比normalizeKernelByRoll方法多一个复制归一化的output到x_norm
     * sqrtVariance 事先算好的平方根方差的倒数，
     * 用倒数可以消除除法  value = (value - mean[f]) / sqrtVariance[f];
     */
   __global__ void normalizeKernel(int totalSize, float *outputDataArray,float *xs,float *xnorms,
         float *mean,float *variance,int filters, int spatial,float *scales,float *biases){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index >= totalSize)
          return;
       int f = (index / spatial) % filters;
       float value = outputDataArray[index];
       xs[index]=value;//在正则化前复制输出数据到x,否则需要在之前备份outputDataArray到x
       value= (value - mean[f]) / (sqrtf(variance[f] + .00001f));
       xnorms[index]=value;
       outputDataArray[index] = value*scales[f]+biases[f];
   }

   __global__ void normalize_kernel(int N, float *x, float *mean, float *variance,
         int batch, int filters, int spatial){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index >= N)
          return;
       int f = (index / spatial) % filters;
       x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + .00001f));
   }

   TestData *createTestData(NData *outputData){
      float *outputDataArray=outputData->getDataArray();
      int size = outputData->getSize();
      float *outputsCPU=malloc(batch*size*sizeof(float));
      float *xs=malloc(batch*size*sizeof(float*));
      float *xnorms=malloc(batch*size*sizeof(float*));
      int spatial=h*w;
      int outputs=channels * spatial;
      printf("createTestData --- %d %d\n",outputs,outputData->getSize());
      MtcsMem.memcpy(outputsCPU,outputDataArray,batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
      TestData *data=malloc(sizeof(TestData));
      data->outputsData=outputsCPU;
      data->xs=xs;
      data->xnorms=xnorms;
      return data;
   }

   private$ TestData *normalizeCPU(float *outputDataArray,int size,float *scales,float *biases,float *cpuMean,float *cpuVariance){
      int spatial=h*w;
      float *xs=malloc(batch*size*sizeof(float*));
      float *xnorms=malloc(batch*size*sizeof(float*));
      int b, f, i,j;
      for(b = 0; b < batch; ++b){
         float *outputData=outputDataArray+b*size;
         float *xnorm=xnorms+b*size;
         float *xc=xs+b*size;
         float value=0;
         for(f = 0; f < channels; ++f){
            float *data=outputData+f*spatial;
            float *xnorm1=xnorm+f*spatial;
            float *xdata=xc+f*spatial;
            for(i = 0; i < spatial; i++){
               value =data[i];
               xdata[i]=value;
               value= (value - cpuMean[f]) / (sqrtf(cpuVariance[f] + .00001f));

               if(isnan(value)){
                  printf("MtcsNormData.c normalizeCPU 无效数据\n");
                  exit(0);
               }
               xnorm1[i]=value;
               data[i] = value*scales[f]+biases[f];
            }
         }
      }
      TestData *data=malloc(sizeof(TestData));
      data->outputsData=outputDataArray;
      data->xs=xs;
      data->xnorms=xnorms;
      return data;
   }

   void compare(float *outputDataArray,TestData *testData){
      int size=channels*h*w;
      float *cpuOut=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuOut,outputDataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);

      float *cpuX=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuX,x,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);

      float *cpuXNorm=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuXNorm,x_norm,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);

      DnnUtils.compare(batch*size,cpuOut,testData->outputsData);
      DnnUtils.compare(batch*size,cpuX,testData->xs);
      DnnUtils.compare(batch*size,cpuXNorm,testData->xnorms);
      free(cpuOut);
      free(cpuX);
      free(cpuXNorm);

      free(testData->outputsData);
      free(testData->xs);
      free(testData->xnorms);
      free(testData);
   }

    //在darknet-alex中是分成调用5个方法完，现在合并在一个
//   copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
//   normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
//   copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
//
//   scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
//   add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
   void normalize(NData *outputData,float *scales,float *biases){
      float *outputDataArray=outputData->getDataArray();
      int batch=outputData->getBatch();
      int width=outputData->getWidth();
      int height=outputData->getHeight();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      /* compare
      int size = outputData->getSize();
      float *cpuOut=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuOut,outputDataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuScales=malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuScales,scales,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuBiases=malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuBiases,biases,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuMean =malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuMean,mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuVariance =malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuVariance,variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      TestData *testData=normalizeCPU(cpuOut,size,cpuScales,cpuBiases,cpuMean,cpuVariance);
      */

      int spatial=h*w;
      const int totalSize = batch * channels * spatial;
      const int num_blocks =MtcsTool.getNumberOfBlocks(totalSize, MTCS_BLOCK);
      normalizeKernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
            (totalSize, outputDataArray, x,x_norm,mean, variance, channels, spatial,scales,biases);

      /*compare
      printf("MtcsNormData 比较output x xnorm数据\n");
      compare(outputDataArray,testData);
      free(cpuScales);
      free(cpuBiases);
      free(cpuMean);
      free(cpuVariance);
      */
   }

   __global__ void normalizeKernelByRoll(int totalSize, float *outputDataArray,float *xs,
         float *rollMean,float *rollVariance,int filters, int spatial,float *scales,float *biases){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index >= totalSize)
          return;
       int f = (index / spatial) % filters;
       float value = outputDataArray[index];
       xs[index]=value;//在正则化前复制输出数据到x,否则需要在之前备份outputDataArray到x
       value = (value - rollMean[f])/ (sqrtf(rollVariance[f] + .00001f));
       outputDataArray[index] = value*scales[f]+biases[f];
   }

   void normalizeByRoll(NData *outputData,float *scales,float *biases){
      float *outputDataArray=outputData->getDataArray();
      int batch=outputData->getBatch();
      int width=outputData->getWidth();
      int height=outputData->getHeight();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      int spatial=h*w;
      const int totalSize = batch * channels * spatial;
      const int num_blocks =MtcsTool.getNumberOfBlocks(totalSize, MTCS_BLOCK);
      normalizeKernelByRoll <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
            (totalSize, outputDataArray, x,rolling_mean, rolling_variance, channels, spatial,scales,biases);
   }

   __global__ void inverse_variance_kernel(int size, float *src, float *dst, float epsilon){
       int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index < size)
           dst[index] = 1.0f / sqrtf(src[index] + epsilon);
   }
   /**
    * 负方差
    * 原型 inverse_variance_ongpu blas.h blas_kernels.cu
    */
   void inverseVariance(){
       const int num_blocks = channels / MTCS_BLOCK + 1;
       inverse_variance_kernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(channels, rolling_variance, variance, 0.00001);
   }

   __global__ void  fast_v_cbn_kernel(const float *outputs, float *mean, int batch, int filters,
         int spatial, int minibatch_index, int max_minibatch_index, float *m_avg, float *v_avg, float *variance,
       const float alpha, float *rolling_mean_gpu, float *rolling_variance_gpu, int inverse_variance, float epsilon){
      const int threads = MTCS_BLOCK;
      __shared__ float local[threads];

      int id = threadIdx.x;
      local[id] = 0;

      int filter = blockIdx.x;

      int i, j;
      for (j = 0; j < batch; ++j) {
         for (i = 0; i < spatial; i += threads) {
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i + id < spatial) ? powf(outputs[index], 2) : 0;
         }
      }
      __syncthreads();

      if (id == 0) {
         float v_tmp = 0;
         v_tmp = 0;
         for (i = 0; i < threads; ++i) {
            v_tmp += local[i];
         }
         v_tmp /= (spatial * batch - 1);

         v_tmp = fmax(v_tmp, powf(mean[filter], 2));
         const float alpha_cbn = 1.0f / minibatch_index;

         m_avg[filter] = alpha_cbn * mean[filter] + (1 - alpha_cbn) * m_avg[filter];
         mean[filter] = m_avg[filter];

         v_avg[filter] = alpha_cbn * v_tmp + (1 - alpha_cbn) * v_avg[filter];

         float variance_tmp = fmax(0.0f, v_avg[filter] - powf(m_avg[filter], 2));
         if (inverse_variance)
            variance[filter] = 1.0f / sqrtf(variance_tmp + epsilon);
         else
            variance[filter] = variance_tmp;

         {
            if(rolling_mean_gpu)
               rolling_mean_gpu[filter] = alpha * mean[filter] + (1 - alpha) * rolling_mean_gpu[filter];

            if(rolling_variance_gpu)
               rolling_variance_gpu[filter] = alpha * variance_tmp + (1 - alpha) * rolling_variance_gpu[filter];
         }
      }
   }

   //原型 fast_v_cbn_gpu blas.h blas_kernels.cu
   void fastVcbn(const float *outputs, int minibatch_index, int max_minibatch_index,
            float *m_avg, float *v_avg, const float alpha, int inverse_variance, float epsilon){
      int spatial=w*h;
      fast_v_cbn_kernel <<<channels, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(outputs, mean, batch,
         channels, spatial, minibatch_index, max_minibatch_index, m_avg, v_avg,
         variance, alpha, rolling_mean, rolling_variance, inverse_variance, epsilon);
   }

   __global__ void normalize_scale_bias_kernel(int N, float *outputs, float *mean,float *variance, float *scales,
         float *biases,  int filters, int spatial, int inverse_variance, float epsilon){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index >= N)
          return;
       int f = (index / spatial) % filters;

       float val = 0;
       if(inverse_variance)
          val = (outputs[index] - mean[f]) * variance[f];
       else
          val = (outputs[index] - mean[f]) / (sqrtf(variance[f] + epsilon));
       val *= scales[f];
       val += biases[f];

       if (!isnan(val) && !isinf(val))
          outputs[index] = val;
   }

   //原型 normalize_scale_bias_gpu blas.h blas_kernels.cu
   void normalize(float *outputs, float *scales, float *biases,  int inverse_variance, float epsilon){
      int spatial=w*h;
      int filters = channels;
      const int current_size = batch * filters * spatial;
      const int num_blocks = MtcsTool.getNumberOfBlocks(current_size, MTCS_BLOCK);
      normalize_scale_bias_kernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
               (current_size, outputs, mean, variance, scales, biases, filters, spatial, inverse_variance, epsilon);
   }

   ~MtcsNormData(){
      if(x){
         MtcsMem.free(x);
         x=NULL;
      }
      if(x_norm){
         MtcsMem.free(x_norm);
         x_norm=NULL;
      }
      if(mean){
         MtcsMem.free(mean);
         mean=NULL;
      }
      if(variance){
         MtcsMem.free(variance);
         variance=NULL;
      }

      if(rolling_mean){
         MtcsMem.free(rolling_mean);
         rolling_mean=NULL;
      }

      if(rolling_variance){
         MtcsMem.free(rolling_variance);
         rolling_variance=NULL;
      }
   }


};

