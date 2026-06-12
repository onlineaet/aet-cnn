#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>

#include "mtcs/MtcsTool.h"

#include  "NNetwork.h"
#include "SoftMaxLayer.h"
#include "DataFactory.h"
#include "cnnmicro.h"
#include "DnnUtils.h"

/**
 * 如果分类是互斥的用softmax,如果每个分类中也包含其它分类
 * 就用多个logitic 逻辑分类或二元分类
 */

impl$ SoftMaxLayer{

   SoftMaxLayer(int batch, int w,int h,int c,float temperature,int spatial,
         int noloss,int inputs, int groups,tree *softMaxTree){
      a_assert(inputs%groups == 0);
      self->type = LayerType.SOFTMAX;
      self->batch = batch;
      self->groups = groups;
      self->inputs = inputs;
      self->outputs = inputs;
      setInputDimen(w,h,c);
      self->temperature=temperature;
      self->spatial=spatial;
      self->noloss=noloss;
      self->softMaxTree=softMaxTree;
      self->loss =DataFactory.getInstance()->createData(inputs,batch);
      outputData= DataFactory.getInstance()->createOutputData(inputs,batch);
      deltaData=DataFactory.getInstance()->createDeltaData(inputs,batch);
      self->cost = 0;
      fprintf(stderr, "softmax                                        %4d\n",  inputs);
   }

   //原型 softmax_x_ent_cpu blash.h blas.c
   //计算本层的误差，因为本层是输出层所以误差公式用δ = (∂C/∂a)*σ`(z)
   //C 函数定义为交叉熵函数 δ 推导的结果是 y-a 其中y是真实值 p是计算值。
   void softmaxXEntCpu(int n, float *pred, float *truth, float *delta, float *error){
      int i;
      for(i = 0; i < n; ++i){
         float t = truth[i];
         float p = pred[i];
         error[i] = (t) ? -logf(p) : 0;
         delta[i] = t-p;
      }
   }

   void softmax_cpuxx(float *input, int n, int batch, int batch_offset,
         int groups, int group_offset, int stride, float temp, float *output){
      int g, b;
      for(b = 0; b < batch; ++b)
         for(g = 0; g < groups; ++g)
            DnnUtils.softmax(input + b*batch_offset + g*group_offset, n,
                  temp, output + b*batch_offset + g*group_offset, stride);

   }

   void softmax(float *input, int n, float temp, float *output, int stride)
   {
       int i;
       float sum = 0;
       float largest = -FLT_MAX;
       for(i = 0; i < n; ++i){
           if(input[i*stride] > largest) largest = input[i*stride];
       }
       for(i = 0; i < n; ++i){
           float e = exp(input[i*stride]/temp - largest/temp);
           sum += e;
           output[i*stride] = e;
       }
       for(i = 0; i < n; ++i){
           output[i*stride] /= sum;
       }
   }


   void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
   {
       int g, b;
       for(b = 0; b < batch; ++b){
           for(g = 0; g < groups; ++g){
               softmax(input + b*batch_offset + g*group_offset, n, temp, output + b*batch_offset + g*group_offset, stride);
           }
       }
   }


   //作为输出层，用公式计算输出值 ，激活函数是linear f'(x)=1
   //交叉熵损失函数 C=-∑ylna y是真标签向量 one-hot编码 a =e^z/∑e^z z输入值
   void forwardCPU(NetworkState state){
      float *input = state.input->getDataArray();
      float *output = outputData->getDataArray();
      if(softMaxTree){
         int i;
         int count = 0;
         for (i = 0; i < softMaxTree->groups; ++i) {
            int group_size = softMaxTree->group_size[i];
            softmax_cpu(input + count, group_size, batch, inputs, 1, 0, 1, temperature, output + count);
            count += group_size;
         }
      } else {
         softmax_cpu(input, inputs/groups, batch, inputs, groups, inputs/groups, 1, temperature, output);
         //testoutput(output,outputData->getSize()*batch,"SoftMaxLayer forward的 CPU 输出数据 ",TRUE);
      }

      if(state.truth && !noloss){
         softmaxXEntCpu(batch*inputs, output, state.truth->getDataArray(), deltaData->getDataArray(), loss->getDataArray());
         cost = loss->sum();
      }
   }

   //原型 softmax_device_new_api blas_kernels.cu
   __device__ void softmax_device_new_api(float *input, int n, float temp, int stride, float *output){
       int i;
       float sum = 0;
       float largest = -INFINITY;
       for (i = 0; i < n; ++i) {
           int val = input[i*stride];
           largest = (val>largest) ? val : largest;
       }
       for (i = 0; i < n; ++i) {
           float e = expf(input[i*stride] / temp - largest / temp);
           sum += e;
           output[i*stride] = e;
       }
       for (i = 0; i < n; ++i) {
           output[i*stride] /= sum;
       }
   }

   //原型 softmax_tree_kernel blas_kernels.cu
   __global__ void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp,
         float *output, int groups, int *group_size, int *group_offset){
       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (id >= spatial*batch*groups)
          return;
       int s = id % spatial;
       id = id / spatial;
       int g = id % groups;
       int b = id / groups;
       int goff = group_offset[g] * spatial;
       int boff = b*stride;
       softmax_device_new_api(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
   }

   void softmax_tree_gpu(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier){
      // int *tree_groups_size = cuda_make_int_array_new_api(hier.group_size, hier.groups);
       int *tree_groups_size =MtcsMem.malloc(sizeof(int)*hier.groups,TRUE);
       MtcsMem.memcpy(tree_groups_size,hier.group_size,sizeof(int)*hier.groups,MtcsCpyKind.HOST2DEV);
       //int *tree_groups_offset = cuda_make_int_array_new_api(hier.group_offset, hier.groups);
       int *tree_groups_offset =MtcsMem.malloc(sizeof(int)*hier.groups,TRUE);
       MtcsMem.memcpy(tree_groups_offset,hier.group_offset,sizeof(int)*hier.groups,MtcsCpyKind.HOST2DEV);
       int num = spatial*batch*hier.groups;
       softmax_tree_kernel <<<MtcsTool.gridSize(num)/*!cuda_gridsize(num)*/, MTCS_BLOCK,
             0, MtcsTool.getStream()/*!get_cuda_stream()*/>>>
             (input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
       //CHECK_CUDA(cudaPeekAtLastError());
       MtcsMem.free/*!cuda_free((float *)tree_groups_size);*/(tree_groups_size);
       MtcsMem.free/*!cuda_free((float *)tree_groups_offset);*/(tree_groups_offset);
   }

   //原型 softmax_kernel_new_api blas_kernels.cu
   __global__ void softmax_kernel_new_api(float *input, int n, int batch, int batch_offset,
         int groups, int group_offset, int stride, float temp, float *output){
       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (id >= batch*groups)
          return;
       int b = id / groups;
       int g = id % groups;
       softmax_device_new_api(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
   }

   //原型 softmax_gpu_new_api blas.h blas_kernels.cu
   void softmax_gpu_new_api(float *input, int n, int batch, int batch_offset, int groups,
         int group_offset, int stride, float temp, float *output){
       softmax_kernel_new_api <<<MtcsTool.gridSize(batch*groups), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
             (input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
   }


   //原型 softmax_x_ent_kernel blas.h blas_kernels.cu
   __global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error){
       int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (i < n) {
           float t = truth[i];
           float p = pred[i];
           error[i] = (t) ? -logf(p) : 0;
           delta[i] = t - p;
       }
   }

   //原型 softmax_x_ent_gpu blas.h blas_kernels.cu
   void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error){
       softmax_x_ent_kernel <<<MtcsTool.gridSize(n), MTCS_BLOCK, 0, MtcsTool.getStream()>>>(n, pred, truth, delta, error);
   }

   //原型 mask_kernel_new_api blas_kernels.cu
   __global__ void mask_kernel_new_api(int n, float *x, float mask_num, float *mask, float val){
       int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (i < n && mask[i] == mask_num)
          x[i] = val;
   }


   //原型 mask_gpu_new_api blas.h blas_kernels.cu
   void mask_gpu_new_api(int n, float * X, float mask_num, float * mask, float val){
       mask_kernel_new_api <<<MtcsTool.gridSize(n), MTCS_BLOCK, 0, MtcsTool.getStream() >>>(n, X, mask_num, mask, val);
   }

   void testoutput(float *src,int size,char *explain,aboolean iscpu){
      int n=size>100?100:size;
      float *cpudata = src;
      if(!iscpu){
         cpudata=(float*)malloc(n*sizeof(float));
         MtcsMem.memcpy(cpudata,src,n*sizeof(float),MtcsCpyKind.DEV2HOST);
      }
      int i;
      for(i=0;i<n;i++)
         printf("%s is i:%d %f layer:%d\n",explain,i,cpudata[i],getOrderNumber());
      if(!iscpu)
      free(cpudata);
   }

   void forwardMTCS(NetworkState state){
      float *input=state.input->getDataArray();
      float *output=outputData->getDataArray();
      if(softMaxTree){
         softmax_tree_gpu(input, 1,batch, inputs, temperature, output, *softMaxTree);
      }else{
         int w =inputDimen.w;
         int h =inputDimen.h;
         int c = inputDimen.channels;
         if(spatial){
            softmax_gpu_new_api(input, c, batch*c, inputs/c, w*h, 1, w*h, 1, output);
         }else{
            /* compare
            float *inputCpu=malloc(inputs*batch*sizeof(float));
            MtcsMem.memcpy(inputCpu,state.input->getDataArray(),inputs*batch*sizeof(float),MtcsCpyKind.DEV2HOST);
            float *outputCpu=malloc(batch*outputs*sizeof(float));
            MtcsMem.memcpy(outputCpu,outputData->getDataArray(),batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
            softmax_cpu(inputCpu, inputs/groups, batch, inputs, groups, inputs/groups, 1, temperature, outputCpu);
            */
            softmax_gpu_new_api(input,inputs/groups, batch, inputs, groups, inputs/groups, 1, temperature, output);
            //testoutput(output,outputData->getSize()*batch,"SoftMaxLayer forward的 MTCS 输出数据 ",FALSE);

            /* compare
            float *compareCpu=malloc(batch*outputs*sizeof(float));
            MtcsMem.memcpy(compareCpu,outputData->getDataArray(),batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
            printf("SoftMaxLayer forward 比较\n");
            DnnUtils.compare(batch*outputs,compareCpu,outputCpu);
            free(inputCpu);
            free(outputCpu);
            free(compareCpu);
            */
         }
      }
      if(state.truth && !noloss){
         int n = batch*inputs;
         //testoutput(deltaData->getDataArray(),deltaData->getSize()*batch,"SoftMaxLayer forward的 MTCS delta数据 ",FALSE);

         softmax_x_ent_gpu(n, outputData->getDataArray(), state.truth->getDataArray(),
               deltaData->getDataArray(), loss->getDataArray());
         if(softMaxTree){
            mask_gpu_new_api(n, deltaData->getDataArray(), SECRET_NUM, state.truth->getDataArray(), 0);
            mask_gpu_new_api(n, loss->getDataArray(), SECRET_NUM, state.truth->getDataArray(), 0);
         }
         cost = loss->sum();
         //printf("SoftMaxLayer loss -- %f train:%d loss:%d\n",cost,state.train,loss->getSize());
      }
   }


   void forward(NetworkState state){
      if(devType==DeviceType.MTCS)
         forwardMTCS(state);
      else
         forwardCPU(state);
   }

   //上一层(隐藏层)的误差公式是δ^l = ((w^l+1 )^T δ^(l+1)) ⊙ σ′(z^l)
   //总是乘与它的下一层的误差，所以在这里把误差加入到上一层的误差中。这样上一层只用计算本层的σ′(z^l)
   void backward(NetworkState state){
      if(devType==DeviceType.MTCS){
         float lossScale=((NNetwork *)network)->loss_scale;
         /* compare
         float *deltaCPU =malloc(batch*inputs*sizeof(float));
         MtcsMem.memcpy(deltaCPU,state.delta->getDataArray(),batch*inputs*sizeof(float),MtcsCpyKind.DEV2HOST);
         */

         MtcsTool.axpy(batch*inputs,lossScale, deltaData->getDataArray(),state.delta->getDataArray());

         /*compare
         float *myDeltaCPU =malloc(batch*deltaData->getSize()*sizeof(float));
         MtcsMem.memcpy(myDeltaCPU,deltaData->getDataArray(),batch*deltaData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
         DnnUtils.axpy(batch*inputs,lossScale,myDeltaCPU,deltaCPU);
         float *compareCpu =malloc(batch*inputs*sizeof(float));
         MtcsMem.memcpy(compareCpu,state.delta->getDataArray(),batch*inputs*sizeof(float),MtcsCpyKind.DEV2HOST);
         printf("SoftMaxLayer backward 比较\n");
         DnnUtils.compare(batch*inputs, compareCpu,deltaCPU);
         free(deltaCPU);
         free(myDeltaCPU);
         free(compareCpu);
         */
      }else{
         DnnUtils.axpy(batch*inputs,1.0,deltaData->getDataArray(),state.delta->getDataArray());
      }
   }

   /**
   * 实现输出层的接口
   */
   float getCost(){
      return cost;
   }


};

