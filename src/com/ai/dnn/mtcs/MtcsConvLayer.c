#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <aet/lang/System.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include <aet/time/Time.h>
#include <aet/lang/AAssert.h>

#include "MtcsConvLayer.h"
#include "../DnnUtils.h"
#include "../Activation.h"
#include "../DstbCompute.h"
#include "../DataFactory.h"
#include "../cnnmicro.h"
#include "MtcsTool.h"
#include "MtcsActivation.h"
#include "MtcsIOData.h"
#include "MtcsConvKernel.h"

impl$ MtcsConvLayer{

   MtcsConvLayer(int batch, int steps, int h, int w, int c,
            int n, int groups, int size, int stride_x, int stride_y, int dilation,
            int padding, ActivationType activation, int batch_normalize, int binary,
            int xnor, int adam, int use_bin_output, int index, int antialiasing,
            ConvolutionalLayer *share_layer, int assisted_excitation, int deform, int train){
      super$(batch,steps,h,w,c,n,groups,size,stride_x,stride_y,dilation,
            padding,activation,batch_normalize,binary,
            xnor,adam,use_bin_output,index,antialiasing,
            share_layer,assisted_excitation,deform,train);
      im2ColCPU = new$ Im2Col();
      /*!l.input_antialiasing_gpu = cuda_make_array(NULL, l.batch*l.outputs);*/
      input_antialiasing=DataFactory.getInstance()->createInputData(outputs,batch);
      m_cbn_avg =  (float*)MtcsMem.calloc(n*sizeof(float),TRUE);
      v_cbn_avg =  (float*)MtcsMem.calloc(n*sizeof(float),TRUE);
      devType=DeviceType.MTCS;
   }

   //原型 assisted_excitation_forward convolutional_layer.h convolutional_layer.c
   public$ void assistedExcitationForward(NetworkState state){
      a_error("mtcs不未实现 assistedExcitationForward\n");
   }

   void testoutput(float *src,char *explain){
      int n=100;
      float *cpudata=(float*)malloc(n*sizeof(float));
      MtcsMem.memcpy(cpudata,src,n*sizeof(float),MtcsCpyKind.DEV2HOST);
      int i;
      for(i=0;i<n;i++)
         printf("%s is i:%d %f layer:%d\n",explain,i,cpudata[i],getOrderNumber());
      free(cpudata);
   }

  //原型 forward_convolutional_layer convolutional_layer.h convolutional_layer.c
   void forward(NetworkState state){
      int out_h = getOutHeight/*!convolutional_out_height*/();
      int out_w = getOutWidth/*!convolutional_out_width*/();
      int i, j;
      NNetwork *net=(NNetwork *)network;
      ((IOData*)outputData)->setZero();/*!fill_cpu(l.outputs*l.batch, 0, l.output, 1);*/
      float *weights=weightData->getWeights();
      int m = filters / groups;
      int k = ksize*ksize*inputDimen.channels / groups;
      int n = out_h*out_w;
      int channels=inputDimen.channels;
      int lw=inputDimen.w;
      int lh=inputDimen.h;
      static int u = 0;
      u++;
//      printf("convolutional foward 00 m:%d k:%d n:%d channels:%d lw:%d lh:%d groups:%d\
//             batch_normalize:%d ksize:%d pad:%d dilation:%d sx:%d sy:%d\n",
//            m,k,n,channels,lw,lh,groups,batch_normalize,ksize,pad,dilation,stride_x,stride_y);
      float *output=outputData->getDataArray();
      float *input=state.input->getDataArray();
      for(i = 0; i < batch; ++i){
         for (j = 0; j < groups; ++j){

            float *im = input + (i*groups + j)*channels/groups*lh*lw;
            float *a = weights +j*nweights / groups;
            float *b = state.workspace;
            float *c =output +(i*groups + j)*n*m;/*!l.output +(i*groups + j)*n*m;*/
            a_assert(n*m==outputs);
            if (xnor && align_bit_weights && !state.train && stride_x == stride_y){
               ;
            }else{

               /*!float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;*/
               if (ksize == 1 && stride == 1 && dilation == 1) {
                  b = im;
               }else {
                 // printf("输入数据存入到工作空间作为矩阵 b i:%d c:%d lh:%d lw:%d ksize:%d,stride_x:%d stride_y:%d pad:%d dilation:%d space:%p c:%p\n",
                 //       i,inputDimen.channels,inputDimen.h, inputDimen.w,ksize,stride_x,stride_y,pad,dilation,b,c);
                  im2Col->im2col/*!im2col_cpu_ext*/(im,   // input
                        inputDimen.channels / groups,     // input channels
                        inputDimen.h, inputDimen.w,           // input size (h, w)
                        ksize, ksize,     // kernel size (h, w)
                        pad * dilation, pad * dilation,       // padding (h, w)
                        stride_y, stride_x, // stride (h, w)
                        dilation, dilation, // dilation (h, w)
                        state.workspace);                 // output

                  /*compare
                  int im2colSize=outputDimen.h*outputDimen.w*ksize*ksize*inputDimen.channels;
                  float *cpudata=(float*)malloc(sizeof(float)*im2colSize);
                  float *cpuIm = (float*)malloc(sizeof(float)*channels*lw*lh);
                  MtcsMem.memcpy(cpuIm,im,channels*lw*lh*sizeof(float),MtcsCpyKind.DEV2HOST);
                  im2ColCPU->im2col(cpuIm,   // input
                                   inputDimen.channels / groups,     // input channels
                                   inputDimen.h, inputDimen.w,           // input size (h, w)
                                   ksize, ksize,     // kernel size (h, w)
                                   pad * dilation, pad * dilation,       // padding (h, w)
                                   stride_y, stride_x, // stride (h, w)
                                   dilation, dilation, // dilation (h, w)
                                   cpudata);                 // output
                  printf("MtcsConvLayer.c forward 比较 im2col layer:%d\n",getOrderNumber());
                  float *comparedata=(float*)malloc(im2colSize*sizeof(float));
                  MtcsMem.memcpy(comparedata,state.workspace,im2colSize*sizeof(float),MtcsCpyKind.DEV2HOST);
                  DnnUtils.compare(im2colSize,comparedata,cpudata);
                  free(cpudata);
                  free(cpuIm);
                  free(comparedata);
                  */
               }
               //printf("MtcsConvLayer.c forward 11 layer:%d \n",getOrderNumber());
               gemm->gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
               //printf("MtcsConvLayer  forward im2col i:%d m:%d n:%d k:%d %lli\n",i,m,n,k,Time.monotonic()-startTime);
               /* compare
               int im2colSize=outputDimen.h*outputDimen.w*ksize*ksize*inputDimen.channels;
               float *cpuA=(float*)calloc(nweights,sizeof(float));
               float *cpuB=(float*)calloc(im2colSize,sizeof(float));
               float *cpuC=(float*)calloc(outputs,sizeof(float));
               MtcsMem.memcpy(cpuA,a,sizeof(float)*nweights,MtcsCpyKind.DEV2HOST);
               MtcsMem.memcpy(cpuB,b,sizeof(float)*im2colSize,MtcsCpyKind.DEV2HOST);
               float *compareC=(float*)calloc(outputs,sizeof(float));
               MtcsMem.memcpy(compareC,c,sizeof(float)*outputs,MtcsCpyKind.DEV2HOST);
               gemm(0, 0, m, n, k, 1, cpuA, k, cpuB, n, 1, cpuC, n);
               printf("MtcsConvLayer.c forward 比较卷积的矩阵乘结果 layer:%d n*k:%d m*n:%d outputs:%d im2colSize:%d\n",getOrderNumber(),n*k,m*n,outputs,im2colSize);
               DnnUtils.compare(outputs,compareC,cpuC);
               free(cpuC);
               free(cpuA);
               free(cpuB);
               free(compareC);
               */
            }
         }
      }
      //printf("MtcsConvLayer   forward im2col和gemm i:%d m:%d n:%d k:%d 平均:%lli %lli\n",i,m,n,k,totaltime1/batch,totaltime2/batch);
      //testoutput(outputData->getDataArray(),"Output数据 ");
      if(batch_normalize){
         forwardNorm/*!forward_batchnorm_layer*/(state);
      }else{
         /*! add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);*/
         outputData->addBias(biasData->getBias());
      }

      if (activation == ActivationType.SWISH){
         /*! activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);*/
         outputData->activateArraySwish(activation_input);
      }else if (activation == ActivationType.MISH)
         /*!activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);*/
         outputData->activateArrayMish(activation_input);
      else if (activation == ActivationType.HARD_MISH)
         /*!activate_array_hard_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);*/
         outputData->activateArrayHardMish(activation_input);
      else if (activation == ActivationType.NORM_CHAN)
         /*!activate_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output);*/
         outputData->activateArrayNormalizeChannels();
      else if (activation == ActivationType.NORM_CHAN_SOFTMAX)
         /*!activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 0);*/
         outputData->activateArrayNormalizeChannelsSoftmax(0);
      else if (activation == ActivationType.NORM_CHAN_SOFTMAX_MAXVAL)
         /*!activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 1);*/
         outputData->activateArrayNormalizeChannelsSoftmax(1);
      else{
         outputData->activate(activation);/*!activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);*/
      }
      if(binary || xnor)
         swapBinary/*!swap_binary*/();
      if (net->tryFixNan()) {
          MtcsTool.fixNanAndInf/*!fix_nan_and_inf*/(outputData->getDataArray(),outputs*batch);
      }
      if(assisted_excitation && state.train)
         assistedExcitationForward/*!assisted_excitation_forward*/(state);

      if (antialiasing) {
         NetworkState s = { 0 };
         s.train = state.train;
         s.workspace = state.workspace;
         s.input =(InputData*)((IOData*)outputData);
         input_layer->forward(s);/*!forward_convolutional_layer(*(l.input_layer), s);*/
         //simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
         ((IOData*)outputData)->copy((IOData*)input_antialiasing);
         //simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.input_layer->output_gpu, l.output_gpu);
         ((IOData*)input_layer->outputData)->copy((IOData*)outputData);
      }

      if (coordconv) {
         int totalSize = outputs*batch;
         const int num_blocks = MtcsTool.getNumberOfBlocks/*!get_number_of_blocks*/(totalSize, MTCS_BLOCK);
         coordConvKernel/*!coord_conv_gpu*/<<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
               (outputData->getDataArray(),outputDimen.w,outputDimen.h,outputDimen.channels, batch, 0);
      }
   }

   //原型 coord_conv_kernel blas_kernels.cu
   __global__ void coordConvKernel(float *dst, int w, int h, int chan, int batch, int type){
      int i = blockIdx.x*blockDim.x + threadIdx.x;

      const int x = i % w;
      i = i / w;
      const int y = i % h;
      i = i / h;
      const int c = i % chan;

      if (type == 0) {
         if (c == 0) {
            const float x_val = (2.0f * x) / w - 1.0f;  // [-1; 1)
            dst[i] = x_val; // x - coord
         } else if (c == 1) {
            const float y_val = (2.0f * y) / h - 1.0f;  // [-1; 1)
            dst[i] = y_val; // y - coord
         } else if (c == 2) {
            const float x_val = (2.0f * x) / w - 1.0f;  // [-1; 1)
            const float y_val = (2.0f * y) / h - 1.0f;  // [-1; 1)
            const float rad_val = sqrtf(x_val*x_val + y_val*y_val);  // [0; 1.414)
            dst[i] = rad_val; // rad - coord
         }
      } else if (type == 1) {
         if (c >= 0 && c <= 2) {
            dst[i] = 0;
         }
      }
   }

   //原型 backward_convolutional_layer_gpu convolutional_layer.h  convolutional_layer.c
   void backward(NetworkState state){
      NNetwork *net=(NNetwork *)network;
      if (coordconv) {
         /*!coord_conv_gpu(l.delta_gpu, l.outputs*l.batch, l.out_w, l.out_h, l.out_c, l.batch, 1);*/
         int totalSize = outputs*batch;
         const int num_blocks = MtcsTool.getNumberOfBlocks/*!get_number_of_blocks*/(totalSize, MTCS_BLOCK);
         coordConvKernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>> (deltaData->getDataArray(),
         outputDimen.w,outputDimen.h,outputDimen.channels,batch,1);
      }

      if (antialiasing) {
         NetworkState s = { 0 };
         s.train = state.train;
         s.workspace = state.workspace;
         s.delta = deltaData;  // s.delta will be returned to l.delta_gpu
         s.input = input_antialiasing;
         //if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
         /*!simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.delta_gpu, l.input_layer->delta_gpu);*/
         deltaData->copy(input_layer->deltaData);
         /*!backward_convolutional_layer_gpu(*(l.input_layer), s);*/
         input_layer->backward(s);
         //simple_copy_ongpu(l.outputs*l.batch, l.input_antialiasing_gpu, l.output_gpu);
         ((IOData*)input_antialiasing)->copy((IOData*)outputData);
      }

      if(((NNetwork*)network)->tryFixNan()/*!state.net.try_fix_nan*/)
         /*!constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);*/
         MtcsTool.constrain(outputs*batch,1.0,deltaData->getDataArray());


      if (activation == ActivationType.SWISH)
         /*!gradient_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);*/
         MtcsActivation.gradientSwish<<<MtcsTool.gridSize(outputs*batch),MTCS_BLOCK,0,MtcsTool.getStream()>>>
         (outputData->getDataArray(),outputs*batch,activation_input->getDataArray(),deltaData->getDataArray());

      else if (activation == ActivationType.MISH)
         /*!gradient_array_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);*/
         MtcsActivation.gradientMish<<<MtcsTool.gridSize(outputs*batch),MTCS_BLOCK,0,MtcsTool.getStream()>>>
         (outputs*batch,activation_input->getDataArray(),deltaData->getDataArray());
      else if (activation == ActivationType.HARD_MISH)
         /*!gradient_array_hard_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);*/
         MtcsActivation.gradientHardMish<<<MtcsTool.gridSize(outputs*batch),MTCS_BLOCK,0,MtcsTool.getStream()>>>
         (outputs*batch,activation_input->getDataArray(),deltaData->getDataArray());
      else if (activation == ActivationType.NORM_CHAN_SOFTMAX || activation == ActivationType.NORM_CHAN_SOFTMAX_MAXVAL)
         /*!gradient_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);*/
         MtcsActivation.gradientNormlizeChannelsSoftMax(outputData->getDataArray(),
         outputs*batch,batch,outputDimen.channels,outputDimen.w*outputDimen.h,deltaData->getDataArray());
      else if (activation == ActivationType.NORM_CHAN)
         /*!gradient_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);*/
         MtcsActivation.gradientNormlizeChannels(outputData->getDataArray(),
         outputs*batch,batch,outputDimen.channels,outputDimen.w*outputDimen.h,deltaData->getDataArray());
      else{
         /* compare
         float *cpuDelta=malloc(batch*outputs*sizeof(float));
         MtcsMem.memcpy(cpuDelta,deltaData->getDataArray(),batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
         */
         MtcsActivation.gradient/*!gradient_array_ongpu*/(outputData->getDataArray(),
               batch*outputs, activation, deltaData->getDataArray());

         /* compare
         float *cpuOut=malloc(batch*outputs*sizeof(float));
         MtcsMem.memcpy(cpuOut,outputData->getDataArray(),batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
         float *compareDelta=malloc(batch*outputs*sizeof(float));
         MtcsMem.memcpy(compareDelta,deltaData->getDataArray(),batch*outputs*sizeof(float),MtcsCpyKind.DEV2HOST);
         Activation.gradientArray(cpuOut,batch*outputs,activation,cpuDelta);
         printf("MtcsConvLayer.c backward 比较output的梯度 layer:%d %s\n",getOrderNumber(),Activation.getString(activation));
         DnnUtils.compare(batch*outputs,cpuDelta,compareDelta);
         free(cpuDelta);
         free(compareDelta);
         free(cpuOut);
         */
      }

      if (!batch_normalize){
         /*!backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);*/
         biasData->calcGrad(deltaData);
      }

      InputData *original_input = state.input;
      if(xnor)
         state.input = binary_input;

      if (batch_normalize) {
         backwardNorm/*!backward_batchnorm_layer_gpu(l, state);*/(state);
      }

      int m = filters / groups;
      int n = ksize*ksize*inputDimen.channels / groups;
      int k = outputDimen.w*outputDimen.h;
      float *weights=weightData->getWeights();
      float *weightUpdates=weightData->getUpdates();
      float *delta=deltaData->getDataArray();
      int i, j;
      for(i = 0; i < batch; ++i){
         for (j = 0; j < groups; ++j) {
            /*!float * a = l.delta_gpu + (i*l.groups + j)*m*k;*/
            float * a = delta + (i*groups + j)*m*k;
            float * b = state.workspace;
            float * c = weightUpdates + j*nweights / groups;
            /*!float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;*/
            float *im = state.input->getData(i);
            if (!net->adversarial() && !train_only_bn) {
               im2Col->im2col/*!im2col_gpu_ext*/(im,          // input
                     inputDimen.channels / groups,         // input channels
                     inputDimen.h, inputDimen.w,               // input size (h, w)
                     ksize, ksize,         // kernel size (h, w)
                     pad * dilation, pad * dilation,   // padding (h, w)
                     stride_y, stride_x,     // stride (h, w)
                     dilation, dilation, // dilation (h, w)
                     state.workspace);       // output
               /* compare
               float *cpuC=(float*)calloc(nweights,sizeof(float));
               MtcsMem.memcpy(cpuC,c,nweights*sizeof(float),MtcsCpyKind.DEV2HOST);
               float *cpuA=(float*)calloc(m*k,sizeof(float));
               MtcsMem.memcpy(cpuA,a,m*k*sizeof(float),MtcsCpyKind.DEV2HOST);
               float *cpuB=(float*)calloc(net->getWorkspaceSize(),sizeof(float));
               MtcsMem.memcpy(cpuB,state.workspace,net->getWorkspaceSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
               gemm(0, 1, m, n, k, 1, cpuA, k, cpuB, k, 1, cpuC, n);
               */
               gemm->gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

               /* compare
               float *compareCpu=(float*)calloc(nweights,sizeof(float));
               MtcsMem.memcpy(compareCpu,c,nweights*sizeof(float),MtcsCpyKind.DEV2HOST);
               printf("MtcsConvLayer backward 比较矩阵运算的weight i:%d layer:%d m:%d n:%d k:%d %d\n",i,getOrderNumber(),m,n,k,nweights);
               DnnUtils.compare(nweights,compareCpu,cpuC);
               free(compareCpu);
               free(cpuC);
               free(cpuA);
               free(cpuB);
               */
            }
            if (state.delta) {
               if (binary || xnor)
                  swapBinary/*!swap_binary*/();
               float * a =weights + j*nweights / groups;
               float * b = delta + (i*groups + j)*m*k;
               float * c = state.workspace;

               /* compare
               float *cpuC=(float*)calloc(net->getWorkspaceSize(),sizeof(float));
               float *cpuA=(float*)calloc(nweights,sizeof(float));
               MtcsMem.memcpy(cpuA,a,nweights*sizeof(float),MtcsCpyKind.DEV2HOST);
               float *cpuB=(float*)calloc(m*k,sizeof(float));
               MtcsMem.memcpy(cpuB,delta,m*k*sizeof(float),MtcsCpyKind.DEV2HOST);
               gemm(1, 0, n, k, m, 1, cpuA, n, cpuB, k, 0, cpuC, k);
               */

               gemm->gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

               /* compare
               float *compareCPU=(float*)calloc(net->getWorkspaceSize(),sizeof(float));
               MtcsMem.memcpy(compareCPU,c,net->getWorkspaceSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
               printf("MtcsConvLayer backward 比较矩阵运算的delta i:%d layer:%d m:%d n:%d k:%d\n",i,getOrderNumber(),n,k,m);
               DnnUtils.compare(n*k,compareCPU,cpuC);
               free(compareCPU);
               free(cpuC);
               free(cpuA);
               free(cpuB);
               */
               float *inputDelta = state.delta->getDataArray()
                     + (i*groups + j)*inputDimen.channels / groups*inputDimen.h*inputDimen.w;
               im2Col->col2im/*!col2im_gpu_ext*/(
                     state.workspace,        // input
                     inputDimen.channels / groups, // input channels
                     inputDimen.h, inputDimen.w, // input size (h, w)
                     ksize, ksize,         // kernel size (h, w)
                     pad * dilation, pad * dilation,   // padding size (h, w)
                     stride_y, stride_x,     // stride size (h, w)
                     dilation, dilation, // dilation size (h, w)
                     inputDelta);                 // output (delta)

               /* compare
               float *cpudelta=(float*)calloc(state.delta->getSize(),sizeof(float));
               float *space=malloc(net->getWorkspaceSize()*sizeof(float));
               MtcsMem.memcpy(space,state.workspace,net->getWorkspaceSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
               im2ColCPU->col2im(
                     space,        // input
                     inputDimen.channels / groups,         // input channels
                     inputDimen.h, inputDimen.w, // input size (h, w)
                     ksize, ksize,         // kernel size (h, w)
                     pad * dilation, pad * dilation,   // padding size (h, w)
                     stride_y, stride_x,     // stride size (h, w)
                     dilation, dilation, // dilation size (h, w)
                     cpudelta);                 // output (delta)

               printf("MtcsConvLayer backward 比较col2im的delta i:%d layer:%d\n",i,getOrderNumber());
               float *compareCpu=malloc(state.delta->getSize()*sizeof(float));
               MtcsMem.memcpy(compareCpu,inputDelta,state.delta->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
               DnnUtils.compare(state.delta->getSize(),compareCpu,cpudelta);
               free(cpudelta);
               free(compareCpu);
               free(space);
                */
               if (binary || xnor) {
                  swapBinary/*!swap_binary(&l)*/();
               }
               if (xnor)
                  /*!gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);*/
                  MtcsActivation.gradient/*!gradient_array_ongpu*/(original_input->getData(i),inputs,activation, state.delta->getData(i));
            }
         }
      }

      if (net->tryFixNan()) {
         if (state.delta) {
            MtcsTool.resetNanAndInf/*!reset_nan_and_inf*/(state.delta->getDataArray(),inputs *batch);
         }
         int size = nweights;
         MtcsTool.resetNanAndInf/*!reset_nan_and_inf*/(weightUpdates, size);
         MtcsTool.fixNanAndInf/*!fix_nan_and_inf*/(weights, size);
      }
   }

   /**
    * 覆盖父类NLayer的方法
    */
   public$ float *getMCbn(){
      return m_cbn_avg;
   }

   public$ float *getVCbn(){
      return v_cbn_avg;
   }

   //原型 gradient_centralization_kernel blas_kernels.cu
   __global__ void gradient_centralization_kernel(int filters, int f_size, float *in){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;
      const int tid = index % WARP_SIZE;
      const int f = index / WARP_SIZE;

      if (f >= filters)
         return;

      float mean = 0;
      for (int i = 0; i < f_size; i += WARP_SIZE) {
         mean += MtcsTool.warpAllReduceSum(in[f*f_size + i + tid]);
      }
      mean = mean / f_size;
      for (int i = 0; i < f_size; i += WARP_SIZE) {
         in[f*f_size + i + tid] -= mean;
      }
   }

   //原型 gradient_centralization_gpu blas.h blas_kernels.cu
   void gradientCentralization(int w, int h, int c, int f, float *in){
      const int size = f * WARP_SIZE;
      const int f_size = c * h * w;
      if (f_size % WARP_SIZE == 0) {
         gradient_centralization_kernel <<<MtcsTool.getNumberOfBlocks(size, MTCS_BLOCK), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
               (f, f_size, in);
      }
   }

   //原型 adam_kernel blas_kernels.cu
   __global__ void adamKernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t){
       int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (index >= N)
          return;

       float mhat = m[index] / (1.f - powf(B1, t));
       float vhat = v[index] / (1.f - powf(B2, t));

       x[index] = x[index] + rate * mhat / (sqrtf(vhat) + eps);
   }
   //原型 adam_gpu blas.h blas_kernels.cu
   void adam(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t){
      adamKernel <<<MtcsTool.gridSize(n), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
             (n, x, m, v, B1, B2, rate, eps, t);
   }


   //原型 adam_update_gpu blas.h blas_kernels.cu
   void adam_update_gpu(float *w, float *d, float *m, float *v, float B1,
         float B2, float eps, float decay, float rate, int n, int batch, int t){
       MtcsTool.scal/*!scal_ongpu(n, B1, m, 1);*/(n, B1, m);
       MtcsTool.scal/*!scal_ongpu(n, B2, v, 1);*/(n, B2, v);
       MtcsTool.axpy/*!axpy_ongpu(n, -decay*batch, w, 1, d, 1);*/(n, -decay*batch, w, d);

       MtcsTool.axpy/*!axpy_ongpu(n, (1 - B1), d, 1, m, 1);*/(n, (1 - B1), d,  m);
       MtcsTool.mul/*!mul_ongpu(n, d, 1, d, 1);*/(n,d,d);
       MtcsTool.axpy/*!axpy_ongpu(n, (1 - B2), d, 1, v, 1);*/(n, (1 - B2), d, v);

       adam/*!adam_gpu*/(n, w, m, v, B1, B2, rate, eps, t);
       MtcsTool.fill/*!fill_ongpu(n, 0, d, 1);*/(n, 0, d);
   }

   //原型 mult_inverse_array_kernel blas_kernels.cu
   __global__  void mult_inverse_array_kernel(const float *src_gpu, float *dst_gpu, int size, const float eps,
       float divider, const float clip, const float abs_add){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;

       if (index < size) {
           float val = src_gpu[index];
           float sign = (val < 0) ? -1 : 1;
           // eps = 1 by default
           // eps = 2 - lower delta
           // eps = 0 - higher delta (linear)
           // eps = -1 - high delta (inverse number)
           // = (abs(x)*10+1)^(-1)
           float unsigned_val = powf(fabs(val)*10 + abs_add, eps);
           unsigned_val = unsigned_val / divider;
           if (unsigned_val > clip && clip != 0.0)
              unsigned_val = clip;
           if (isnan(unsigned_val) || isinf(unsigned_val))
              unsigned_val = 0;
           dst_gpu[index] = unsigned_val * sign;
       }
   }


   //原型 mult_inverse_array_gpu blas.h blas_kernels.cu
    void multInverseArray (const float *src_gpu, float *dst_gpu, int size,
          float eps, float divider, float clip, float abs_add){
       const int block_size = MTCS_BLOCK;
       const int num_blocks = MtcsTool.getNumberOfBlocks(size, block_size);
       mult_inverse_array_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>>
             (src_gpu, dst_gpu, size, eps, divider, clip, abs_add);
   }

   //原型 update_convolutional_layer_gpu convolutional_kernels.cu
    //覆盖 父类的 update方法
   void update(int batch, float learning_rate_init, float momentum, float decay, float loss_scale){
      if (deform) {
         if (rotate)
            /*!rotate_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 1);*/
            ((MtcsConvKernel*)weightData)->rotate(1);
         else if (sway)
            /*!sway_and_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);*/
            ((MtcsConvKernel*)weightData)->swayAndFlip(angle,1);
         else if (stretch)
            /*!stretch_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 0, 1);*/
            ((MtcsConvKernel*)weightData)->stretch(0,1);

         else if (stretch_sway)
              /*!stretch_sway_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);*/
            ((MtcsConvKernel*)weightData)->stretchSwayFlip(angle,1);

         /*!reduce_and_expand_array_gpu(l.weight_deform_gpu, l.weight_updates_gpu, l.nweights, 4);*/
         ((MtcsConvKernel*)weightData)->reduceAndExpandArray(4);
      }

      // Loss scale for Mixed-Precision on Tensor-Cores
      float learning_rate = learning_rate_init*learning_rate_scale / loss_scale;
     // printf("convolutional update 00 deform:%d adam:%d reverse:%f index:%d grad_centr:%d learning_rate:%f\n",
            //deform,adam,reverse,getOrderNumber(),grad_centr,learning_rate);
      a_assert(weightData->getSize()==nweights);
      MtcsTool.resetNanAndInf/*!reset_nan_and_inf*/(weightData->getUpdates(),weightData->getSize());
      MtcsTool.fixNanAndInf/*!fix_nan_and_inf*/(weightData->getWeights() ,weightData->getSize());

      // Gradient Centralization
      if (grad_centr && batch_normalize) {
         gradientCentralization/*!gradient_centralization_gpu*/(ksize,ksize,
               inputDimen.channels/groups,filters, weightData->getUpdates());
      }


      if (adam) {
         adam_update_gpu(weightData->getWeights(), weightData->getUpdates(),self->m,
               self->v, B1, B2, eps, decay, learning_rate, nweights, batch, self->t);

         adam_update_gpu(biasData->getBias(), biasData->getUpdates(),
               self->bias_m, self->bias_v, B1, B2, eps, decay, learning_rate, filters, batch, self->t);
         if (scaleData->getScales()!=NULL/*!l.scales_gpu*/) {
            adam_update_gpu(scaleData->getScales(),scaleData->getUpdates(),self->scale_m, self->scale_v, B1, B2, eps,
                  decay, learning_rate, filters, batch, self->t);
         }
      } else {
         float *old_weight_updates_gpu =weightData->getUpdates();
         if (reverse) {
            float clip = 0.0;
            float divider = 1.0;
            float abs_add = 1.0;
            multInverseArray/*!mult_inverse_array_gpu*/(weightData->getUpdates(),
                  outputData->getDataArray(), inputs*batch, reverse, divider, clip, abs_add);
            /*!l.weight_updates_gpu = l.output_gpu;*/
            weightData->setUpdates(outputData->getDataArray());
         }

         MtcsTool.axpy/*!axpy_ongpu*/(nweights, -decay*batch*loss_scale, weightData->getWeights(), weightData->getUpdates());
         MtcsTool.axpy/*!axpy_ongpu*/(nweights, learning_rate / batch,  weightData->getUpdates(), weightData->getWeights());

         /*!l.weight_updates_gpu = old_weight_updates_gpu;*/
         weightData->setUpdates( old_weight_updates_gpu);

         MtcsTool.scal/*!scal_ongpu*/(nweights, momentum, weightData->getUpdates());

         MtcsTool.axpy/*!axpy_ongpu*/(filters, learning_rate / batch, biasData->getUpdates(), biasData->getBias());
         MtcsTool.scal/*!scal_ongpu*/(filters, momentum, biasData->getUpdates());
         if (scaleData && scaleData->getScales()!=NULL/*!l.scales_gpu*/) {
            MtcsTool.axpy/*!axpy_ongpu*/(filters, learning_rate / batch, scaleData->getUpdates(), scaleData->getScales());
            MtcsTool.scal/*!scal_ongpu*/(filters, momentum, scaleData->getUpdates());
         }
      }

      if (deform) {
         /*!expand_array_gpu(l.weights_gpu, l.weight_deform_gpu, l.nweights, 4);*/
         ((MtcsConvKernel*)weightData)->expandArray(4);
         if (rotate)
            /*!rotate_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0);*/
            ((MtcsConvKernel*)weightData)->rotate(0);
         else if (sway)
            /*!sway_and_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);*/
            ((MtcsConvKernel*)weightData)->swayAndFlip(angle,0);

         else if (stretch)
            /*!stretch_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0, 0);*/
            ((MtcsConvKernel*)weightData)->stretch(0,0);
         else if (stretch_sway)
            /*!stretch_sway_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);*/
            ((MtcsConvKernel*)weightData)->stretchSwayFlip(angle,0);
      }

      if (clip) {
         MtcsTool.constrain/*!constrain_ongpu*/(nweights, clip, weightData->getWeights());
      }
   }

   private$ void mtcsDataToHost(){
      MtcsStream *stream=MtcsTool.getStream();
      MtcsCpyKind kind=MtcsCpyKind.DEV2HOST;
      if(!saveData.weights){
         saveData.weights=malloc(nweights*sizeof(float));
         saveData.biases=malloc(biasData->getSize()*sizeof(float));
         if (weightData->getUpdates()){
            saveData.weightUpdates=malloc(nweights*sizeof(float));
         }
         if(biasData->getUpdates()){
            saveData.biasUpdates=malloc(biasData->getSize()*sizeof(float));
         }
         if (batch_normalize){
            saveData.scales=malloc(scaleData->getSize()*sizeof(float));
            saveData.rollMean=malloc(normData->getChannels()*sizeof(float));
            saveData.rollVariance=malloc(normData->getChannels()*sizeof(float));
         }
         if (adam){
            saveData.m=malloc(nweights*sizeof(float));
            saveData.v=malloc(nweights*sizeof(float));
         }
      }
      MtcsMem.memcpyAsync(saveData.weights,weightData->getWeights(),nweights*sizeof(float),kind,stream);
      MtcsMem.memcpyAsync(saveData.biases,biasData->getBias(),biasData->getSize()*sizeof(float),kind,stream);

      if (weightData->getUpdates()){
         MtcsMem.memcpyAsync(saveData.weightUpdates,weightData->getUpdates(),
               nweights*sizeof(float),kind,stream);
      }
      if(biasData->getUpdates()){
         MtcsMem.memcpyAsync(saveData.biasUpdates,biasData->getUpdates(),
               biasData->getSize()*sizeof(float),kind,stream);
      }

      if (batch_normalize){
         MtcsMem.memcpyAsync(saveData.scales,scaleData->getScales(),
               scaleData->getSize()*sizeof(float),kind,stream);
         MtcsMem.memcpyAsync(saveData.rollMean,normData->getRollingMean(),
               normData->getChannels()*sizeof(float),kind,stream);
         MtcsMem.memcpyAsync(saveData.rollVariance,normData->getRollingVariance(),
               normData->getChannels()*sizeof(float),kind,stream);
      }
      if (adam){
         MtcsMem.memcpyAsync(saveData.m,m,nweights*sizeof(float),kind,stream);
         MtcsMem.memcpyAsync(saveData.v,v,nweights*sizeof(float),kind,stream);
      }
      //在cuda平台等同调用 cudaStreamSynchronize
      stream->sync();
   }

   //原型 save_convolutional_weights parser.c
   void saveWeights(FILE *fp){
      mtcsDataToHost();
      fwrite(saveData.biases, sizeof(float),biasData->getSize(), fp);
      if (batch_normalize){
         fwrite(saveData.scales, sizeof(float),scaleData->getSize(), fp);
         fwrite(saveData.rollMean, sizeof(float), normData->getChannels(), fp);
         fwrite(saveData.rollVariance, sizeof(float),normData->getChannels(), fp);
      }
      fwrite(saveData.weights, sizeof(float), nweights, fp);
   }

   //原型 save_convolutional_weights_ema parser.c
   void saveWeightsEma(FILE *fp){
      mtcsDataToHost();
      if(!saveData.biasEma){
         saveData.biasEma = malloc(biasData->getSize()*sizeof(float));
      }
      if (batch_normalize) {
         if(!saveData.scaleEma){
            saveData.scaleEma = malloc(scaleData->getSize()*sizeof(float));
         }
      }
      if(!saveData.weightEma){
         saveData.weightEma = malloc(nweights*sizeof(float));
      }
      MtcsStream *stream=MtcsTool.getStream();
      MtcsCpyKind kind=MtcsCpyKind.DEV2HOST;
      MtcsMem.memcpyAsync(saveData.biasEma,biasData->getEma(),biasData->getSize()*sizeof(float),kind,stream);
      if (batch_normalize) {
         MtcsMem.memcpyAsync(saveData.scaleEma,scaleData->getEma(),scaleData->getSize()*sizeof(float),kind,stream);
      }
      MtcsMem.memcpyAsync(saveData.weightEma,weightData->getEma(),nweights*sizeof(float),kind,stream);

      fwrite(saveData.biasEma, sizeof(float),biasData->getSize(), fp);
      if (batch_normalize) {
         fwrite(saveData.scaleEma, sizeof(float),scaleData->getSize(), fp);
         fwrite(saveData.rollMean, sizeof(float), normData->getChannels(), fp);
         fwrite(saveData.rollVariance, sizeof(float),normData->getChannels(), fp);
      }
      fwrite(saveData.weightEma, sizeof(float), nweights, fp);
   }

   void loadWeights(FILE *fp){
      float *temp = malloc(workspace_size*sizeof(float));
      MtcsCpyKind kind=MtcsCpyKind.HOST2DEV;
      MtcsStream *stream= MtcsTool.getStream();
      int num = nweights;
      int read_bytes;
      read_bytes = fread(temp, sizeof(float), biasData->getSize(), fp);
      if (read_bytes > 0 && read_bytes < biasData->getSize())
         printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", getOrderNumber());
      printf("mtcsconvlayer loadWeights 00:%d flipped:%d layer:%d\n",read_bytes,flipped,getOrderNumber());
      MtcsMem.memcpyAsync(biasData->getBias(),temp, biasData->getSize()*sizeof(float),kind,stream);
      if (batch_normalize && (!dontloadscales)){
         read_bytes = fread(temp, sizeof(float), scaleData->getSize(), fp);
         MtcsMem.memcpyAsync(scaleData->getScales(),temp, scaleData->getSize()*sizeof(float),kind,stream);
         if (read_bytes > 0 && read_bytes < scaleData->getSize())
            printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", getOrderNumber());
         read_bytes = fread(temp, sizeof(float), normData->getChannels(), fp);
         MtcsMem.memcpyAsync(normData->getRollingMean(),temp, normData->getChannels()*sizeof(float),kind,stream);
         if (read_bytes > 0 && read_bytes <normData->getChannels())
            printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n",  getOrderNumber());
         read_bytes = fread(temp, sizeof(float), normData->getChannels(), fp);
         MtcsMem.memcpyAsync(normData->getRollingVariance(),temp, normData->getChannels()*sizeof(float),kind,stream);
         if (read_bytes > 0 && read_bytes < normData->getChannels())
            printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", getOrderNumber());
      }
      printf("mtcsconvlayer loadWeights 11:%d flipped:%d layer:%d\n",read_bytes,flipped,getOrderNumber());

      read_bytes = fread(temp, sizeof(float), num, fp);
      if (read_bytes > 0 && read_bytes < filters)
         printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", getOrderNumber());

      if (flipped) {
         transposeMatrix(temp, (inputDimen.channels/groups)*ksize*ksize , filters);
      }
      MtcsMem.memcpyAsync(weightData->getWeights(),temp,num*sizeof(float),kind,stream);
      free(temp);
   }

   __global__ void fuseWeights(float *biases,float *scales,float *rollingMean,float *rollingVariance,float *weights){
      int f;
      //printf("convolutional -batch_normalize 11 - %d weights:%p n:%d %d\n",j,weights,cl->n,cl->filters);
      for (f = 0; f < filters; ++f){
         biases[f] = biases[f] - (double)scales[f] * rollingMean[f] / (sqrt((double)rollingVariance[f] + .00001));
         double precomputed =scales[f] / (sqrt((double)rollingVariance[f] + .00001));
         const size_t filter_size = ksize*ksize*inputDimen.channels / groups;
         int i;
         for (i = 0; i < filter_size; ++i) {
            int w_index = f*filter_size + i;
            weights[w_index] *= precomputed;
         }
      }
   }

   //原型 fuse_conv_batchnorm darknet.h network.c
    public$ void fuseConvBatchnorm(){
       //printf("fuseConvBatchnorm convolutional -batch_normalize 00 - %d\n",j);
       float *biases=biasData->getBias();
       float *scales=scaleData->getScales();
       float *rollingMean=normData->getRollingMean();
       float *rollingVariance=normData->getRollingVariance();
       float *weights=weightData->getWeights();
       //只能是1x1的线程数
       fuseWeights(biases,scales,rollingMean,rollingVariance,weights);
    }

};

