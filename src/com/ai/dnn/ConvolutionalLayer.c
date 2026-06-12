#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <aet/lang/System.h>

#include "ConvolutionalLayer.h"
#include "DnnUtils.h"
#include "Activation.h"
#include "DstbCompute.h"
#include "DataFactory.h"
#include "cnnmicro.h"

/**
 * 运行单位是批次，
 * 做卷积运算是图片数。
 * batch=指数
 * n=图片数
 * batch的运用是在 TrainDetector.c 中的
 * cfg中的batch是指图片数。
 * subdivisions是指循环次数
 * batch/subdivisions 每次计算的图片数
 * for(i=0;i<subdivisions;i++){
 *    train_net();
 * }
 * 在forward中会循环batch/subdivisions 次
 * train_net(){
 *   for(i=0;i<layers;i++){
 *     layer->forward()
 *   }
 * }
 */

impl$ ConvolutionalLayer{

   /**
   * rand_normal比GaussPRNG快4倍。都是用的Box–Muller transform算法
   * 但rand_normal没有用do while循环。
   * 每个卷积核与原始图的一个通道矩阵运算一次 生成一个新的feature map(特征图)。如图有多个通道，每个通道
   * 对应的feature map相加，生成的最后feature map就是一个filter对应的特征图
   */
   ConvolutionalLayer(int batch, int steps, int h, int w, int c,
            int n, int groups, int size, int stride_x, int stride_y, int dilation,
            int padding, ActivationType activation, int batch_normalize, int binary,
            int xnor, int adam, int use_bin_output, int index, int antialiasing,
            ConvolutionalLayer *share_layer, int assisted_excitation, int deform, int train){
      int total_batch = batch*steps;
      int i;
      self->type = LayerType.CONVOLUTIONAL;
      self->train = train;
      if (xnor)
         groups = 1;   // disable groups for XNOR-net
      if (groups < 1)
         groups = 1;

      const int blur_stride_x = stride_x;
      const int blur_stride_y = stride_y;
      self->antialiasing = antialiasing;
      if (antialiasing)
         stride_x = stride_y = stride = self->stride_x = self->stride_y = 1; // use stride=1 in host-layer

      self->wait_stream_id = -1;
      self->deform = deform;
      self->assisted_excitation = assisted_excitation;
      self->share_layer = share_layer;
      self->index = index;
      setInputDimen(w,h,c);
      self->groups = groups;
      self->n = n;
      self->filters=n;
      self->binary = binary;
      self->xnor = xnor;//是否对权重以及输入进行二值化
      self->use_bin_output = use_bin_output;
      self->batch = batch;
      self->steps = steps;
      self->stride = stride_x;
      self->stride_x = stride_x;
      self->stride_y = stride_y;
      self->dilation = dilation;
      self->ksize = size;
      self->pad = padding;
      self->batch_normalize = batch_normalize;
      self->learning_rate_scale = 1;
      self->nweights = (c / groups) * n * size * size;
      //printf("ConvolutionalLayer.c antialiasing:%d share_layer:%p train:%d size:%d nweights:%d steps:%d,groups:%d binary:%d xnor:%d batch_normalize:%d adam:%d\n",
      // antialiasing,share_layer,train,size,nweights,steps,groups,binary,xnor,batch_normalize,adam);
      if (share_layer) {
         if (size != share_layer->ksize || nweights != share_layer->nweights
         || c != share_layer->inputDimen.channels || n != share_layer->n) {
            a_error("Layer size, nweights, channels or filters don't match for the share_layer");
         }
         //wigthData已包含了 weight_updates
         self->weightData = share_layer->weightData;
         //biases已包含了 bias_updates
         self->biasData = share_layer->biasData;
         printf("conv 共享 share_layer\n");
      }else{
         //l.weights = (float*)xcalloc(l.nweights, sizeof(float));
         //l.biases = (float*)xcalloc(n, sizeof(float));
         self->weightData=DataFactory.getInstance()->createConvKernel(size,c,self->filters,padding,stride); //多张图对应同一个权重
         self->biasData = DataFactory.getInstance()->createBiasData(self->filters); //filters 卷积核个数 一个核一个偏差
         if (train) {
            //                  l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
            //                  l.bias_updates = (float*)xcalloc(n, sizeof(float));
            //
            //                  l.weights_ema = (float*)xcalloc(l.nweights, sizeof(float));
            //                  l.biases_ema = (float*)xcalloc(n, sizeof(float));

            self->weightData->createUpdates();
            self->weightData->createEma();
            self->biasData->createEma();
            self->biasData->createUpdates();
         }
      }

      /*!
      // float scale = 1./sqrt(size*size*c);
      float scale = sqrt(2./(size*size*c/groups));
      if (l.activation == NORM_CHAN || l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) {
      for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;   // rand_normal();
      }
      else {
      for (i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);   // rand_normal();
      }
      */
      ((ConvKernel *)weightData)->initData(activation);
      int out_w = getOutWidth();
      int out_h = getOutHeight();
      outputData=DataFactory.getInstance()->createOutputData(out_w,out_h,self->filters,total_batch);
      setOutputDimen(out_w,out_h,self->filters);
      self->outputs = out_w*out_h*self->filters;
      self->inputs =   w * h * c;
      self->activation = activation; //激活函数类型
      if (train)
         //l.delta = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
         self->deltaData=DataFactory.getInstance()->createDeltaData(out_w,out_h,self->filters,total_batch);

      if(binary){
         self->binWeightData =new$ ConvKernel(nweights);
         self->cweights = (char*)xcalloc(nweights, sizeof(char));
         self->scaleData=DataFactory.getInstance()->createScaleData(self->filters);
      }
      if(xnor){
         self->binWeightData =new$ ConvKernel(nweights);
         /*!(float*)xcalloc(self->inputs * batch, sizeof(float));*/
         self->binary_input = DataFactory.getInstance()->createInputData(inputs,batch);

         int align = 32;// 8;
         int src_align = out_h*out_w;
         self->bit_align = src_align + (align - src_align % align);

         self->mean_arr = (float*)xcalloc(n, sizeof(float));

         const size_t new_c = c / 32;
         size_t in_re_packed_input_size = new_c * w * h + 1;
         self->bin_re_packed_input = (auint32*)xcalloc(in_re_packed_input_size, sizeof(auint32));

         self->lda_align = 256;  // AVX2
         int k = size*size*c;
         size_t k_aligned = k + (self->lda_align - k%self->lda_align);
         size_t t_bit_input_size = k_aligned * self->bit_align / 8;
         self->t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
      }
      if(batch_normalize){
         if (share_layer) {
            //l.scales = l.share_layer->scales;
            //l.scale_updates = l.share_layer->scale_updates;
            //l.mean = l.share_layer->mean;
            //l.variance = l.share_layer->variance;
            //l.mean_delta = l.share_layer->mean_delta;
            //l.variance_delta = l.share_layer->variance_delta;
            //l.rolling_mean = l.share_layer->rolling_mean;
            //l.rolling_variance = l.share_layer->rolling_variance;
            self->scaleData=share_layer->scaleData;
            self->normData=share_layer->normData;
            self->deltaData=share_layer->deltaData;
         }else {
            /*
            l.scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
            l.scales[i] = 1;
            }
            */
            self->scaleData=DataFactory.getInstance()->createScaleData(self->filters);
            self->scaleData->init(1.0);
            self->normData=DataFactory.getInstance()->createNormData(out_w,out_h,self->filters,total_batch);
            if (train) {
               //l.scales_ema = (float*)xcalloc(n, sizeof(float));
               // l.scale_updates = (float*)xcalloc(n, sizeof(float));
               self->scaleData->createEma();
               self->scaleData->createUpdates();

               // l.mean = (float*)xcalloc(n, sizeof(float));
               // l.variance = (float*)xcalloc(n, sizeof(float));
               self->normData->createMean();
               self->normData->createVariance();
               //l.mean_delta = (float*)xcalloc(n, sizeof(float));
               // l.variance_delta = (float*)xcalloc(n, sizeof(float));
            }
         //NormData创建时已创建 rolling_mean rolling_variance
         // l.rolling_mean = (float*)xcalloc(n, sizeof(float));
         //l.rolling_variance = (float*)xcalloc(n, sizeof(float));
         }

         if (train) {
            //l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
            // l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
            self->normData->createX();
         }
      }
      if (self->activation == ActivationType.SWISH
      || self->activation == ActivationType.MISH
      || self->activation == ActivationType.HARD_MISH)
         /*!(float*)calloc(total_batch*outputs, sizeof(float));*/
         self->activation_input = DataFactory.getInstance()->createOutputData(outputs,total_batch);

      if(adam){
         self->adam = 1;
         self->m = (float*)xcalloc(nweights, sizeof(float));
         self->v = (float*)xcalloc(nweights, sizeof(float));
         self->bias_m = (float*)xcalloc(n, sizeof(float));
         self->scale_m = (float*)xcalloc(n, sizeof(float));
         self->bias_v = (float*)xcalloc(n, sizeof(float));
         self->scale_v = (float*)xcalloc(n, sizeof(float));
      }

      // l.workspace_size = get_convolutional_workspace_size(l);
      self->workspace_size = getWorkSpaceSize();

      //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
      self->bflops = (2.0 * nweights * out_h*out_w) / 1000000000.;
      if (self->xnor)
         self->bflops = self->bflops / 32;
      if (self->xnor && self->use_bin_output)
         fprintf(stderr, "convXB");
      else if (self->xnor)
         fprintf(stderr, "convX ");
      else if (self->share_layer)
         fprintf(stderr, "convS ");
      else if (self->assisted_excitation)
         fprintf(stderr, "convAE");
      else
         fprintf(stderr, "conv  ");

      if (groups > 1)
         fprintf(stderr, "%5d/%4d ", n, groups);
      else
         fprintf(stderr, "%5d      ", n);

      if (stride_x != stride_y)
         fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
      else {
         if (dilation > 1)
            fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
         else
            fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
      }

      fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c,
               out_w, out_h, self->outputDimen.channels, self->bflops);

      //fprintf(stderr, "%5d/%2d %2d x%2d /%2d(%d)%4d x%4d x%4d  -> %4d x%4d x%4d %5.3f BF\n", n, groups, size, size, stride, dilation, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

      if (antialiasing) {
         printf("AA:  ");
         // l.input_layer = (layer*)calloc(1, sizeof(layer));
         int blur_size = 3;
         int blur_pad = blur_size / 2;
         if (antialiasing == 2) {
            blur_size = 2;
            blur_pad = 0;
         }
         input_layer = new$ ConvolutionalLayer(batch, steps, out_h, out_w, n, n, n,
            blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, ActivationType.LINEAR,
            0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
         const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
         int i;
         if (blur_size == 2) {
            float *weights=input_layer->weightData->getWeights();
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
               weights[i + 0] = 1 / 4.f;
               weights[i + 1] = 1 / 4.f;
               weights[i + 2] = 1 / 4.f;
               weights[i + 3] = 1 / 4.f;
            }
         }else {
            float *weights=input_layer->weightData->getWeights();
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
               weights[i + 0] = 1 / 16.f;
               weights[i + 1] = 2 / 16.f;
               weights[i + 2] = 1 / 16.f;

               weights[i + 3] = 2 / 16.f;
               weights[i + 4] = 4 / 16.f;
               weights[i + 5] = 2 / 16.f;

               weights[i + 6] = 1 / 16.f;
               weights[i + 7] = 2 / 16.f;
               weights[i + 8] = 1 / 16.f;
            }
         }
         /*!
         for (i = 0; i < n; ++i)
         biases[i] = 0;
         */
         input_layer->biasData->clear();
      }

      im2Col =DataFactory.getInstance()->createIm2Col();
      gemm  =DataFactory.getInstance()->createGemm();
   }

   //讲解im2col和卷积运算
   //https://blog.csdn.net/mrhiuser/article/details/52672824
   //https://blog.csdn.net/jiachen0212/article/details/81531175 caffe的计算梯度方法
   //原型 binarize_weights convolutional_layer.h convolutional_layer.c
   static void binarizeWeights(float *weights, int n, int size, float *binary){
       int i, f;
       for(f = 0; f < n; ++f){
           float mean = 0;
           for(i = 0; i < size; ++i){
               mean += fabs(weights[f*size + i]);
           }
           mean = mean / size;
           for(i = 0; i < size; ++i){
               binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
           }
       }
   }

   /**
   * 原始图片经过卷积计算后输出的宽度
   * 公式：(w+2*pad-kernelSize)/stride+1
   */
   //原型 convolutional_out_width convolutional_layer.h convolutional_layer.c
   int getOutWidth(){
      return (inputDimen.w + 2*self->pad - ksize) /stride_x + 1;
   }

   //原型 convolutional_out_height convolutional_layer.h convolutional_layer.c
   int getOutHeight(){
      return (inputDimen.h + 2*self->pad - ksize) / stride_y + 1;
   }

   size_t getWorkSpaceSize(){
      return (size_t)outputDimen.h*outputDimen.w*ksize*ksize*inputDimen.channels*sizeof(float);
   }

   //原型 swap_binary convolutional_layer.h convolutional_layer.c
   void swapBinary(){
      WeightData *swap = self->weightData;
      self->weightData = self->binWeightData;
      self->binWeightData = swap;
   }

   static void checkOutput(char *explain){
      int i,b;
      int equalzero=0;
      int smallzerothanMinusOne=0;
      int thanzeroLessOne=0;
      int thanone=0;
      int other=0;
      int n=((IOData*)outputData)->getSize();
      int out_w = getOutWidth();
      int out_h = getOutHeight();
      // printf("vvv ---ss %d %d %d %d %d\n",n,out_w,out_h,filters,out_w*out_h*filters);
      for(b = 0; b < batch; ++b){
         float *outputDataes=outputData->getData(b);
         for(i=0;i<n;i++){
            float output=outputDataes[i];
            if(output>1.0){
               thanone++;
            }else if(output==0){
               equalzero++;
            }else if(output<0 && output>-1.0){
               smallzerothanMinusOne++;
            }else if(output>0 && output<1.0){
               thanzeroLessOne++;
            }else{
               other=0;
            }
         }
      }
      printf("%s :out_w:%d out_h:%d channels:%d equalzero=%d smallzerothanMinusOne=%d thanzeroLessOne=%d thanone=%d other=%d\n",
            explain,out_w,out_h,inputDimen.channels,  equalzero,smallzerothanMinusOne,thanzeroLessOne,thanone,other);
   }

   static void checkDelta(char *explain){
      int i,b;
      int equalzero=0;
      int smallzerothanMinusOne=0;
      int thanzeroLessOne=0;
      int thanone=0;
      int other=0;
      int n=deltaData->getSize();
      int out_w = getOutWidth();
      int out_h = getOutHeight();
      // printf("vvv ---ss %d %d %d %d %d\n",n,out_w,out_h,filters,out_w*out_h*filters);
      for(b = 0; b < batch; ++b){
         float *outputDataes=deltaData->getData(b);
         for(i=0;i<n;i++){
            float output=outputDataes[i];
            if(output>1.0){
               thanone++;
            }else if(output==0){
               equalzero++;
            }else if(output<0 && output>-1.0){
               smallzerothanMinusOne++;
            }else if(output>0 && output<1.0){
               thanzeroLessOne++;
            }else{
               other=0;
            }
         }
      }
      printf("%s :out_w:%d out_h:%d channels:%d equalzero=%d smallzerothanMinusOne=%d thanzeroLessOne=%d thanone=%d other=%d\n",
               explain,out_w,out_h,inputDimen.channels,  equalzero,smallzerothanMinusOne,thanzeroLessOne,thanone,other);
   }

   //原型 save_convolutional_weights parser.c
   void saveWeights(FILE *fp){
      fwrite(biasData->getBias(), sizeof(float),biasData->getSize(), fp);
      if (batch_normalize){
         fwrite(scaleData->getScales(), sizeof(float),scaleData->getSize(), fp);
         fwrite(normData->getRollingMean(), sizeof(float), normData->getChannels(), fp);
         fwrite(normData->getRollingVariance(), sizeof(float),normData->getChannels(), fp);
      }
      fwrite(weightData->getWeights(), sizeof(float), nweights, fp);
   }

   //原型 save_convolutional_weights_ema parser.c
   void saveWeightsEma(FILE *fp){
      fwrite(biasData->getEma(), sizeof(float),biasData->getSize(), fp);
      if (batch_normalize) {
         fwrite(scaleData->getEma(), sizeof(float),scaleData->getSize(), fp);
         fwrite(normData->getRollingMean(), sizeof(float), normData->getChannels(), fp);
         fwrite(normData->getRollingVariance(), sizeof(float),normData->getChannels(), fp);
      }
      fwrite(weightData->ema, sizeof(float), nweights, fp);
   }

   void transposeMatrix(float *a, int rows, int cols){
      float *transpose = calloc(rows*cols, sizeof(float));
      int x, y;
      for(x = 0; x < rows; ++x){
         for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
         }
      }
      memcpy(a, transpose, rows*cols*sizeof(float));
      free(transpose);
   }

   void printdata(float *data,int size,char *explain){
      int i;
      for(i=0;i<size;i++)
         printf("weights %s %d %f layer:%d\n",explain,i,data[i],getOrderNumber());
   }

   void loadWeights(FILE *fp){
      int num = nweights;
      int read_bytes;
      read_bytes = fread(biasData->getBias(), sizeof(float), biasData->getSize(), fp);
    //  printdata(biasData->getBias(),biasData->getSize(),"偏差数据");
      if (read_bytes > 0 && read_bytes < biasData->getSize())
         printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", getOrderNumber());
      if (batch_normalize && (!dontloadscales)){
         read_bytes = fread(scaleData->getScales(), sizeof(float), scaleData->getSize(), fp);
         if (read_bytes > 0 && read_bytes < scaleData->getSize())
            printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", getOrderNumber());
         read_bytes = fread(normData->getRollingMean(), sizeof(float), normData->getChannels(), fp);
         if (read_bytes > 0 && read_bytes <normData->getChannels())
            printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n",  getOrderNumber());
         read_bytes = fread(normData->getRollingVariance(), sizeof(float), normData->getChannels(), fp);
         if (read_bytes > 0 && read_bytes < normData->getChannels())
            printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", getOrderNumber());
      }
      read_bytes = fread(weightData->getWeights(), sizeof(float), num, fp);
      if (read_bytes > 0 && read_bytes < filters)
         printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", getOrderNumber());

      if (flipped) {
         transposeMatrix(weightData->getWeights(), (inputDimen.channels/groups)*ksize*ksize, filters);
      }
   }


   public$ int getDilation(){
      return dilation;
   }

   public$ int getSize(){
      return ksize;
   }

   public$ int getStride(){
      return stride;
   }

   public$ NLayer *getInputLayer(){
      return input_layer;
   }

   public$ aboolean isAntialiasing(){
      return antialiasing!=0;
   }

   //原型 free_convolutional_batchnorm convolutional_layer.h convolutional_layer.c
   void freeBatchnorm(){
      if (!self->share_layer) {
         if(scaleData){
            scaleData->unref();
            scaleData=NULL;
         }
         if(normData){
            normData->unref();
            normData=NULL;
         }
         if(deltaData){
            deltaData->unref();
            deltaData=NULL;
         }
      }
   }

   void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
      size_t i, counter;
      counter = 0;
      for (i = 0; i < size; i += size / filters) {
         mean_arr[counter++] = fabs(src[i]);
      }
   }

   //原型 binary_align_weights convolutional_layer.h convolutional_layer.c
   void binaryAlignWeights(){
      int m = self->n;   // (l->n / l->groups)
      int k = self->ksize*self->ksize*self->inputDimen.channels;   // ->size*l->size*(l->c / l->groups)
      size_t new_lda = k + (self->lda_align - k % self->lda_align); // (k / 8 + 1) * 8;
      self->new_lda = new_lda;
      float *weights = weightData->getWeights();
      float *binaryWeights=binWeightData->getWeights();
      binarizeWeights/*!binarize_weights*/(weights, m, k, binaryWeights);

      size_t align_weights_size = new_lda * m;
      self->align_bit_weights_size = align_weights_size / 8 + 1;
      float* align_weights = (float*)xcalloc(align_weights_size, sizeof(float));
      self->align_bit_weights = (char*)xcalloc(self->align_bit_weights_size, sizeof(char));

      size_t i, j;
      // align A without transpose
      for (i = 0; i < m; ++i) {
         for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = binaryWeights[i*k + j];
         }
      }

      if (self->inputDimen.channels % 32 == 0){
         int fil, chan;
         const int items_per_filter = self->inputDimen.channels * self->ksize * self->ksize;
         for (fil = 0; fil < self->n; ++fil){
            for (chan = 0; chan < self->inputDimen.channels ; chan += 32){
               const int items_per_channel = self->ksize*self->ksize;
               for (i = 0; i < items_per_channel; ++i){
                  int c_pack;
                  for (c_pack = 0; c_pack < 32; ++c_pack) {
                     float src = binaryWeights[fil*items_per_filter + (chan + c_pack)*items_per_channel + i];
                     //align_weights[fil*items_per_filter + chan*items_per_channel + i * 32 + c_pack] = src;
                     align_weights[fil*new_lda + chan*items_per_channel + i*32 + c_pack] = src;
                  }
               }
            }
         }

         DnnUtils.floatToBit/*!float_to_bit*/(align_weights, (unsigned char*)self->align_bit_weights, align_weights_size);
         int gpu_index=-1;
         if(gpu_index >= 0){
            for (i = 0; i < align_weights_size / 8; ++i)
               self->align_bit_weights[i] = ~(self->align_bit_weights[i]);
         }
         get_mean_array(binaryWeights, m*k, self->n, self->mean_arr);
      }else {
         DnnUtils.floatToBit/*!float_to_bit*/(align_weights, (unsigned char*)self->align_bit_weights, align_weights_size);
         get_mean_array(binaryWeights, m*k, self->n, self->mean_arr);
      }

      free(align_weights);
   }

   // binary transpose
   size_t binary_transpose_align_input(int k, int n, float *b, char **t_bit_input, size_t ldb_align, int bit_align){
       size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
       size_t t_intput_size = new_ldb * bit_align;// n;
       size_t t_bit_input_size = t_intput_size / 8;// +1;
       memset(*t_bit_input, 0, t_bit_input_size * sizeof(char));
       gemm->transpose_bin((auint32*)b, (auint32*)*t_bit_input, k, n, bit_align, new_ldb, 8);
       return t_intput_size;
   }


   //原型 assisted_excitation_forward convolutional_layer.h convolutional_layer.c
   void assistedExcitationForward(NetworkState state){
      NNetwork *net=(NNetwork *)network;
      const int iteration_num = (net->seen) / (net->batch*net->subdivisions);
      float alpha = (1 + cos(3.141592 * iteration_num / net->max_batches));
      if (assisted_excitation > 1) {
         if (iteration_num > assisted_excitation)
            alpha = 0;
         else
            alpha = (1 + cos(3.141592 * iteration_num / assisted_excitation));
      }

      //printf("\n epoch = %f, alpha = %f, seen = %d, max_batches = %d, train_images_num = %d \n",
      //    epoch, alpha, (*state.net.seen), state.net.max_batches, state.net.train_images_num);
      int out_w=outputDimen.w;
      int out_h=outputDimen.h;
      int out_c=outputDimen.channels;
      /*!float *a_avg = (float *)xcalloc(out_w * out_h * batch, sizeof(float));*/
      /*!float *g = (float *)xcalloc(out_w * out_h * batch, sizeof(float));*/
      NData *a_avg=new$ NData(out_w * out_h ,batch);
      NData *g=new$ NData(out_w * out_h ,batch);

      int b;
      int w, h, c;

      int max_boxes/*!l.max_boxes*/ = net->num_boxes;
      int truths/*!l.truths*/ = max_boxes/*!l.max_boxes*/*(4 + 1);

      for (b = 0; b < batch; ++b){
         // calculate G
         float *data=g->getData(b);
         int t;
         for (t = 0; t < net->num_boxes; ++t) {
            Box truth = Box.floatToBoxStride/*!float_to_box_stride*/(
                  (float*)state.truth->truths[b].values[t]/*!state.truth + t*(4 + 1) + b*truths*/, 1);
            if (!truth.x)
               break;  // continue;

            int left = floor((truth.x - truth.w / 2) * out_w);
            int right = ceil((truth.x + truth.w / 2) * out_w);
            int top = floor((truth.y - truth.h / 2) * out_h);
            int bottom = ceil((truth.y + truth.h / 2) * out_h);

            for (w = left; w <= right; w++) {
               for (h = top; h < bottom; h++) {
                  data[w + out_w * h ] = 1;
               }
            }
         }
      }

      for (b = 0; b < batch; ++b){
         // calculate average A
         float *data=a_avg->getData(b);
         float *output=outputData->getData(b);
         for (w = 0; w < out_w; w++) {
            for (h = 0; h < out_h; h++) {
               for (c = 0; c < out_c; c++) {
                  data[w + out_w*h] += output[w + out_w*(h + out_h*c)];
               }
               data[w + out_w*h] /= out_c;  // a_avg / d
            }
         }
      }

      // change activation
      for (b = 0; b < batch; ++b){
         float *avg=a_avg->getData(b);
         float *g0=g->getData(b);
         float *output=outputData->getData(b);
         for (w = 0; w < out_w; w++) {
            for (h = 0; h < out_h; h++) {
               for (c = 0; c < out_c; c++){
                  output[w + out_w*(h + out_h*c)] += alpha * g0[w + out_w*h] * avg[w + out_w*h];
               }
            }
         }
      }
      g->unref();
      a_avg->unref();
   }

  //原型 forward_convolutional_layer convolutional_layer.h convolutional_layer.c
   void forward(NetworkState state){
      int out_h = getOutHeight/*!convolutional_out_height*/();
      int out_w = getOutWidth/*!convolutional_out_width*/();
      int i, j;
      NNetwork *net=(NNetwork *)network;
      ((IOData*)outputData)->setZero();/*!fill_cpu(l.outputs*l.batch, 0, l.output, 1);*/
      float *weights=weightData->getWeights();
     // ((ConvKernel *)weightData)->testprint();
      if (xnor && (!align_bit_weights || state.train)) {
         printf("convolutional foward 00 进入 xnor && (!align_bit_weights || net->state.train) xor:%d\n",xnor);
         if (!align_bit_weights || state.train) {
            float *binaryWeights=binWeightData->getWeights();
            binarizeWeights/*!binarize_weights*/(weights, filters, nweights,binaryWeights);
            printf("\n binarize_weights l.align_bit_weights = %p \n", align_bit_weights);
         }
         swapBinary/*!swap_binary*/();
         state.input->binarize(binary_input);/*!binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);*/
         state.input = binary_input;/*!state.input = l.binary_input;*/
      }
      int m = filters / groups;
      int k = ksize*ksize*inputDimen.channels / groups;
      int n = out_h*out_w;
      int channels=inputDimen.channels;
      int lw=inputDimen.w;
      int lh=inputDimen.h;
      static int u = 0;
      u++;
      //printf("convolutional foward 11 m:%d k:%d n:%d channels:%d lw:%d lh:%d groups:%d\
            // batch_normalize:%d ksize:%d pad:%d dilation:%d sx:%d sy:%d\n",
           // m,k,n,channels,lw,lh,groups,batch_normalize,ksize,pad,dilation,stride_x,stride_y);
      float *output=outputData->getDataArray();
      float *input=state.input->getDataArray();
      for(i = 0; i < batch; ++i){
         for (j = 0; j < groups; ++j){
            float *im = input + (i*groups + j)*channels/groups*lh*lw;
            float *a = weights +j*nweights / groups;
            float *b = state.workspace;
            float *c =output +(i*groups + j)*n*m;/*!l.output +(i*groups + j)*n*m;*/
            //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            //gemm_nn_custom(m, n, k, 1, a, k, b, n, c, n);
            if (xnor && align_bit_weights && !state.train && stride_x == stride_y){
               memset(b, 0, bit_align*ksize*ksize*channels * sizeof(float));
               if (channels % 32 == 0){
                  //printf(" l.index = %d - new XNOR \n", l.index);
                  int ldb_align = lda_align;
                  size_t new_ldb = k + (ldb_align - k%ldb_align);
                  int re_packed_input_size = channels*lw*lh;
                  memset(state.workspace, 0, re_packed_input_size * sizeof(float));
                  const size_t new_c = channels / 32;
                  size_t in_re_packed_input_size = new_c * lw * lh + 1;
                  memset(bin_re_packed_input, 0, in_re_packed_input_size * sizeof(auint32));
                  gemm->repack_input(state.input->getData(i), state.workspace, lw, lh, channels);
                  // 32 x floats -> 1 x auint32
                  DnnUtils.floatToBit/*!float_to_bit*/(state.workspace, (unsigned char *)bin_re_packed_input, channels * lw * lh);

                  im2Col->im2col/*!im2col_cpu_custom*/((float *)bin_re_packed_input, new_c, lh, lw, ksize, stride, pad, state.workspace);
                  int new_k =ksize*ksize*channels / 32;
                  gemm->transpose_uint32((auint32 *)state.workspace, (auint32*)t_bit_input, new_k, n, n, new_ldb);
                  // the main GEMM function
                  gemm->gemm_nn_custom_bin_mean_transposed(m, n, k, 1,
                        (unsigned char*)align_bit_weights, new_ldb, (unsigned char*)t_bit_input, new_ldb, c, n, mean_arr);

               }else{ // else (l.c % 32 != 0)
                  im2Col->im2colCustomBin/*!im2col_cpu_custom_bin*/(state.input->getData(i),
                        channels, lh, lw, ksize, stride, pad, state.workspace, bit_align);

                  {
                     //size_t ldb_align = 256; // 256 bit for AVX2
                     int ldb_align = lda_align;
                     size_t new_ldb = k + (ldb_align - k%ldb_align);
                     size_t t_intput_size = binary_transpose_align_input(k, n, state.workspace, &t_bit_input, ldb_align, bit_align);

                     // 5x times faster than gemm()-float32
                     gemm->gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)align_bit_weights,
                              new_ldb, (unsigned char*)t_bit_input, new_ldb, c, n, mean_arr);

                  }
               }
               outputData->addBias(biasData->getBias());/*!add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);*/
               //activate_array(l.output, m*n*l.batch, l.activation);
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
               else
                  outputData->activate(activation);/*!activate_array_cpu_custom(l.output, m*n*l.batch, l.activation);*/
               return;

            }else{
               //printf(" l.index = %d - FP32 \n", l.index);
               /*!float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;*/
               if (ksize == 1 && stride == 1 && dilation == 1) {
                  b = im;
               }else {
                  //printf("输入数据存入到工作空间作为矩阵 b c:%d lh:%d lw:%d ksize:%d,stride_x:%d stride_y:%d pad:%d dilation:%d space:%p\n",
                       // inputDimen.channels,inputDimen.h, inputDimen.w,ksize,stride_x,stride_y,pad,dilation,b);
                  im2Col->im2col/*!im2col_cpu_ext*/(im,   // input
                        inputDimen.channels / groups,     // input channels
                        inputDimen.h, inputDimen.w,           // input size (h, w)
                        ksize, ksize,     // kernel size (h, w)
                        pad * dilation, pad * dilation,       // padding (h, w)
                        stride_y, stride_x, // stride (h, w)
                        dilation, dilation, // dilation (h, w)
                        b);                 // output
               }
               gemm->gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            }
         }
      }

      if(batch_normalize){
         forwardNorm/*!forward_batchnorm_layer*/(state);
      }else{
         outputData->addBias(biasData->getBias());/*! add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);*/
      }
      //activate_array(l.output, m*n*l.batch, l.activation);
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
      else
         outputData->activate(activation);/*!activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);*/

      if(binary || xnor)
         swapBinary/*!swap_binary*/();

      if(assisted_excitation && state.train)
         assistedExcitationForward/*!assisted_excitation_forward*/(state);

      if (antialiasing) {
         NetworkState s = { 0 };
         s.train = state.train;
         s.workspace = state.workspace;
         //s.net = state.net;
         //s.input = output;
         s.input=(InputData*)outputData;

         input_layer->forward(s);/*!forward_convolutional_layer(*(l.input_layer), s);*/
         //simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
         /*!memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));*/
         ((IOData*)input_layer->outputData)->copy((IOData*)outputData);
      }
   }

   //原型 backward_convolutional_layer convolutional_layer.h convolutional_layer.c
   /*https://www.google.com.hk/search?newwindow=1&sca_esv=cb0b2358a0071c17&biw=1478&bih=746&sxsrf=ANbL-n4_beOw43010sODxApf1Ykxtu5qzw%3A1772787856531&ei=kJiqaY-WIIjh2roPwbDk-AU&ved=2ahUKEwiYpLHhpYuTAxVa1jQHHeB3MNQQ0NsOegQIXxAA&uact=5&sclient=gws-wiz-serp&aep=26&ntc=1&udm=50&mtid=hMuqadazH53k2roP87W6gQ4&mstk=AUtExfBx2ORFkxVeoa-Bn1nryx2ENytxjUih2LoeAOnXkHU34SQJfjqSwdigWQYsTO1ju76Jzrm5V_GaN7d8c3iw3ccazzEu8p2fvqgyBm0yHo-Rr2IR_c3CAkt5d75uWExKS5IM9TdsF9pJNiL9DPcJAxjWQGtdws9fhNdzHNbMRdyBpKInQa4H9IR3n7RPw8PbhIZ2iEPykms8kCNB7nSMIPJiIEKlSaJrTAjeoBmMQUhzq5u-B6O1C-hAyZCMD-6D2l512sIg-__Zxb3d4KayxiK-jTMyfrsIGzgMxO4rUTttMGWptNfMnsxhuCig2AUhl2EbZpUtLpGDRQ&csuir=1&q=softmax&atvm=2
   */
   //求偏差的梯度 ∂C/∂b^lj=δ^lj l是层数，j是第几号偏差
   //输出的大小是out_w,out_h,维度是filters
   //所以偏差有filters个，deltaData中的channels也是filters 所以一个channels对应的out_wxout_h个误差之和才是一个filter的偏差。
   //因为δ^l是高维张量，而b只是一个向量，不能像DNN那样直接和δ^l相等。通常的做法是将δ^l的各个子矩阵的项分别求和，得到一个误差向量，即为b的梯度：
   void backward(NetworkState state){
      int i, j;
      int out_h=outputDimen.h;
      int out_w=outputDimen.w;
      int out_c=outputDimen.channels;

      int m = filters / groups;
      int n = ksize*ksize*inputDimen.channels / groups;
      int k = out_h*out_w;

      int channels=inputDimen.channels;
      int lw=inputDimen.w;
      int lh=inputDimen.h;
      int lc=inputDimen.channels;

      if (activation == ActivationType.SWISH)
         /*! gradient_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.delta);*/
         Activation.gradientArraySwish(outputData->getDataArray(), batch*outputs,
                  activation_input->getDataArray(),deltaData->getDataArray());
      else if (activation == ActivationType.MISH)
         /*!gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);*/
         Activation.gradientArrayMish(batch*outputs, activation_input->getDataArray(),deltaData->getDataArray());
      else if (activation == ActivationType.HARD_MISH)
         /*!gradient_array_hard_mish(l.outputs*l.batch, l.activation_input, l.delta);*/
         Activation.gradientArrayHardMish(batch*outputs,activation_input->getDataArray(),deltaData->getDataArray());
      else if (activation == ActivationType.NORM_CHAN_SOFTMAX || activation == ActivationType.NORM_CHAN_SOFTMAX_MAXVAL)
         /*!gradient_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta);*/
         Activation.gradientArrayNormalizeChannelsSoftmax(outputData->getDataArray(),
               batch*outputs,batch,out_c,out_w*out_h, deltaData->getDataArray());
      else if (activation == ActivationType.NORM_CHAN)
         /*!gradient_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta);*/
         Activation.gradientArrayNormalizeChannels(outputData->getDataArray(),
               batch*outputs,batch,out_c, out_w*out_h,deltaData->getDataArray());
      else
         /*!gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);*/
         Activation.gradientArray(outputData->getDataArray(),batch*outputs,activation,deltaData->getDataArray());


      if (batch_normalize) {
         backwardNorm/*!backward_batchnorm_layer*/(state);
      } else {
         biasData->calcGrad(deltaData);/*! backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);*/
      }

      float *weights=weightData->getWeights();
      float *weight_updates=weightData->getUpdates();
      float *myDelta=deltaData->getDataArray();
      float *delta=state.delta?state.delta->getDataArray():NULL;
      float *input=state.input->getDataArray();

      for (i = 0; i < batch; ++i) {
         for (j = 0; j < groups; ++j) {
            float *a = myDelta + (i*groups + j)*m*k;
            float *b = state.workspace;
            float *c = weight_updates + j*nweights / groups;
            float *im = input + (i*groups + j)* (lc / groups)*lh*lw;
            im2Col->im2col/*!im2col_cpu_ext*/(
               im,                 // input
               lc / groups,     // input channels
               lh, lw,           // input size (h, w)
               ksize, ksize,     // kernel size (h, w)
               pad * dilation, pad * dilation,       // padding (h, w)
               stride_y, stride_x, // stride (h, w)
               dilation, dilation, // dilation (h, w)
               b);                 // output

            gemm->gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

            if (state.delta) {
               a = weights + j*nweights / groups;
               b = myDelta + (i*groups + j)*m*k;
               c = state.workspace;
               gemm->gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
               im2Col->col2im/*!col2im_cpu_ext*/(
                  state.workspace,        // input
                  lc / groups,         // input channels (h, w)
                  lh, lw,               // input size (h, w)
                  ksize, ksize,         // kernel size (h, w)
                  pad * dilation, pad * dilation,           // padding (h, w)
                  stride_y, stride_x,     // stride (h, w)
                  dilation, dilation, // dilation (h, w)
                  delta + (i*groups + j)* (lc / groups)*lh*lw); // output (delta)
            }
         }
      }
   }

   /**
   * https://www.cnblogs.com/pinard/p/6494810.html 已知池化层的δ^l，推导上一隐藏层的δ^l−1
   * https://www.bilibili.com/video/BV1Zg411T71b/?spm_id_from=333.788.recommend_more_video.-1
   * 计算本层的误差值公式。
   * δ^l = ((w^(l+1) )^T δ^(l+1)) ⊙ σ′(z^l)
   * https://blog.csdn.net/qq_16137569/article/details/81477906
   * https://www.cvmart.net/community/detail/1755 解释两种计算上一层误差的方法一种是gemm+col2img 另外一种是权重转置然后卷积输入
   */
   //原型 update_convolutional_layer convolutional_layer.h convolutional_layer.c
   void update(int batch, float learning_rate_init, float momentum, float decay){
       float learning_rate = learning_rate_init*learning_rate_scale;
       /*!axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);*/
       weightData->addWeightToUpdates(-decay*batch);
       /*!axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);*/
       weightData->addUpdatesToWeight(learning_rate / batch);
       /*!scal_cpu(l.nweights, momentum, l.weight_updates, 1);*/
       weightData->scaleUpdates(momentum);
       /*!axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);*/
       biasData->addUpdatesToBias(learning_rate / batch);
       /*!scal_cpu(l.n, momentum, l.bias_updates, 1);*/
       biasData->scaleUpdates(momentum);
       if (scaleData) {
         // printf("conv update is :%d\n",orderNumber);
           /*!axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);*/
          // printf("update conv scal %f,%d,momentum:%f decay:%f\n",learning_rate,batch,momentum,decay);
           scaleData->addUpdatesToScale(learning_rate/batch);
          // printf("update conv scal 11\n");
           /*!scal_cpu(l.n, momentum, l.scale_updates, 1);*/
           scaleData->scaleUpdates(momentum);
       }
   }

   //原型 fuse_conv_batchnorm darknet.h network.c
   public$ void fuseConvBatchnorm(){
     // printf("fuseConvBatchnorm convolutional -batch_normalize 00 \n");
      int f;
      float *biases=biasData->getBias();
      float *scales=scaleData->getScales();
      float *rollingMean=normData->getRollingMean();
      float *rollingVariance=normData->getRollingVariance();
      float *weights=weightData->getWeights();
      //printf("convolutional -batch_normalize 11 - %d weights:%p n:%d %d\n",j,weights,cl->n,cl->filters);
      for (f = 0; f < filters; ++f){
         biases[f] = biases[f] - (double)scales[f] * rollingMean[f] / (sqrt((double)rollingVariance[f] + .00001));
         double precomputed =scales[f] / (sqrt((double)rollingVariance[f] + .00001));
         const size_t filter_size = ksize*ksize*inputDimen.channels /groups;
         int i;
         for (i = 0; i < filter_size; ++i) {
            int w_index = f*filter_size + i;
            weights[w_index] *= precomputed;
            //printf("weights:f:%d i:%d wi:%d %f layer:%d\n",f,i,w_index,weights[w_index],getOrderNumber());
         }
      }
   }
};

