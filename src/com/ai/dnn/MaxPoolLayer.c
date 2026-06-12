#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include  "NNetwork.h"
#include "MaxPoolLayer.h"
#include "ConvolutionalLayer.h"
#include  "Gemm.h"

impl$ MaxPoolLayer{

   MaxPoolLayer (int batch, int h, int w, int c, int size, int stride_x, int stride_y,
            int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train){
      self->avgpool = avgpool;
      if (avgpool)
         self->type = LayerType.LOCAL_AVGPOOL;
      else
         self->type =LayerType.MAXPOOL;
      self->train = train;

      const int blur_stride_x = stride_x;
      const int blur_stride_y = stride_y;
      self->antialiasing = antialiasing;
      if (antialiasing) {
         stride_x = stride_y = self->stride = self->stride_x = self->stride_y = 1; // use stride=1 in host-layer
      }

      self->batch = batch;
      self->inputDimen.h = h;
      self->inputDimen.w = w;
      self->inputDimen.channels = c;
      self->pad = padding;
      self->maxpool_depth = maxpool_depth;
      self->out_channels = out_channels;
      if (maxpool_depth) {
         self->outputDimen.channels = out_channels;
         self->outputDimen.w = w;
         self->outputDimen.h = h;
      } else {
         self->outputDimen.w= (w + padding - size) / stride_x + 1;
         self->outputDimen.h = (h + padding - size) / stride_y + 1;
         self->outputDimen.channels  = c;
      }
      self->outputs = outputDimen.h * outputDimen.w * outputDimen.channels;
      self->inputs = h*w*c;
      self->size = size;
      self->stride = stride_x;
      self->stride_x = stride_x;
      self->stride_y = stride_y;
      if (train) {
         if (!avgpool)
            self->indexes =(int*)DataFactory.getInstance()->createIndexes(batch,outputs);// (int*)xcalloc(output_size, sizeof(int));
         deltaData=DataFactory.getInstance()->createDeltaData(outputDimen.w,outputDimen.h,outputDimen.channels,batch);
      }
      outputData=DataFactory.getInstance()->createOutputData(outputDimen.w,outputDimen.h,outputDimen.channels,batch);
      //printf("maxpoolayer outputData:%p %d %d %d %d\n",outputData,outputDimen.w,outputDimen.h,outputDimen.channels,batch);
      self->bflops = (self->size*self->size*inputDimen.channels * outputDimen.h*outputDimen.w) / 1000000000.;
      if (avgpool) {
         if (stride_x == stride_y)
            fprintf(stderr, "avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  size, size, stride_x, w, h, c, outputDimen.w,outputDimen.h,outputDimen.channels, self->bflops);
         else
            fprintf(stderr, "avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  size, size, stride_x, stride_y, w, h, c, outputDimen.w,outputDimen.h,outputDimen.channels, self->bflops);
      } else {
         if (maxpool_depth)
            fprintf(stderr, "max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  size, size, stride_x, w, h, c, outputDimen.w,outputDimen.h,outputDimen.channels, self->bflops);
         else if (stride_x == stride_y)
            fprintf(stderr, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  size, size, stride_x, w, h, c, outputDimen.w,outputDimen.h,outputDimen.channels, self->bflops);
         else
            fprintf(stderr, "max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  size, size, stride_x, stride_y, w, h, c, outputDimen.w,outputDimen.h,outputDimen.channels, self->bflops);
      }
      if (self->antialiasing) {
         printf("AA:  ");
         int blur_size = 3;
         int blur_pad = blur_size / 2;
         if (self->antialiasing == 2) {
            blur_size = 2;
            blur_pad = 0;
         }
         input_layer = new$ ConvolutionalLayer(batch, 1, outputDimen.h, outputDimen.w,
                  outputDimen.channels,  outputDimen.channels,  outputDimen.channels,
                  blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, ActivationType.LINEAR,
                  0, 0, 0, 0, 0, 1, 0, NULL, 0, 0, train);
         const int blur_nweights = outputDimen.channels * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
         int i;
         float *weights=input_layer->weightData->getWeights();
         if (blur_size == 2) {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
               weights[i + 0] = 1 / 4.f;
               weights[i + 1] = 1 / 4.f;
               weights[i + 2] = 1 / 4.f;
               weights[i + 3] = 1 / 4.f;
            }
         } else {
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
         float *biases=input_layer->biasData->getBias();
         for (i = 0; i < outputDimen.channels; ++i)
            biases[i] = 0;
      }
   }

   void forwardAvgPool(NetworkState state){
      int b, i, j, k, m, n;
      int w_offset = -pad / 2;
      int h_offset = -pad / 2;

      int h = outputDimen.h;
      int w = outputDimen.w;
      int c = inputDimen.channels;
      int lw=inputDimen.w;
      int lh=inputDimen.h;
      int lc=inputDimen.channels;
      float *input = state.input->getDataArray();
      float *output = outputData->getDataArray();

      for (b = 0; b < batch; ++b) {
         for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
               for (j = 0; j < w; ++j) {
                  int out_index = j + w*(i + h*(k + c*b));
                  float avg = 0;
                  int counter = 0;
                  for (n = 0; n < size; ++n) {
                     for (m = 0; m < size; ++m) {
                        int cur_h = h_offset + i*stride_y + n;
                        int cur_w = w_offset + j*stride_x + m;
                        int index = cur_w + lw*(cur_h + lh*(k + b*lc));
                        int valid = (cur_h >= 0 && cur_h < lh &&
                        cur_w >= 0 && cur_w < lw);
                        if (valid) {
                           counter++;
                           avg += input[index];
                        }
                     }
                  }
                  output[out_index] = avg / counter;
               }
            }
         }
      }
   }

   /**
   * pad补零在输入数据的右边和底边，不像卷积是补在上下左右。
   */
   void forwardMaxPool(NetworkState state){
      int lw = inputDimen.w;
      int lh = inputDimen.h;
      int lc=inputDimen.channels;
      int out_c = outputDimen.channels;
      float *input = state.input->getDataArray();
      float *output = outputData->getDataArray();

      if (maxpool_depth){
         int b, i, j, k, g;
         for (b = 0; b < batch; ++b) {
            #pragma omp parallel for
            for (i = 0; i < lh; ++i) {
               for (j = 0; j < lw; ++j) {
                  for (g = 0; g < out_c; ++g){
                     int out_index = j + lw*(i + lh*(g + out_c*b));
                     float max = -FLT_MAX;
                     int max_i = -1;

                     for (k = g; k < lc; k += out_c){
                        int in_index = j + lw*(i + lh*(k + lc*b));
                        float val =input[in_index];
                        max_i = (val > max) ? in_index : max_i;
                        max = (val > max) ? val : max;
                     }
                     output[out_index] = max;
                     if (indexes)
                        indexes[out_index] = max_i;
                  }
               }
            }
         }
         return;
      }

      if (!state.train && stride_x == stride_y) {
         /*!forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);*/
         float *src = state.input->getDataArray();
         float *dst =   outputData->getDataArray();
         int w = inputDimen.w;
         int h = inputDimen.h;
         int c = inputDimen.channels;
         int out_w = outputDimen.w;
         int out_h = outputDimen.h;
         int b, k;
         const int w_offset = -pad / 2;
         const int h_offset = -pad / 2;
         for (b = 0; b < batch; ++b) {
           // #pragma omp parallel for
            for (k = 0; k < c; ++k) {
               int i, j, m, n;
               for (i = 0; i < out_h; ++i) {
                  for (j = 0; j < out_w; ++j) {
                     int out_index = j + out_w*(i + out_h*(k + c*b));
                     float max = -FLT_MAX;
                     int max_i = -1;
                     for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                           int cur_h = h_offset + i*stride + n;
                           int cur_w = w_offset + j*stride + m;
                           int index = cur_w + w*(cur_h + h*(k + b*c));
                           int valid = (cur_h >= 0 && cur_h < h && cur_w >= 0 && cur_w < w);
                           float val = (valid != 0) ? src[index] : -FLT_MAX;
                           max_i = (val > max) ? index : max_i;
                           max = (val > max) ? val : max;
                        }
                     }
                     dst[out_index] = max;
                     if (indexes)
                        indexes[out_index] = max_i;
                  }
               }
            }
         }
      }else{
         int b, i, j, k, m, n;
         int w_offset = -pad / 2;
         int h_offset = -pad / 2;

         int h = outputDimen.h;
         int w = outputDimen.w;
         int c = outputDimen.channels;

         for (b = 0; b < batch; ++b) {
            for (k = 0; k < c; ++k) {
               for (i = 0; i < h; ++i) {
                  for (j = 0; j < w; ++j) {
                     int out_index = j + w*(i + h*(k + c*b));
                     float max = -FLT_MAX;
                     int max_i = -1;
                     for (n = 0; n < size; ++n) {
                        for (m = 0; m < size; ++m) {
                           int cur_h = h_offset + i*stride_y + n;
                           int cur_w = w_offset + j*stride_x + m;
                           int index = cur_w + lw*(cur_h + lh*(k + b*lc));
                           int valid = (cur_h >= 0 && cur_h < lh &&
                           cur_w >= 0 && cur_w < lw);
                           float val = (valid != 0) ? input[index] : -FLT_MAX;
                           max_i = (val > max) ? index : max_i;
                           max = (val > max) ? val : max;
                        }
                     }
                     output[out_index] = max;
                     if (indexes)
                        indexes[out_index] = max_i;
                  }
               }
            }
         }
      }

      if (antialiasing) {
         NetworkState s = { 0 };
         s.train = state.train;
         s.workspace = state.workspace;

         OutputData *out=getOutputData();
         s.input = (InputData*)out;

         input_layer->forward(state);
         /*!memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));*/
         ((IOData*)input_layer->outputData)->copy((IOData*)self->outputData);
      }
   }

   void forward(NetworkState state){
      if(avgpool){
         forwardAvgPool(state);
      }else{
         forwardMaxPool(state);
      }
   }

   void resize(int w, int h){
      setInputDimen(w,h);
      self->inputs = h*w*inputDimen.channels;
      int out_w = (w + self->pad - self->size)/self->stride_x + 1;
      int out_h = (h + self->pad - self->size)/self->stride_y + 1;
      setOutputDimen(out_w,out_h);
      self->outputs = out_w *out_h * outputDimen.channels;
      int output_size = self->outputs * batch;
      if (self->train) {
         if (!avgpool)
            indexes = (int*)xrealloc(indexes, output_size * sizeof(int));
         deltaData->resize(out_w,out_h);
      }
      outputData->resize(out_w,out_h);
   }

   /**
   * 根据误差公式
   *  δ^l = ((w^(l+1) )^T δ^(l+1)) ⊙ σ′(z^l)
   *  用本层的误差填充上一层的delta数组，上一层自已再算 σ′(z^l)然后与delta相乘
   *  得到本层的误差。
   */
   void backwardMaxpool(NetworkState state){
      int i;
      int h = outputDimen.h;
      int w = outputDimen.w;
      int c = outputDimen.channels;
      int totalSize=h*w*c*batch;
      float *delta = state.delta->getDataArray();
      float *myDelta = deltaData->getDataArray();
      //#pragma omp parallel for
      for(i = 0; i < totalSize; ++i){
         int index = indexes[i];
         delta[index] += myDelta[i];
      }
   }

   void backwardAvgPool(NetworkState state){
      int b, i, j, k, m, n;
      int w_offset = -pad / 2;
      int h_offset = -pad / 2;

      int h = outputDimen.h;
      int w = outputDimen.w;
      int c = inputDimen.channels;

      int lh = inputDimen.h;
      int lw = inputDimen.w;
      int lc = inputDimen.channels;
      float *delta = state.delta->getDataArray();
      float *myDelta = deltaData->getDataArray();

      for (b = 0; b < batch; ++b) {
         for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
               for (j = 0; j < w; ++j) {
                  int out_index = j + w*(i + h*(k + c*b));
                  for (n = 0; n < size; ++n) {
                     for (m = 0; m < size; ++m) {
                        int cur_h = h_offset + i*stride_y + n;
                        int cur_w = w_offset + j*stride_x + m;
                        int index = cur_w + lw*(cur_h + lh*(k + b*lc));
                        int valid = (cur_h >= 0 && cur_h < lh &&
                        cur_w >= 0 && cur_w < lw);

                        if (valid)
                           delta[index] += myDelta[out_index] / (size*size);
                     }
                  }
               }
            }
         }
      }
   }


   void backward(NetworkState state){
      if(avgpool){
         backwardAvgPool(state);
      }else{
         backwardMaxpool(state);
      }
   }

   ~MaxPoolLayer(){
      if(indexes){
         a_free(indexes);
         indexes = NULL;
      }
   }
};

