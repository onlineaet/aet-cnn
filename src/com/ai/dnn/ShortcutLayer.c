#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include "ShortcutLayer.h"
#include "DnnUtils.h"
#include "NNetwork.h"

impl$ ShortcutLayer{
   /**
   * @param batch 包含图片的张数
   * @param w 输入图片的宽度
   * @param h 输入图片的高度
   * @param c 输入图片的通道数
   * @param n 层数
   * @return
   */
   ShortcutLayer(int batch, int n, NLayer **inputLayers, int w, int h, int c,
            float **layers_output_gpu, float **layers_delta_gpu,
            WEIGHTS_TYPE_T weights_type, WEIGHTS_NORMALIZATION_T weights_normalization,
            ActivationType activation, int train){

      fprintf(stderr, "Shortcut Layer: ");
      int i;
      for(i = 0; i < n; ++i)
         fprintf(stderr, "%d, ", inputLayers[i]->getOrderNumber());

      self->type =LayerType.SHORTCUT;
      self->batch = batch;
      self->inputLayers=inputLayers;
      self->train = train;
      self->activation = activation;
      self->n = n;
     // self->input_layers = input_layers;
     // self->input_sizes = input_sizes;
      //self->layers_output = layers_output;
     // self->layers_delta = layers_delta;
      self->weights_type = weights_type;
      self->weights_normalization = weights_normalization;
      self->learning_rate_scale = 1;  // not necessary

      setInputDimen(w,h,c);
      setOutputDimen(w,h,c);
      self->outputs = w*h*c;
      self->inputs =self->outputs;
      if (train)
         deltaData=DataFactory.getInstance()->createDeltaData(w,h,c,batch);

      //l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));
      outputData=DataFactory.getInstance()->createOutputData(w,h,c,batch);

      self->nweights = 0;
      if (weights_type == PER_FEATURE)
         self->nweights = (n + 1);
      else if (weights_type == PER_CHANNEL)
         self->nweights = (n + 1) * c;

      if (self->nweights > 0) {
         weightData = new$ WeightData(self->nweights);/*!(float*)calloc(self->nweights, sizeof(float));*/
         float scale = sqrt(2. / self->nweights);
         weightData->setValue(1.0);
         //for (i = 0; i < self->nweights; ++i)
           // self->weights[i] = 1;// +0.01*rand_uniform(-1, 1);// scale*rand_uniform(-1, 1);   // rand_normal();

         if (train)
            weightData->createUpdates();//self->weight_updates = (float*)calloc(self->nweights, sizeof(float));
         //update = update_shortcut_layer;
      }

      if (activation == ActivationType.SWISH || activation == ActivationType.MISH)
         self->activation_input = DataFactory.getInstance()->createOutputData(self->outputs,batch);

      self->bflops = self->outputs* n / 1000000000.;
      if (weights_type)
         self->bflops *= 2;
      fprintf(stderr, " wt = %d, wn = %d, outputs:%4d x%4d x%4d %5.3f BF\n",
            weights_type, weights_normalization, outputDimen.w, outputDimen.h, outputDimen.channels, bflops);
   }

   void shortcut(int w1, int h1, int c1, NData *addData, int w2, int h2, int c2, float s1, float s2, NData *outData){
      int stride = w1/w2;
      int sample = w2/w1;
      a_assert(stride == h1/h2);
      a_assert(sample == h2/h1);
      if(stride < 1)
         stride = 1;
      if(sample < 1)
         sample = 1;
      int minw = (w1 < w2) ? w1 : w2;
      int minh = (h1 < h2) ? h1 : h2;
      int minc = (c1 < c2) ? c1 : c2;

      int i,j,k,b;
      for(b = 0; b < batch; ++b){
         float *add=addData->getData(b);
         float *out=outData->getData(b);
         for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
               for(i = 0; i < minw; ++i){
                  int out_index = i*sample + w2*(j*sample + h2*(k + c2*b*0));
                  int add_index = i*stride + w1*(j*stride + h1*(k + c1*b*0));
                  out[out_index] = s1*out[out_index] + s2*add[add_index];
               }
            }
         }
      }
   }


   /**
   * 根据误差公式
   *  δ^l = ((w^(l+1) )^T δ^(l+1)) ⊙ σ′(z^l)
   *  通过hadamard乘积 本层的激活函数导数值,
   *  该方法由下一层调用。
   */
   //原型 shortcut_multilayer_cpu blas.h blas.c
   private$ inline float relu(float src) {
       return  (src > 0) ? src:0;
   }




   //原型 backward_shortcut_multilayer_cpu blas.h blas.c
//   backward_shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes,
//       l.layers_delta, state.delta, l.delta, l.weights, l.weight_updates, l.nweights,
//       state.input, l.layers_output, l.weights_normalization);
//   void backward_shortcut_multilayer_cpu(int size, int src_outputs, int n, int *outputs_of_layers,
//   float **layers_delta, float *delta_out, float *delta_in, float *weights, float *weight_updates,
//   int nweights, float *in, float **layers_output, WEIGHTS_NORMALIZATION_T weights_normalization){
   void backwardMultilayerCpu(NetworkState state){
      // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
      const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
      int step = 0;
      int src_outputs = self->outputs;
      if (nweights > 0)
         step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

      int id,b,j;
      #pragma omp parallel for private(b,j,id)
      for (b = 0; b < batch; ++b) {
         int src_b = b;
         float *weights=weightData->getWeights();
         float *weight_updates = weightData->getUpdates();
         float *delta_out=state.delta->getData(b);
         float *delta_in = deltaData->getData(b);
         float *in = state.input->getData(b);
         for(j=0;j<src_outputs;++j){
            int src_i = j;

            //       for (id = 0; id < size; ++id) {
            //           int src_id = id;
            //           int src_i = src_id % src_outputs;
            //           src_id /= src_outputs;
            //           int src_b = src_id;

            float grad = 1, sum = 1, max_val = -FLT_MAX;;
            int i;
            if (weights && weights_normalization) {
               if (weights_normalization == SOFTMAX_NORMALIZATION) {
                  for (i = 0; i < (n + 1); ++i) {
                     const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                     float w = weights[weights_index];
                     if (max_val < w)
                        max_val = w;
                  }
               }
               const float eps = 0.0001;
               sum = eps;
               for (i = 0; i < (n + 1); ++i) {
                  const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                  const float w = weights[weights_index];
                  if (weights_normalization == RELU_NORMALIZATION)
                     sum += relu(w);
                  else if (weights_normalization == SOFTMAX_NORMALIZATION)
                     sum += expf(w - max_val);
               }

               /*
               grad = 0;
               for (i = 0; i < (n + 1); ++i) {
               const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
               const float delta_w = delta_in[id] * in[id];
               const float w = weights[weights_index];
               if (weights_normalization == RELU_NORMALIZATION) grad += delta_w * relu(w) / sum;
               else if (weights_normalization == SOFTMAX_NORMALIZATION) grad += delta_w * expf(w - max_val) / sum;
               }
               */
            }

            if (weights) {
               float w = weights[src_i / step];
               if (weights_normalization == RELU_NORMALIZATION)
                  w = relu(w) / sum;
               else if (weights_normalization == SOFTMAX_NORMALIZATION)
                  w = expf(w - max_val) / sum;
               delta_out[j/*!id*/] += delta_in[j/*!id*/] * w; // [0 or c or (c, h ,w)]
               weight_updates[src_i / step] += delta_in[j/*!id*/] * in[j/*!id*/] * grad;
            } else
               delta_out[j/*!id*/] += delta_in[j/*!id*/];

            // layers
            for (i = 0; i < n; ++i) {
               NLayer *inputLayer=inputLayers[i];
               int add_outputs =inputLayer->outputs/*!outputs_of_layers[i]*/;
               if (src_i < add_outputs) {
                  int add_index = add_outputs*src_b + src_i;
                  int out_index = id;

                  float *layer_delta =inputLayer->deltaData->getData(b)/*!layers_delta[i]*/;
                  if (weights) {
                     float *add = inputLayer->outputData->getData(b)/*!layers_output[i]*/;

                     const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                     float w = weights[weights_index];
                     if (weights_normalization == RELU_NORMALIZATION)
                        w = relu(w) / sum;
                     else if (weights_normalization == SOFTMAX_NORMALIZATION)
                        w = expf(w - max_val) / sum;

                     layer_delta[add_index] += delta_in[j/*!id*/] * w; // [0 or c or (c, h ,w)]
                     weight_updates[weights_index] += delta_in[j/*!id*/] * add[add_index] * grad;
                  }else
                     layer_delta[add_index] += delta_in[j/*!id*/];
               }
            }
         }
      }
   }

   void backward(NetworkState state){
      if (activation == ActivationType.SWISH)
         /*!gradient_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.delta);*/
         Activation.gradientArraySwish(outputData->getDataArray(),batch*outputs,
               activation_input->getDataArray(),deltaData->getDataArray());
      else if (activation == ActivationType.MISH)
         /*!gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);*/
         Activation.gradientArrayMish(outputs*batch, activation_input->getDataArray(),deltaData->getDataArray());
      else
         /*!gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);*/
         Activation.gradientArray(outputData->getDataArray(),outputs*batch,activation,deltaData->getDataArray());

      /*!
      backward_shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes,
      l.layers_delta, state.delta, l.delta, l.weights, l.weight_updates, l.nweights,
      state.input, l.layers_output, l.weights_normalization);
      */
      backwardMultilayerCpu(state);
   }

   void resize(int w, int h){
      a_assert(inputDimen.w == outputDimen.w);
      a_assert(inputDimen.h == outputDimen.h);
      setInputDimen(w,h);
      setOutputDimen(w,h);
      self->outputs = w*h*outputDimen.channels;
      self->inputs = self->outputs;
      outputData->resize(w,h);
      deltaData->resize(w,h);
   }

   void setAlpha(float alpha){
      self->alpha=alpha;
   }

   void setBeta(float beta){
      self->beta=beta;
   }

   //原型 shortcut_multilayer_cpu blas.h blas.c
   void forwardMultilayer(InputData *input,WEIGHTS_NORMALIZATION_T weights_normalization){
      int src_outputs=self->outputs;
     // printf("shortcutMultilayer 00 ---batch:%d %p\n",src_outputs,weightData);
      float *weights=NULL;
      // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
      const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
      int step = 0;
      if (nweights > 0){
         step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1
         weights=weightData->getWeights();
      }
      int id,i,j;
   #pragma omp parallel for private(id,i,j)
      for (id = 0; id < batch; ++id) {
         float *out=outputData->getData(id);
         float *in=input->getData(id);
         int src_b = id;
         //printf("shortcutMultilayer 00 ---batch:%d id:%d\n",batch,id);
         for(j=0;j<src_outputs;j++){
            const int src_i = j;// src_id % src_outputs;
            //src_id /= src_outputs;
            float sum = 1, max_val = -FLT_MAX;
            if (weights && weights_normalization) {
              // printf("shortcutMultilayer 11 ---nomal:%d %d\n",weights_normalization,SOFTMAX_NORMALIZATION);
               if (weights_normalization == SOFTMAX_NORMALIZATION) {
                  for (i = 0; i < (n + 1); ++i) {
                     const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                     float w = weights[weights_index];
                     if (max_val < w)
                        max_val = w;
                  }
               }
               const float eps = 0.0001;
               sum = eps;
               for (i = 0; i < (n + 1); ++i) {
                  const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                  const float w = weights[weights_index];
                  if (weights_normalization == RELU_NORMALIZATION)
                     sum += relu(w);
                  else if (weights_normalization == SOFTMAX_NORMALIZATION)
                     sum += expf(w - max_val);
               }
            }

            if (weights) {
               float w = weights[src_i / step];
               if (weights_normalization == RELU_NORMALIZATION)
                  w = relu(w) / sum;
               else if (weights_normalization == SOFTMAX_NORMALIZATION)
                  w = expf(w - max_val) / sum;

               out[j] = in[id] * w; // [0 or c or (c, h ,w)]
            }else
               out[j] = in[j];

            // layers
            for (i = 0; i < n; ++i) {
               NLayer *layer=inputLayers[i];
               int add_outputs = layer->outputs;
               //int add_outputs = outputs_of_layers[i];
               if (src_i < add_outputs) {
                  int add_index =src_i;// add_outputs*src_b + src_i;
                  int out_index = j;// id;

                  float *add = layer->outputData->getData(src_b);// layers_output[i];

                  if (weights) {
                     const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                     float w = weights[weights_index];
                     if (weights_normalization == RELU_NORMALIZATION)
                        w = relu(w) / sum;
                     else if (weights_normalization == SOFTMAX_NORMALIZATION)
                        w = expf(w - max_val) / sum;

                     out[out_index] += add[add_index] * w; // [0 or c or (c, h ,w)]
                  } else
                     out[out_index] += add[add_index];
               }
            }
         }
      }
   }

   //原型 forward_shortcut_layer shortcut_layer.c
   void forward(NetworkState state){
      NNetwork *net=(NNetwork *)network;
      NLayer *layer=inputLayers[0];
      int from_w = layer->inputDimen.w;/*!state.net.layers[l.index].w;*/
      int from_h = layer->inputDimen.h;/*!state.net.layers[l.index].h;*/
      int from_c = layer->inputDimen.channels;/*! state.net.layers[l.index].c;*/
      int lw=inputDimen.w;
      int lh=inputDimen.h;
      int lc=inputDimen.channels;

      if (nweights == 0 && n == 1 && from_w == lw && from_h == lh && from_c == lc) {
         int size = lw * lh * lc;
         int i,b;
        // printf("shortcut forward 00 fw:%d fh:%d fc:%d w:%d h:%d c:%d nweights:%d 输入层的序号:%d\n",
              // from_w,from_h,from_c,lw,lh,lc,nweights,layer->getOrderNumber());
      #pragma omp parallel for private(b,i)
         for(b=0;b<batch;++b){
            float *output=outputData->getData(b);
            float *input= state.input->getData(b);
            float *loutput= layer->outputData->getData(b);
            for(i = 0; i < size; ++i)
               output[i] = input[i] + loutput[i];
         }
      } else {
         //printf("shortcut forward 11 fw:%d fh:%d fc:%d w:%d h:%d c:%d nweights:%d 输入层的序号:%d\n",
              // from_w,from_h,from_c,lw,lh,lc,nweights,layer->getOrderNumber());
         forwardMultilayer(state.input,weights_normalization);
      }

      //copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
      //shortcut_cpu(l.batch, from_w, from_h, from_c, state.net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);

      //activate_array(l.output, l.outputs*l.batch, l.activation);
      if (activation == ActivationType.SWISH)
         /*!activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);*/
         outputData->activateArraySwish(activation_input);
      else if (activation == ActivationType.MISH)
         /*!activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);*/
         outputData->activateArrayMish(activation_input);
      else
         outputData->activate(activation);/*!activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);*/
//      if(getOrderNumber()==4){
//        outputData->testprint();
//        printf("shortcut --- sizs--- %d\n",outputData->getSize());
//        exit(0);
//      }
   }
   /**
    * 返回每个输入层的在网络的索引号
    */
   public$ int getFirstInputLayerIndex(){
      return inputLayers[0]->getOrderNumber();
   }

//void forward_shortcut_layer_gpu(const layer l, network_state state)
//{
//    //copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
//    //simple_copy_ongpu(l.outputs*l.batch, state.input, l.output_gpu);
//    //shortcut_gpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
//
//    //input_shortcut_gpu(state.input, l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
//
//    //-----------
//    //if (l.outputs == l.input_sizes[0])
//    //if(l.n == 1 && l.nweights == 0)
//    //{
//    //    input_shortcut_gpu(state.input, l.batch, state.net.layers[l.index].w, state.net.layers[l.index].h, state.net.layers[l.index].c,
//    //        state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
//    //}
//    //else
//    {
//        shortcut_multilayer_gpu(l.outputs, l.batch, l.n, l.input_sizes_gpu, l.layers_output_gpu,
//              l.output_gpu, state.input, l.weights_gpu, l.nweights, l.weights_normalization);
//    }
//
//    if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
//    else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
//    else activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
//
//}
//
//extern "C" void shortcut_multilayer_gpu(int src_outputs, int batch, int n, int *outputs_of_layers_gpu, float **layers_output_gpu, float *out, float *in, float *weights_gpu, int nweights, WEIGHTS_NORMALIZATION_T weights_normalization)
//{
//    //printf(" src_outputs = %d, batch = %d, n = %d \n", src_outputs, batch, n);
//   shortcut_multilayer_gpu(l.outputs, l.batch, l.n, l.input_sizes_gpu, l.layers_output_gpu, l.output_gpu, state.input, l.weights_gpu, l.nweights, l.weights_normalization);
//
//    int size = batch * src_outputs;
//    if (nweights == 0 && n == 1) {
//        shortcut_singlelayer_simple_kernel <<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>> (size, src_outputs, batch, n, outputs_of_layers_gpu, layers_output_gpu, out, in, weights_gpu, nweights, weights_normalization);
//    }
//    else {
//        shortcut_multilayer_kernel <<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>> (size, src_outputs, batch, n, outputs_of_layers_gpu, layers_output_gpu, out, in, weights_gpu, nweights, weights_normalization);
//    }
//    CHECK_CUDA(cudaPeekAtLastError());
//}


};

