#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include "DropoutLayer.h"
#include "DnnUtils.h"
#include "NNetwork.h"

impl$ DropoutLayer{

   DropoutLayer(int batch, int inputs, float probability, int dropblock,
            float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c){
      self->type = LayerType.DROPOUT;
      self->probability = probability;
      self->dropblock = dropblock;
      self->dropblock_size_rel = dropblock_size_rel;
      self->dropblock_size_abs = dropblock_size_abs;
      if (dropblock) {
         outputDimen.w=inputDimen.w=w;
         outputDimen.h=inputDimen.h=h;
         outputDimen.channels=inputDimen.channels=c;
         if (w <= 0 || h <= 0 || c <= 0) {
            printf(" Error: DropBlock - there must be positive values for: l.w=%d, l.h=%d, l.c=%d \n", w, h, c);
            a_error("Error!");
         }
      }
      self->inputs = inputs;
      self->outputs = inputs;
      self->batch = batch;
      self->rand = (float*)xcalloc(inputs * batch, sizeof(float));
      self->scale = 1./(1.0 - probability);
      // l.forward = forward_dropout_layer;
      // l.backward = backward_dropout_layer;
   #ifdef GPU
      l.forward_gpu = forward_dropout_layer_gpu;
      l.backward_gpu = backward_dropout_layer_gpu;
      l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
      if (l.dropblock) {
         l.drop_blocks_scale = cuda_make_array_pinned(l.rand, l.batch);
         l.drop_blocks_scale_gpu = cuda_make_array(l.rand, l.batch);
      }
   #endif
      if (dropblock) {
         if(dropblock_size_abs)
            fprintf(stderr, "dropblock    p = %.3f   l.dropblock_size_abs = %d    %4d  ->   %4d\n",
                  probability, dropblock_size_abs, inputs, inputs);
         else
            fprintf(stderr, "dropblock    p = %.3f   l.dropblock_size_rel = %.2f    %4d  ->   %4d\n",
                  probability, dropblock_size_rel, inputs, inputs);
      } else
         fprintf(stderr, "dropout    p = %.3f        %4d  ->   %4d\n", probability, inputs, inputs);
   }

   void forward(NetworkState state){
      int i,j;
      if (!state.train)
         return;
      for(i = 0; i <batch; ++i){
         float *input=state.input->getData(i);
         for(j=0;j<inputs;++j){
            float r = DnnUtils.randUniform/*!rand_uniform*/(0, 1);
            self->rand[j+i*inputs] = r;
            if(r < probability)
               input[j] = 0;
            else
               input[j] *= scale;
         }
      }
   }

   void backward(NetworkState state){
      int i,j;
      if(!state.delta)
         return;
      for(i = 0; i < batch; ++i){
         float *delta=state.delta->getData(i);
         for(j=0;j<inputs;++j){
            float r = self->rand[j+i*inputs];
            if(r < probability)
               delta[j] = 0;
            else
               delta[j] *= scale;
         }
      }
   }
};

