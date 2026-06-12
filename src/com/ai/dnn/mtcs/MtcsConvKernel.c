#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>

#include "MtcsConvKernel.h"
#include "../DnnUtils.h"
#include "../cnnmicro.h"
#include "MtcsTool.h"

impl$ MtcsConvKernel{

   MtcsConvKernel(int ksize,int channels,int filters,int pad,int stride){
      self->size=channels*filters*ksize*ksize;
      weights = MtcsMem.calloc(size*sizeof(float),TRUE);
      weight_updates =NULL;//只有train=true才需要weight_updates数据  calloc(size, sizeof(float));
      ema=NULL;
      self->ksize=ksize;
      self->channels=channels;
      self->pad=pad;
      self->stride=stride;
      self->filters=filters;
   }

   public$ void   createUpdates(){
      if(!weight_updates)
         weight_updates= MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   //只有train才需要创建ema数据
   public$ void   createEma(){
      if(!ema)
         ema=MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   void initData(ActivationType activation){
      int nweights=channels*filters*ksize*ksize;
      float *hostWeights=malloc(nweights*sizeof(float));
      int i;
      float scale = sqrt(2./(ksize*ksize*channels));
      if (activation == ActivationType.NORM_CHAN || activation == ActivationType.NORM_CHAN_SOFTMAX
      ||activation ==ActivationType.NORM_CHAN_SOFTMAX_MAXVAL) {
         for (i = 0; i < nweights; ++i)
            hostWeights[i] = 1;   // rand_normal();
      }else{
         for (i = 0; i < nweights; ++i)
            hostWeights[i] = scale*DnnUtils.randUniform/*!rand_uniform*/(-1, 1);   // rand_normal();
      }
      MtcsMem.memcpy(weights,hostWeights,nweights*sizeof(float),MtcsCpyKind.HOST2DEV);
      free(hostWeights);
   }

   //原型 rotate_weights_kernel blas_kernels.cu
   __global__  void rotate_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu,
      int nweights, int n, int kernel_size, int reverse){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;
      const int kernel_area = kernel_size * kernel_size;
      const int i = index * kernel_area;

      const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
      const int stage_id = index / stage_step;

      // nweights = (c / groups) * n * size * size;
      // kernel_area = size*size

      if (i < nweights){
         if (stage_id == 0) {
            // simple copy
            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  const int src_i = x + y*kernel_size + i;
                  const int dst_i = x + y*kernel_size + i;
                  if (reverse)
                     weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                  else
                     weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
               }
            }
         } else if (stage_id == 1){
            // 90 degree clockwise rotation - 1
            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  const int src_i = x + y*kernel_size + i;
                  const int dst_i = (kernel_size - 1 - y) + x*kernel_size + i;
                  if (reverse)
                     weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                  else
                     weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
               }
            }
         }else if (stage_id == 2){
            // 180 degree clockwise rotation - 2
            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  const int src_i = x + y*kernel_size + i;
                  const int dst_i = (kernel_size - 1 - x) + (kernel_size - 1 - y)*kernel_size + i;
                  if (reverse)
                     weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                  else
                     weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
               }
            }
         }else if (stage_id == 3){
            // 270 degree clockwise rotation - 3
            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  const int src_i = x + y*kernel_size + i;
                  const int dst_i = y + (kernel_size - 1 - x)*kernel_size + i;
                  if (reverse)
                     weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
                  else
                     weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
               }
            }
         }
      }
   }

   //原型 rotate_weights_gpu blas.h blas_kernels.cu
   void rotate(int reverse){
      const int kernel_area = ksize*ksize;
      const int block_size = MTCS_BLOCK;
      const int num_blocks = MtcsTool.getNumberOfBlocks(size / kernel_area, block_size);
      //if (l.rotate) rotate_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 1);
      //if (l.rotate) rotate_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0);
      float *src=weight_updates;
      float *deform=weight_deform;
      if(!reverse){
        src=weight_deform;
        deform=weights;
      }
      rotate_weights_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>>
            (src, deform, size, filters, ksize, reverse);
   }

   //原型 expand_array_kernel blas_kernels.cu
   __global__  void expand_array_kernel(const float *src_gpu, float *dst_gpu, int current_size, int groups){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index < current_size) {
         for (int i = 0; i < groups; ++i) {
            dst_gpu[index + i*current_size] = src_gpu[index];
         }
      }
   }
   //原型 expand_array_gpu blas.h blas_kernels.cu
   void expandArray(int groups){
      /*expand_array_gpu(l.weights_gpu, l.weight_deform_gpu, l.nweights, 4);*/
      const int current_size = size / groups;
      const int block_size = MTCS_BLOCK;
      const int num_blocks = MtcsTool.getNumberOfBlocks(current_size, block_size);
      expand_array_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>> (weights, weight_deform, current_size, groups);
   }

   //原型 sway_and_flip_weights_kernel  blas_kernels.cu

   __global__  void sway_and_flip_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu,
         int nweights, int n, int kernel_size, int angle, int reverse){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;
      const int kernel_area = kernel_size * kernel_size;
      const int i = index * kernel_area;

      const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
      const int stage_id = index / stage_step;

      // nweights = (c / groups) * n * size * size;
      // kernel_area = size*size

      if (i < nweights){

         if (stage_id == 0) {
            // simple copy
            for (int x = 0; x < kernel_size; ++x) {
               for (int y = 0; y < kernel_size; ++y) {
                  weight_deform_gpu[x + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
               }
            }
         }else if (stage_id == 1 || stage_id == 2){
            // rotate left or right
            if (stage_id == 2)
               angle = -angle;
            if (reverse)
               angle = -angle;

            const float cos_a = cosf(angle * 3.14159265 / 180);
            const float sin_a = sinf(angle * 3.14159265 / 180);
            const int x_c = kernel_size / 2;
            const int y_c = kernel_size / 2;

            float dropout_sum = 0;

            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  // Xsource = x*cos(alpha) + y*sin(alpha)
                  // Ysource = -x*sin(alpha) + y*cos(alpha)

                  float x_s = x_c + (x - x_c)*cos_a + (y - y_c)*sin_a;
                  float y_s = y_c - (x - x_c)*sin_a + (y - y_c)*cos_a;

                  int x_0 = floorf(x_s);   // round down
                  int x_1 = ceilf(x_s);    // round up
                  if (x_0 == x_1)
                     x_1 = x_0 + 1;
                  int y_0 = floorf(y_s);
                  int y_1 = ceilf(y_s);
                  if (y_0 == y_1)
                     y_1 = y_0 + 1;

                  float c_x_0 = x_1 - x_s;
                  float c_x_1 = x_s - x_0;
                  float c_y_0 = y_1 - y_s;
                  float c_y_1 = y_s - y_0;

                  float val = 0;
                  if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size)
                     val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                  else
                     dropout_sum += c_x_0 * c_y_0;

                  if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size)
                     val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                  else
                     dropout_sum += c_x_1 * c_y_0;

                  if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size)
                     val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                  else
                     dropout_sum += c_x_0 * c_y_1;

                  if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size)
                     val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                  else
                     dropout_sum += c_x_1 * c_y_1;

                  weight_deform_gpu[x + y*kernel_size + i] = val;
               }
            }

            // compensate for dropped items
            const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  weight_deform_gpu[x + y*kernel_size + i] *= coef;
               }
            }
         }else if (stage_id == 3){
            // flip
            for (int y = 0; y < kernel_size; ++y) {
               for (int x = 0; x < kernel_size; ++x) {
                  weight_deform_gpu[(kernel_size - x - 1) + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
               }
            }
         }
      }
   }

   //原型 sway_and_flip_weights_gpu blas.h blas_kernels.cu
   void swayAndFlip(int angle,int reverse){
      // sway_and_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);
      // sway_and_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);
      const int kernel_area = ksize*ksize;
      const int block_size = MTCS_BLOCK;
      const int num_blocks = MtcsTool.getNumberOfBlocks(size / kernel_area, block_size);
      float *src=weight_updates;
      float *deform=weight_deform;
      if(!reverse){
         src=weight_deform;
         deform=weights;
      }
      sway_and_flip_weights_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>>
      (src, deform, size, filters, ksize, angle, reverse);
   }

   //原型 stretch_weights_kernel blas_kernels.cu
   __global__  void stretch_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu,
         int nweights, int n, int kernel_size, float scale, int reverse){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       const int kernel_area = kernel_size * kernel_size;
       const int i = index * kernel_area;

       const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
       const int stage_id = index / stage_step;

       // nweights = (c / groups) * n * size * size;
       // kernel_area = size*size

       if (i < nweights){

           if (stage_id == 0) {
               // simple copy
               for (int x = 0; x < kernel_size; ++x) {
                   for (int y = 0; y < kernel_size; ++y) {
                       weight_deform_gpu[x + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                   }
               }
           }else if (stage_id > 0){
               if (stage_id == 1) scale = 0.65;
               else if (stage_id == 2) scale = 0.8;
               else if (stage_id == 3) scale = 1.3;

               if (reverse) scale = 1 / scale;

               const int x_c = kernel_size / 2;
               const int y_c = kernel_size / 2;

               float dropout_sum = 0;

               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       // Xsource = x_c + (x_d - x_c) / scale
                       // Ysource = y_c + (y_d - y_c) / scale

                       float x_s = x_c + (x - x_c) / scale;
                       float y_s = y_c + (y - y_c) / scale;

                       int x_0 = floorf(x_s);   // round down
                       int x_1 = ceilf(x_s);    // round up
                       if (x_0 == x_1) x_1 = x_0 + 1;
                       int y_0 = floorf(y_s);
                       int y_1 = ceilf(y_s);
                       if (y_0 == y_1) y_1 = y_0 + 1;

                       float c_x_0 = x_1 - x_s;
                       float c_x_1 = x_s - x_0;
                       float c_y_0 = y_1 - y_s;
                       float c_y_1 = y_s - y_0;

                       float val = 0;
                       if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size)
                          val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                       else
                          dropout_sum += c_x_0 * c_y_0;

                       if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size)
                          val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                       else
                          dropout_sum += c_x_1 * c_y_0;

                       if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size)
                          val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                       else
                          dropout_sum += c_x_0 * c_y_1;

                       if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size)
                          val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                       else
                          dropout_sum += c_x_1 * c_y_1;

                       weight_deform_gpu[x + y*kernel_size + i] = val;
                   }
               }

               // compensate for dropped items
               //const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       //if (scale < 1) weight_deform_gpu[x + y*kernel_size + i] /= scale;// *= coef;
                       weight_deform_gpu[x + y*kernel_size + i] /= scale;// *= coef;
                   }
               }
           }
       }
   }

   //原型 stretch_weights_gpu blas.h blas_kenrels.cu
   void stretch(float scale, int reverse){
      //stretch_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 0, 1);
      //stretch_weights_gpu(l.weight_deform_gpu, l.weights_gpu,        l.nweights, l.n, l.size, 0, 0);
      const int kernel_area = ksize*ksize;
      const int block_size = MTCS_BLOCK;
      const int num_blocks = MtcsTool.getNumberOfBlocks(size / kernel_area, block_size);
      float *src=weight_updates;
      float *deform=weight_deform;
      if(!reverse){
         src=weight_deform;
         deform=weights;
      }
      stretch_weights_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>>
            (src, deform, size, filters, ksize, scale, reverse);
   }

   __global__  void stretch_sway_flip_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, float angle, int reverse)
   {
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       const int kernel_area = kernel_size * kernel_size;
       const int i = index * kernel_area;

       const int stage_step = (nweights / kernel_area) / 8;  // 8 stages
       const int stage_id = index / stage_step;

       // nweights = (c / groups) * n * size * size;
       // kernel_area = size*size

       if (i < nweights)
       {

           if (stage_id == 0) {
               // simple copy
               for (int x = 0; x < kernel_size; ++x) {
                   for (int y = 0; y < kernel_size; ++y) {
                       weight_deform_gpu[x + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                   }
               }
           }
           else if (stage_id == 1 || stage_id == 2 || stage_id == 3 || stage_id == 4)
           {
               float scale = 0.5;
               if (stage_id == 1) scale = 0.65;
               else if (stage_id == 2) scale = 0.8;
               else if (stage_id == 3) scale = 1.2;
               else if (stage_id == 4) scale = 1.4;

               if (reverse) scale = 1 / scale;

               const int x_c = kernel_size / 2;
               const int y_c = kernel_size / 2;

               float dropout_sum = 0;

               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       // Xsource = x_c + (x_d - x_c) / scale
                       // Ysource = y_c + (y_d - y_c) / scale

                       float x_s = x_c + (x - x_c) / scale;
                       float y_s = y_c + (y - y_c) / scale;

                       int x_0 = floorf(x_s);   // round down
                       int x_1 = ceilf(x_s);    // round up
                       if (x_0 == x_1) x_1 = x_0 + 1;
                       int y_0 = floorf(y_s);
                       int y_1 = ceilf(y_s);
                       if (y_0 == y_1) y_1 = y_0 + 1;

                       float c_x_0 = x_1 - x_s;
                       float c_x_1 = x_s - x_0;
                       float c_y_0 = y_1 - y_s;
                       float c_y_1 = y_s - y_0;

                       float val = 0;
                       if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                       else dropout_sum += c_x_0 * c_y_0;

                       if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                       else dropout_sum += c_x_1 * c_y_0;

                       if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                       else dropout_sum += c_x_0 * c_y_1;

                       if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                       else dropout_sum += c_x_1 * c_y_1;

                       weight_deform_gpu[x + y*kernel_size + i] = val;
                   }
               }

               // compensate for dropped items
               //const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       if(scale > 1)
                           weight_deform_gpu[x + y*kernel_size + i] /= scale;// *= coef;
                   }
               }
           }
           else if (stage_id == 5 || stage_id == 6)
           {
               // rotate left or right
               if (stage_id == 6) angle = -angle;
               if (reverse) angle = -angle;

               const float cos_a = cosf(angle * 3.14159265 / 180);
               const float sin_a = sinf(angle * 3.14159265 / 180);
               const int x_c = kernel_size / 2;
               const int y_c = kernel_size / 2;

               float dropout_sum = 0;

               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       // Xsource = x*cos(alpha) + y*sin(alpha)
                       // Ysource = -x*sin(alpha) + y*cos(alpha)

                       float x_s = x_c + (x - x_c)*cos_a + (y - y_c)*sin_a;
                       float y_s = y_c - (x - x_c)*sin_a + (y - y_c)*cos_a;

                       int x_0 = floorf(x_s);   // round down
                       int x_1 = ceilf(x_s);    // round up
                       if (x_0 == x_1) x_1 = x_0 + 1;
                       int y_0 = floorf(y_s);
                       int y_1 = ceilf(y_s);
                       if (y_0 == y_1) y_1 = y_0 + 1;

                       float c_x_0 = x_1 - x_s;
                       float c_x_1 = x_s - x_0;
                       float c_y_0 = y_1 - y_s;
                       float c_y_1 = y_s - y_0;

                       float val = 0;
                       if (x_0 >= 0 && x_0 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_0 + y_0*kernel_size + i] * c_x_0 * c_y_0;
                       else dropout_sum += c_x_0 * c_y_0;

                       if (x_1 >= 0 && x_1 < kernel_size && y_0 >= 0 && y_0 < kernel_size) val += src_weight_gpu[x_1 + y_0*kernel_size + i] * c_x_1 * c_y_0;
                       else dropout_sum += c_x_1 * c_y_0;

                       if (x_0 >= 0 && x_0 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_0 + y_1*kernel_size + i] * c_x_0 * c_y_1;
                       else dropout_sum += c_x_0 * c_y_1;

                       if (x_1 >= 0 && x_1 < kernel_size && y_1 >= 0 && y_1 < kernel_size) val += src_weight_gpu[x_1 + y_1*kernel_size + i] * c_x_1 * c_y_1;
                       else dropout_sum += c_x_1 * c_y_1;

                       weight_deform_gpu[x + y*kernel_size + i] = val;
                   }
               }

               // compensate for dropped items
               const float coef = (kernel_size*kernel_size) / (kernel_size*kernel_size - dropout_sum);
               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       weight_deform_gpu[x + y*kernel_size + i] *= coef;
                   }
               }
           }
           else if (stage_id == 7)
           {
               // flip
               for (int y = 0; y < kernel_size; ++y) {
                   for (int x = 0; x < kernel_size; ++x) {
                       weight_deform_gpu[(kernel_size - x - 1) + y*kernel_size + i] = src_weight_gpu[x + y*kernel_size + i];
                   }
               }
           }
       }
   }

   //原型 stretch_sway_flip_weights_gpu blas.h blas_kernels.cu
   void stretchSwayFlip(int angle, int reverse){
      //stretch_sway_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);
      //stretch_sway_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);
      const int kernel_area = ksize*ksize;
      const int block_size = MTCS_BLOCK;
      const int num_blocks = MtcsTool.getNumberOfBlocks(size / kernel_area, block_size);
      float *src=weight_updates;
      float *deform=weight_deform;
      if(!reverse){
         src=weight_deform;
         deform=weights;
      }
      stretch_sway_flip_weights_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>>
            (src, deform, size, filters, ksize, angle, reverse);
   }

   //原型 reduce_and_expand_array_kernel blas_kernels.cu
   __global__  void reduce_and_expand_array_kernel(const float *src_gpu, float *dst_gpu, int current_size, int groups){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;

      if (index < current_size) {
         float val = 0;
         for (int i = 0; i < groups; ++i) {
            val += src_gpu[index + i*current_size];
         }
         for (int i = 0; i < groups; ++i) {
            dst_gpu[index + i*current_size] = val / groups;
         }
      }
   }

   //原型 reduce_and_expand_array_gpu blas.h blas_kernels.cu
   void reduceAndExpandArray(int groups){
      //reduce_and_expand_array_gpu(l.weight_deform_gpu, l.weight_updates_gpu, l.nweights, 4);
      const int current_size = size / groups;
      const int block_size = MTCS_BLOCK;
      const int num_blocks = MtcsTool.getNumberOfBlocks(current_size, block_size);

      reduce_and_expand_array_kernel <<<num_blocks, block_size, 0, MtcsTool.getStream() >>>
            (weight_deform, weight_updates, current_size, groups);
   }



};

