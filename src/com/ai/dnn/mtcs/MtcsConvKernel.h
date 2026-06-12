

#ifndef __COM_AI_DNN_MTCS_WEIGHT_DATA_H__
#define __COM_AI_DNN_MTCS_WEIGHT_DATA_H__

#include <aet.h>
#include "../WeightData.h"

package$ com.ai.dnn.mtcs;


/**
 * 卷积核
 */
public$ class$ MtcsConvKernel extends$ ConvKernel{

   public$ MtcsConvKernel(int ksize,int channels,int filters,int pad,int stride);
   //原型 rotate_weights_gpu blas.h blas_kernels.cu
   public$ void rotate(int reverse);
   //原型 expand_array_gpu blas.h blas_kernels.cu
   public$ void expandArray(int groups);
   //原型 sway_and_flip_weights_gpu blas.h blas_kernels.cu
   public$ void swayAndFlip(int angle,int reverse);
   //原型 stretch_weights_gpu blas.h blas_kenrels.cu
   public$ void stretch(float scale, int reverse);
   //原型 stretch_sway_flip_weights_gpu blas.h blas_kernels.cu
   public$ void stretchSwayFlip(int angle, int reverse);
   //原型 reduce_and_expand_array_gpu blas.h blas_kernels.cu
   public$ void reduceAndExpandArray(int groups);
};


#endif /* __N_MEM_H__ */

