#ifndef __COM_AI_DNN_MTCS_NORM_DATA_H__
#define __COM_AI_DNN_MTCS_NORM_DATA_H__

#include <aet.h>

#include "../NormData.h"

package$ com.ai.dnn.mtcs;

/**
 * 规一化数据,
 */
public$ class$ MtcsNormData extends$ NormData{

   public$ MtcsNormData(int w,int h,int channels,int batch);
   public$ void inverseVariance();
   //原型 fast_v_cbn_gpu blas.h blas_kernels.cu
   public$ void fastVcbn(const float *outputs, int minibatch_index, int max_minibatch_index,
            float *m_avg, float *v_avg, const float alpha, int inverse_variance, float epsilon);
   //原型 normalize_scale_bias_gpu blas.h blas_kernels.cu
   public$ void normalize(float *outputs, float *scales, float *biases,  int inverse_variance, float epsilon);

};




#endif /* __N_MEM_H__ */

