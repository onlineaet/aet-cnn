

#ifndef __COM_AI_CNN_GEMM_H__
#define __COM_AI_CNN_GEMM_H__

#include <aet.h>

package$ com.ai.dnn;

public$ class$ Gemm{

   public$ void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
               float *A, int lda,
               float *B, int ldb,
               float BETA,
               float *C, int ldc);
   public$ void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
         unsigned char *A, int lda,
         unsigned char *B, int ldb,
         float *C, int ldc, float *mean_arr);

   public$ void transpose_bin(auint32 *A, auint32 *B, const int n, const int m,
      const int lda, const int ldb, const int block_size);
   //原型 repack_input gemm.h gemm.c
   public$ void repack_input(float *input, float *re_packed_input, int w, int h, int c);
   //原型 transpose_uint32 gemm.h gemm.c
   public$ void transpose_uint32(auint32 *src, auint32 *dst, int src_h, int src_w, int src_align, int dst_align);

};





#endif /* __N_MEM_H__ */

