#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <aet/lang/AString.h>
#include "Gemm.h"

#define POPCNT(x) __builtin_popcount(x)
#define POPCNT64(x) __builtin_popcountll(x)

static inline unsigned char get_bit(unsigned char const*const src, size_t index)
{
    size_t src_i = index / 8;
    int src_shift = index % 8;
    unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
    //unsigned char val = (src[src_i] & (1 << (8 - src_shift))) > 0;
    return val;
}


static inline auint64 xnor_int64(auint64 a, auint64 b)
{
    return ~(a^b);
}

static inline void set_bit(unsigned char *const dst, size_t index)
{
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
}

static auint8 reverse_8_bit(auint8 a)
{
    return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}


static auint32 reverse_32_bit(auint32 a)
{
    // unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
    return (reverse_8_bit(a >> 24) << 0) |
        (reverse_8_bit(a >> 16) << 8) |
        (reverse_8_bit(a >> 8) << 16) |
        (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

impl$ Gemm {

   void gemmNN(int M, int N, int K, float ALPHA,
       float *A, int lda,
       float *B, int ldb,
       float *C, int ldc){
       int i, j, k;
       for (i = 0; i < M; ++i) {
           for (k = 0; k < K; ++k) {
              register float A_PART = ALPHA * A[i * lda + k];
               for (j = 0; j < N; ++j) {
                   C[i*ldc + j] += A_PART*B[k*ldb + j];
               }
           }
       }
   }

   void gemmNT(int M, int N, int K, float ALPHA,
           float *A, int lda,
           float *B, int ldb,
           float *C, int ldc){
       int i,j,k;
       for(i = 0; i < M; ++i){
           for(j = 0; j < N; ++j){
              register float sum = 0;
               for(k = 0; k < K; ++k){
                   sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
               }
               C[i*ldc+j] += sum;
           }
       }
   }

   void gemmTN(int M, int N, int K, float ALPHA,
           float *A, int lda,
           float *B, int ldb,
           float *C, int ldc){
       int i,j,k;
       for(i = 0; i < M; ++i){
           for(k = 0; k < K; ++k){
              register float A_PART = ALPHA * A[k * lda + i];
               for(j = 0; j < N; ++j){
                   C[i*ldc+j] += A_PART*B[k*ldb+j];
               }
           }
       }
   }

   void gemmTT(int M, int N, int K, float ALPHA,
           float *A, int lda,
           float *B, int ldb,
           float *C, int ldc){
       int i,j,k;
       for(i = 0; i < M; ++i){
           for(j = 0; j < N; ++j){
              register float sum = 0;
               for(k = 0; k < K; ++k){
                   sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
               }
               C[i*ldc+j] += sum;
           }
       }
   }


   void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
               float *A, int lda,
               float *B, int ldb,
               float BETA,
               float *C, int ldc){
      //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
      if (BETA != 1){
         int i, j;
         for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
               C[i*ldc + j] *= BETA;
            }
         }
      }


      int t;
      //#pragma omp parallel for
      for (t = 0; t < M; ++t) {
         if (!TA && !TB)
            gemmNN(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
         else if (TA && !TB)
            gemmTN(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
         else if (!TA && TB)
            gemmNT(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
         else
            gemmTT(1, N, K, ALPHA, A + t, lda, B, ldb, C + t*ldc, ldc);
      }
   }


   //原型 gemm_nn_custom_bin_mean_transposed gemm.h gemm.c
   void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
         unsigned char *A, int lda,
         unsigned char *B, int ldb,
         float *C, int ldc, float *mean_arr){
      int i;

      #pragma omp parallel for
      for (i = 0; i < M; ++i) {   // l.n - filters [16 - 55 - 1024]
         int j, k;
         float mean_val = mean_arr[i];

         for (j = 0; j < N; ++j) { // out_h*out_w - one channel output size [169 - 173056]
            int count = 0;

            for (k = 0; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
               auint64 a_bit64 = *((auint64 *)(A + (i*lda + k) / 8));
               auint64 b_bit64 = *((auint64 *)(B + (j*ldb + k) / 8));
               auint64 c_bit64 = xnor_int64(a_bit64, b_bit64);

               int tmp_count = POPCNT64(c_bit64);

               if (K - k < 64)
                  tmp_count = tmp_count - (64 - (K - k));    // remove extra bits
               count += tmp_count;
            }
            C[i*ldc + j] = (2 * count - K) * mean_val;
         }
      }
   }

   void transpose32_optimized(auint32 A[32]) {
      int j, k;
      unsigned m, t;
      j = 16;
      m = 0x0000FFFF;
      for (k = 0; k < 32; k = (k + j + 1) & ~j) {
         swap(A[k], A[k + j], j, m);
      }

      j = 8;
      m = 0x00ff00ff;
      for (k = 0; k < 32; k = (k + j + 1) & ~j) {
         swap(A[k], A[k + j], j, m);
      }

      j = 4;
      m = 0x0f0f0f0f;
      for (k = 0; k < 32; k = (k + j + 1) & ~j) {
         swap(A[k], A[k + j], j, m);
      }

      j = 2;
      m = 0x33333333;
      for (k = 0; k < 32; k = (k + j + 1) & ~j) {
         swap(A[k], A[k + j], j, m);
      }

      j = 1;
      m = 0x55555555;
      for (k = 0; k < 32; k = (k + j + 1) & ~j) {
         swap(A[k], A[k + j], j, m);
      }

      // reverse Y
      for (j = 0; j < 16; ++j) {
         auint32 tmp = A[j];
         A[j] = reverse_32_bit(A[31 - j]);
         A[31 - j] = reverse_32_bit(tmp);
      }
   }

   void transpose_32x32_bits_reversed_diagonale(auint32 *A, auint32 *B, int m, int n){
      unsigned A_tmp[32];
      int i;
      #pragma unroll
      for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
         transpose32_optimized(A_tmp);
      #pragma unroll
      for (i = 0; i < 32; ++i)
         B[i*n] = A_tmp[i];
   }

   // transpose by 32-bit
   void transpose_bin(auint32 *A, auint32 *B, const int n, const int m,
      const int lda, const int ldb, const int block_size){
      int i;
      #pragma omp parallel for
      for (i = 0; i < n; i += 32) {
         int j;
         for (j = 0; j < m; j += 32) {
            int a_index = i*lda + j;
            int b_index = j*ldb + i;
            transpose_32x32_bits_reversed_diagonale(&A[a_index / 32], &B[b_index / 32], lda / 32, ldb / 32);
         }
         for (; j < m; ++j) {
            if (get_bit((const unsigned char* const)A, i * lda + j))
               set_bit((unsigned char* const)B, j * ldb + i);
         }
      }
   }

   // 32 channels -> 1 channel (with 32 floats)
   // 256 channels -> 8 channels (with 32 floats)
   //原型 repack_input gemm.h gemm.c
   void repack_input(float *input, float *re_packed_input, int w, int h, int c){
      const int items_per_channel = w * h;
      int chan, i;
      for (chan = 0; chan < c; chan += 32){
         for (i = 0; i < items_per_channel; ++i){
            int c_pack;
            for (c_pack = 0; c_pack < 32; ++c_pack) {
               float src = input[(chan + c_pack)*items_per_channel + i];
               re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
            }
         }
      }
   }

   //原型 transpose_uint32 gemm.h gemm.c
   void transpose_uint32(auint32 *src, auint32 *dst, int src_h, int src_w, int src_align, int dst_align){
      int i;
      for (i = 0; i < src_h; i += 1) {
         int j;
         for (j = 0; j < src_w; j += 1){
            ((auint32 *)dst)[j*dst_align / 32 + i] = ((auint32 *)src)[i*src_align + j];
         }
      }
   }

};

