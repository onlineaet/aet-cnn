#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <execinfo.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <aet/time/Time.h>
#include <aet/lang/System.h>
#include <aet/mtcs/MtcsStream.h>
#include "../DnnUtils.h"
#include "MtcsGemm.h"
#include "MtcsTool.h"

static void check_error(cudaError_t status, const char * const filename, const char * const funcname, const int line);
static void check_error_extended(cudaError_t status, const char * const filename, const char * const funcname, const int line);
static void cublas_check_error_extended(cublasStatus_t status, const char * const filename, const char * const funcname, const int line);
#define CHECK_CUDA(X) check_error_extended(X, __FILE__, __func__, __LINE__ );
#define CHECK_CUBLAS(X) cublas_check_error_extended(X, __FILE__, __func__, __LINE__ );

static int cuda_debug_sync = 0;


static void log_backtrace()
{
   void * buffer[50];
   int count = backtrace(buffer, sizeof(buffer));
   char **symbols = backtrace_symbols(buffer, count);

   fprintf(stderr, "backtrace (%d entries)\n", count);

   for (int idx = 0; idx < count; idx ++){
      fprintf(stderr, "%d/%d: %s\n", idx + 1, count, symbols[idx]);
   }

   free(symbols);
}

void error(const char * const msg, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Darknet error location: %s, %s(), line #%d\n", filename, funcname, line);
    perror(msg);
    log_backtrace();
    exit(EXIT_FAILURE);
}

static void check_error(cudaError_t status, const char * const filename, const char * const funcname, const int line)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("\n CUDA Error: %s\n", s);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer, filename, funcname, line);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        char buffer[256];
        printf("\n CUDA Error Prev: %s\n", s);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer, filename, funcname, line);
    }
}


static  void cublas_check_error(cublasStatus_t status)
{
#if defined(DEBUG) || defined(CUDA_DEBUG)
    cudaDeviceSynchronize();
#endif
    if (cuda_debug_sync) {
        cudaDeviceSynchronize();
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Error\n");
    }
}

static  void cublas_check_error_extended(cublasStatus_t status, const char * const filename, const char * const function, const int line)
{
   if (status != CUBLAS_STATUS_SUCCESS) {
      printf("\n cuBLAS status Error in: file: %s function: %s() line: %d\n", filename, function, line);
   }
   #if defined(DEBUG) || defined(CUDA_DEBUG)
   cuda_debug_sync = 1;
   #endif
   if (cuda_debug_sync) {
      cudaError_t status = cudaDeviceSynchronize();
      if (status != CUDA_SUCCESS)
         printf("\n cudaError_t status = cudaDeviceSynchronize() Error in: file: %s function: %s() line: %d\n", filename, function, line);
   }
   cublas_check_error(status);
}

static void check_error_extended(cudaError_t status, const char * const filename, const char * const funcname, const int line)
{
    if (status != cudaSuccess) {
        printf("CUDA status Error: file: %s: func: %s() line: %d\n", filename, funcname, line);
        check_error(status, filename, funcname, line);
    }
#if defined(DEBUG) || defined(CUDA_DEBUG)
    cuda_debug_sync = 1;
#endif
    if (cuda_debug_sync) {
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            printf("CUDA status = cudaDeviceSynchronize() Error: file: %s: func: %s() line: %d\n", filename, funcname, line);
    }
    check_error(status, filename, funcname, line);
}


impl$ MtcsGemm{

   MtcsGemm(){

   }

   void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
           float *A_gpu, int lda,
           float *B_gpu, int ldb,
           float BETA,
           float *C_gpu, int ldc){
      MtcsStream *data=MtcsTool.getStream();
      cudaStream_t stream =  (cudaStream_t)data->getStream();
      cublasHandle_t handle = (cublasHandle_t)blas_handle();
      cudaError_t stream_status = (cudaError_t)cublasSetStream(handle, stream);
      CHECK_CUDA(stream_status);
      cudaError_t status = (cudaError_t)cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
      (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
   }


   static void *blas_handle(){
      if (!blasHandle) {
         MtcsStream *data=MtcsTool.getStream();
         cudaStream_t stream =  (cudaStream_t)data->getStream();
         cublasHandle_t handle;
         CHECK_CUBLAS(cublasCreate(&handle));
         blasHandle = handle;
         cublasStatus_t status = cublasSetStream((cublasHandle_t)blasHandle, stream);
         CHECK_CUBLAS(status);
      }
      return blasHandle;
   }

};

