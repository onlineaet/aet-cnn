

#ifndef __COM_AI_DNN_UTILS_H__
#define __COM_AI_DNN_UTILS_H__

#include <aet.h>


package$ com.ai.dnn;

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))

public$ class$ DnnUtils{

   public$ static float TWO_PI = 6.2831853071795864769252866f;


   public$ static char  *findArg(int argc, char **argv, char *arg, char *def);
   public$ static char  *findArg(int argc, char **argv, char *arg);
   public$ static void   delArg(int argc, char **argv, int index);
   public$ static void   backwardBias(float *bias_updates, float *delta, int batch, int n, int size);
   public$ static float  sum(float *a, int n);
   public$ static void   addBias(float *output, float *biases, int batch, int n, int size);
   public$ static void   scaleBias(float *output, float *scales, int batch, int n, int size);
   public$ static int   *readMap(char *filename);
   public$ static float  randScale(float s);
   public$ static float  randUniform(float min, float max);
   public$ static int    randInt(int min, int max);
   public$ static float  constrain(float min, float max, float a);
   public$ static int    constrain(int a, int min, int max);
   public$ static float* alignAlloc(int size);
   public$ static void   fileError(const char * const s);
   //原型 find_replace_extension utils.c
   private$ static void  find_replace_extension(char *str, char *orig, char *rep, char *output);
   private$ static void  trim(char *str);
   private$ static char *strlaststr(char *haystack, char *needle);
   //原型 replace_image_to_label utils.c
   public$ static void replaceImageToLabel(const char* input_path, char* output_path);
   //原型 find_replace utils.h utils.c
   public$ static void find_replace(const char* str, char* orig, char* rep, char* output);
   //原型 int_index utils.h utils.c
   public$ static int intIndex(int *a, int val, int n);
   public$ static float compare(int size, float *a,float *b);
   public$ static float compare(int batch,int size, float **a,float **b);
   public$ static float compare(int size, int *a,int *b);
   public$ static void axpy(int n, float alpha, float *dest, float *src);

   //原型 mag_array unitls.h untils.c
   public$ static float magArray(float *a, int n){
      int i;
      float sum = 0;
      for(i = 0; i < n; ++i)
         sum += a[i]*a[i];

      return sqrt(sum);
   }

   //原型 fix_nan_inf yolo_layer.c gaussian_yolo_layer.c
   public$ static inline float fixNanInf(float val){
      if (isnan(val) || isinf(val))
         val = 0;
      return val;
   }

   //原型 clip_value yolo_layer.c gaussian_yolo_layer.c
   public$ static inline float clipValue(float val, const float max_val){
      if (val > max_val)
         val = max_val;
      else if (val < -max_val)
         val = -max_val;
      return val;
   }

   public$ static auint randomGen(){
      unsigned int rnd = 0;
      rnd = rand();
   #if (RAND_MAX < 65536)
      rnd = rand()*(RAND_MAX + 1) + rnd;
   #endif  //(RAND_MAX < 65536)
      return rnd;
   }

   public$ static float randomFloat(){
      unsigned int rnd = 0;
      rnd = rand();
   #if (RAND_MAX < 65536)
      rnd = rand()*(RAND_MAX + 1) + rnd;
      return((float)rnd / (float)(RAND_MAX*RAND_MAX));
   #endif  //(RAND_MAX < 65536)
      return ((float)rnd / (float)RAND_MAX);
   }

   public$ static float randUniformStrong(float min, float max){
      if (max < min) {
         float swap = min;
         min = max;
         max = swap;
      }
      return (randomFloat() * (max - min)) + min;
   }

   //原型 get_embedding blas.h blas.c
   public$  static void getEmbedding(float *src, int src_w, int src_h, int src_c,
         int embedding_size, int cur_w, int cur_h, int cur_n, int cur_b, float *dst);
   public$ static void softmax(float *input, int n, float temp, float *output, int stride);
   public$ static void topK(float *a, int n, int k, int *index);
   //原型 float_to_bit gemm.h gemm.c
   public$ static void floatToBit (float *src, unsigned char *dst, size_t size);
};




#endif /* __N_MEM_H__ */

