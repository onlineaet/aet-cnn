#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <aet/lang/AString.h>
#include <aet/lang/AThread.h>
#include <aet/util/ARandom.h>

#include "DnnUtils.h"


static char *fgetl(FILE *fp)
{
       if(feof(fp))
           return 0;
       size_t size = 512;
       char *line = malloc(size*sizeof(char));
       if(!fgets(line, size, fp)){
           free(line);
           return 0;
       }
       size_t curr = strlen(line);
       while((line[curr-1] != '\n') && !feof(fp)){
           if(curr == size-1){
               size *= 2;
               line = realloc(line, size*sizeof(char));
               if(!line) {
                   printf("%ld\n", size);
                   a_error("Malloc error\n");
               }
           }
           size_t readsize = size-curr;
           if(readsize > INT_MAX)
               readsize = INT_MAX-1;
           fgets(&line[curr], readsize, fp);
           curr = strlen(line);
       }
       if(line[curr-1] == '\n')
           line[curr-1] = '\0';
       return line;
}

impl$ DnnUtils{

   static void delArg(int argc, char **argv, int index){
      int i;
      for(i = index; i < argc-1; ++i)
      argv[i] = argv[i+1];
      argv[i] = NULL;
   }
   /**
   * 从argv中找arg参数
   */
   static char *findArg(int argc, char **argv, char *arg, char *def){
      int i;
      for(i = 0; i < argc-1; ++i){
         if(!argv[i])
            continue;
         if(!strcmp(argv[i], arg)){
            def = argv[i+1];
            delArg(argc, argv, i);
            delArg(argc, argv, i);
            break;
         }
      }
      return def;
   }

   char *findArg(int argc, char**argv, char *arg){
      int i;
      for(i = 0; i < argc; ++i) {
         if(!argv[i])
            continue;
         if(0==strcmp(argv[i], arg)) {
            delArg(argc, argv, i);
            return arg;
         }
      }
      return NULL;
   }

   /*
   ** 计算每个卷积核的偏置更新值，所谓偏置更新值，就是bias = bias - alpha * bias_update中的bias_update
   ** 输入： bias_updates     当前层所有偏置的更新值，维度为l.n（即当前层卷积核的个数）
   **       delta            当前层的敏感度图（即l.delta）
   **       batch            一个batch含有的图片张数（即l.batch）
   **       n                当前层卷积核个数（即l.n）
   **       k                当前层输入特征图尺寸（即l.out_w*l.out_h）
   ** 原理：当前层的敏感度图l.delta是误差函数对加权输入的导数，也就是偏置更新值，只是其中每l.out_w*l.out_h个元素都对应同一个
   **      偏置，因此需要将其加起来，得到的和就是误差函数对当前层各偏置的导数（l.delta的维度为l.batch*l.n*l.out_h*l.out_w,
   **      可理解成共有l.batch行，每行有l.n*l.out_h*l.out_w列，而这一大行又可以理解成有l.n，l.out_h*l.out_w列，这每一小行就
   **      对应同一个卷积核也即同一个偏置）
   */
   static void backwardBias(float *bias_updates, float *delta, int batch, int n, int size){
      int i,b;
      // 遍历batch中每张输入图片
      // 注意，最后的偏置更新值是所有输入图片的总和（多张图片无非就是重复一张图片的操作，求和即可）。
      // 总之：一个卷积核对应一个偏置更新值，该偏置更新值等于batch中所有输入图片累积的偏置更新值，
      // 而每张图片也需要进行偏置更新值求和（因为每个卷积核在每张图片多个位置做了卷积运算，这都对偏置更新值有贡献）以得到每张图片的总偏置更新值。
      for(b = 0; b < batch; ++b){
         for(i = 0; i < n; ++i){
            bias_updates[i] += sum(delta+size*(i+b*n), size);
         }
      }
   }

   static float sum(float *a, int n){
      int i;
      float sum = 0;
      for(i = 0; i < n; ++i)
         sum += a[i];
      return sum;
   }

   static void addBias(float *output, float *biases, int batch, int n, int size){
      int i,j,b;
      for(b = 0; b < batch; ++b){
         for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
               output[(b*n + i)*size + j] += biases[i];
            }
         }
      }
   }

   static void scaleBias(float *output, float *scales, int batch, int n, int size){
      int i,j,b;
      for(b = 0; b < batch; ++b){
         for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
               output[(b*n + i)*size + j] *= scales[i];
            }
         }
      }
   }

   static int *readMap(char *filename){
      int n = 0;
      int *map = 0;
      char *str;
      FILE *file = fopen(filename, "r");
      if(!file)
         a_error("打不开文件:%s\n",filename);
      while((str=fgetl(file))){
         ++n;
         map = realloc(map, n*sizeof(int));
         map[n-1] = atoi(str);
      }
      fclose(file);
      return map;
   }

   static float randScalexx(float s){
      float scale = (float)randUniform(1.0,s);
      int rand=ARandom.getInstance()->nextInt();
      if(rand%2)
         return scale;
      return 1.0/scale;
   }



   static unsigned int x = 123456789, y = 362436069, z = 521288629;

   // Marsaglia's xorshf96 generator: period 2^96-1
   static auint randomGenFast(void){
      unsigned int t;
      x ^= x << 16;
      x ^= x >> 5;
      x ^= x << 1;
      t = x;
      x = y;
      y = z;
      z = t ^ x ^ y;
      return z;
   }

   static float randomFloatFast(){
      return ((float)randomGenFast() / (float)UINT_MAX);
   }

   static float randScale(float s){
       float scale = randUniformStrong(1, s);
       if(randomGen/*!random_gen*/()%2)
          return scale;
       return 1./scale;
   }

   //用UniformDistribution uniform=new$ UniformDistribution(min,max);很慢，内存泄漏
   float randUniform(float min, float max){
      if(max < min){
         float swap = min;
         min = max;
         max = swap;
      }

   #if (RAND_MAX < 65536)
      int rnd = rand()*(RAND_MAX + 1) + rand();
      return ((float)rnd / (RAND_MAX*RAND_MAX) * (max - min)) + min;
   #else
      return ((float)rand() / RAND_MAX * (max - min)) + min;
   #endif
   }



   static int randIntxx(int min, int max){
      if (max < min){
         int s = min;
         min = max;
         max = s;
      }
      int rand=ARandom.getInstance()->nextInt();
      int r = (rand%(max - min + 1)) + min;
      return r;
   }

   static int randInt(int min, int max){
      if (max < min){
         int s = min;
         min = max;
         max = s;
      }
      int r = (randomGen()%(max - min + 1)) + min;
      return r;
   }

   float constrain(float min, float max, float a){
      if (a < min)
         return min;
      if (a > max)
         return max;
      return a;
   }

   int constrain(int a, int min, int max){
      if (a < min)
         return min;
      if (a > max)
         return max;
      return a;
   }

   float* alignAlloc(int size){
      void* ptr = 0;
      int iRet = posix_memalign(&ptr, 32, size);
      assert(0 == iRet);
      return ptr;
   }

   void fileError(const char * const s){
      fprintf(stderr, "Couldn't open file: %s\n", s);
      exit(EXIT_FAILURE);
   }

   static void trim(char *str){
      char* buffer = (char*)xcalloc(8192, sizeof(char));
      sprintf(buffer, "%s", str);
      char *p = buffer;
      while (*p == ' ' || *p == '\t')
         ++p;

      char *end = p + strlen(p) - 1;
      while (*end == ' ' || *end == '\t') {
         *end = '\0';
         --end;
      }
      sprintf(str, "%s", p);
      free(buffer);
   }

   //原型 find_replace utils.h utils.c
   void find_replace(const char* str, char* orig, char* rep, char* output){
      char* buffer = (char*)calloc(8192, sizeof(char));
      char *p;
      sprintf(buffer, "%s", str);
      if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
         sprintf(output, "%s", buffer);
         free(buffer);
         return;
      }
      *p = '\0';
      sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
      free(buffer);
   }

   static char *strlaststr(char *haystack, char *needle){
       char *p = strstr(haystack, needle), *r = NULL;
       while (p != NULL){
           r = p;
           p = strstr(p + 1, needle);
       }
       return r;
   }

   //原型 int_index utils.h utils.c
   static int intIndex(int *a, int val, int n){
      int i;
      for (i = 0; i < n; ++i) {
         if (a[i] == val)
            return i;
      }
      return -1;
   }


   //原型 find_replace_extension utils.c
   static void find_replace_extension(char *str, char *orig, char *rep, char *output){
       char* buffer = (char*)calloc(8192, sizeof(char));

       sprintf(buffer, "%s", str);
       char *p = strlaststr(buffer, orig);
       int offset = (p - buffer);
       int chars_from_end = strlen(buffer) - offset;
       if (!p || chars_from_end != strlen(orig)) {  // Is 'orig' even in 'str' AND is 'orig' found at the end of 'str'?
           sprintf(output, "%s", buffer);
           free(buffer);
           return;
       }

       *p = '\0';
       sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
       free(buffer);
   }
   //原型 replace_image_to_label utils.c
   static void replaceImageToLabel(const char* input_path, char* output_path){
       find_replace(input_path, "/images/train2017/", "/labels/train2017/", output_path);    // COCO
       find_replace(output_path, "/images/val2017/", "/labels/val2017/", output_path);        // COCO
       find_replace(output_path, "/JPEGImages/", "/labels/", output_path);    // PascalVOC
       find_replace(output_path, "\\images\\train2017\\", "\\labels\\train2017\\", output_path);    // COCO
       find_replace(output_path, "\\images\\val2017\\", "\\labels\\val2017\\", output_path);        // COCO

       find_replace(output_path, "\\images\\train2014\\", "\\labels\\train2014\\", output_path);    // COCO
       find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\", output_path);        // COCO
       find_replace(output_path, "/images/train2014/", "/labels/train2014/", output_path);    // COCO
       find_replace(output_path, "/images/val2014/", "/labels/val2014/", output_path);        // COCO

       find_replace(output_path, "\\JPEGImages\\", "\\labels\\", output_path);    // PascalVOC
       //find_replace(output_path, "/images/", "/labels/", output_path);    // COCO
       //find_replace(output_path, "/VOC2007/JPEGImages/", "/VOC2007/labels/", output_path);        // PascalVOC
       //find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/", output_path);        // PascalVOC

       //find_replace(output_path, "/raw/", "/labels/", output_path);
       trim(output_path);

       // replace only ext of files
       find_replace_extension(output_path, ".jpg", ".txt", output_path);
       find_replace_extension(output_path, ".JPG", ".txt", output_path); // error
       find_replace_extension(output_path, ".jpeg", ".txt", output_path);
       find_replace_extension(output_path, ".JPEG", ".txt", output_path);
       find_replace_extension(output_path, ".png", ".txt", output_path);
       find_replace_extension(output_path, ".PNG", ".txt", output_path);
       find_replace_extension(output_path, ".bmp", ".txt", output_path);
       find_replace_extension(output_path, ".BMP", ".txt", output_path);
       find_replace_extension(output_path, ".ppm", ".txt", output_path);
       find_replace_extension(output_path, ".PPM", ".txt", output_path);
       find_replace_extension(output_path, ".tiff", ".txt", output_path);
       find_replace_extension(output_path, ".TIFF", ".txt", output_path);

       // Check file ends with txt:
       if(strlen(output_path) > 4) {
           char *output_path_ext = output_path + strlen(output_path) - 4;
           if( strcmp(".txt", output_path_ext) != 0){
               fprintf(stderr, "Failed to infer label file name (check image extension is supported): %s \n", output_path);
           }
       }else{
           fprintf(stderr, "Label file name is too short: %s \n", output_path);
       }
   }

   #define TOLERANCE 1E-4

   static float compare(int size, float *a,float *b){
      float diff = 0.0;
      int errors=0;
      for (int i=0; i<size; i++ ){
         diff = fabs(a[i]-b[i]);
         if(diff>TOLERANCE) {
            printf("\n 错误的数据: i %d diff %f a:%f b:%f\n", i,  diff,a[i],b[i]);
            errors++;
            if(errors>2)
              exit(0);
         }
      }
      return 0;
   }

   public$ static float compare(int batch,int size, float **a,float **b){
      int i,j;
      float diff = 0.0;
      int errors=0;
      for(i=0;i<batch;i++){
         for (j=0; j<size; j++ ){
            diff = fabs(a[i][j]-b[i][j]);
            if(diff>TOLERANCE) {
               printf("\n 错误的数据: i %d j:%d diff %f a:%f b:%f\n", i,j, diff,a[i][j],b[i][j]);
               errors++;
               if(errors>2)
                  exit(0);
            }
         }
      }
      return 0;
   }

   static float compare(int size, int *a,int *b){
      int diff = 0;
      int errors=0;
      for (int i=0; i<size; i++ ){
         diff = a[i]-b[i];
         if(diff!=0) {
            printf("\n 错误的数据: i %d diff %d a:%d b:%d\n", i,  diff,a[i],b[i]);
            errors++;
            if(errors>2)
              exit(0);
         }
      }
      return 0;
   }


   public$  static void axpy(int n, float alpha, float *src, float *dest){
      int i;
      for(i=0;i<n;i++)
         dest[i]+=alpha*src[i];
   }

   //原型 get_embedding blas.h blas.c
   public$  static void getEmbedding(float *src, int src_w, int src_h, int src_c,
         int embedding_size, int cur_w, int cur_h, int cur_n, int cur_b, float *dst){
      int i;
      for (i = 0; i < embedding_size; ++i) {
         const int src_index = cur_b*(src_c*src_h*src_w) + cur_n*(embedding_size*src_h*src_w) + i*src_h*src_w + cur_h*(src_w) + cur_w;
         const float val = src[src_index];
         dst[i] = val;
      }
   }

   static void softmax(float *input, int n, float temp, float *output, int stride){
      int i;
      float sum = 0;
      float largest = -FLT_MAX;
      for(i = 0; i < n; ++i){
         if(input[i*stride] > largest)
            largest = input[i*stride];
      }
      for(i = 0; i < n; ++i){
         float e = exp(input[i*stride]/temp - largest/temp);
         sum += e;
         output[i*stride] = e;
      }
      for(i = 0; i < n; ++i){
         output[i*stride] /= sum;
      }
   }

   static void topK(float *a, int n, int k, int *index){
      int i,j;
      for(j = 0; j < k; ++j)
         index[j] = -1;
      for(i = 0; i < n; ++i){
         int curr = i;
         for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
               int swap = curr;
               curr = index[j];
               index[j] = swap;
            }
         }
      }
   }

   //原型 float_to_bit gemm.h gemm.c
   void floatToBit (float *src, unsigned char *dst, size_t size){
      size_t dst_size = size / 8 + 1;
      memset(dst, 0, dst_size);

      size_t i;
      char* byte_arr = (char*)calloc(size, sizeof(char));
      for (i = 0; i < size; ++i) {
         if (src[i] > 0)
            byte_arr[i] = 1;
      }

      for (i = 0; i < size; i += 8) {
         char dst_tmp = 0;
         dst_tmp |= byte_arr[i + 0] << 0;
         dst_tmp |= byte_arr[i + 1] << 1;
         dst_tmp |= byte_arr[i + 2] << 2;
         dst_tmp |= byte_arr[i + 3] << 3;
         dst_tmp |= byte_arr[i + 4] << 4;
         dst_tmp |= byte_arr[i + 5] << 5;
         dst_tmp |= byte_arr[i + 6] << 6;
         dst_tmp |= byte_arr[i + 7] << 7;
         dst[i / 8] = dst_tmp;
      }
      free(byte_arr);
   }


};

