#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <aet/time/Time.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/System.h>
#include "NData.h"
#include "DnnUtils.h"

static int calcCount=0;
static aint64 scaletime=0;

impl$ NData{

   NData(int w,int h,int channels,int batch){
      self(w,h,channels,batch,TRUE,a_free);
   }

   NData(int w,int h,int channels,int batch,aboolean alloc,ADestroyNotify freeFunc){
      self->w=w;
      self->h=h;
      self->channels=channels;
      self->batch=batch;
      self->size=w*h*channels;
      int i;
      if(alloc){
         dataArray=DnnUtils.alignAlloc(batch*size*sizeof(float));//(float*)malloc(size*sizeof(float));
      }else{
         dataArray=NULL;
      }
      destroyFunc=freeFunc;
      readOnly=FALSE;
      isAlign=FALSE;
      needThreads=System.getCpuCores();
   }

   NData(int w,int h,int channels,int batch,aboolean align){
      self->w=w;
      self->h=h;
      self->channels=channels;
      self->batch=batch;
      self->size=w*h*channels;
      dataArray=DnnUtils.alignAlloc(batch*size*sizeof(float));//(float*)malloc(size*sizeof(float));
      destroyFunc=a_free;
      readOnly=FALSE;
      isAlign=align;
      needThreads=System.getCpuCores();
   }

   /**
   * 没有w,h,channels,只有size和batch
   */
   NData(int size,int batch){
      self->w=0;
      self->h=0;
      self->channels=0;
      self->batch=batch;
      self->size=size;
      dataArray=DnnUtils.alignAlloc(batch*size*sizeof(float));//(float*)malloc(size*sizeof(float));
      destroyFunc=a_free;
      readOnly=FALSE;
      isAlign=FALSE;
      needThreads=System.getCpuCores();
   }

   NData(int size,float *data){
      self->w=0;
      self->h=0;
      self->channels=0;
      self->batch=1;
      self->size=size;
      dataArray= DnnUtils.alignAlloc(batch*size*sizeof(float));
      memcpy(dataArray,data,batch*size*sizeof(float));
      destroyFunc=a_free;
      readOnly=FALSE;
      isAlign=FALSE;
      needThreads=System.getCpuCores();
   }

   int getSize(){
      return size;
   }

   int getWidth(){
      return w;
   }

   int getHeight(){
      return h;
   }

   int  getChannels(){
      return channels;
   }

   int  getBatch(){
      return batch;
   }

   /**
   * 引用外部的数据，对象初始化时不分配内存。
   * 例如：ConvolutionalLayer使用方方法。
   */
   void  setData(float *data,int index){
      if(index>batch){
         a_error("错误:index>batch :%d>%d",index,batch);
      }
      char *d=(char*)dataArray;
      memcpy(d+index*size*sizeof(float),data,size*sizeof(float));
   }

   /**
   * 用src数据重构本身，重要的是不能复制数据。不能free dataArray。
   */
   void setData(NData *src){
      self->batch=src->getBatch();
      self->size=src->getSize();
      self->w=src->getWidth();
      self->h=src->getWidth();
      self->channels=src->getChannels();
      memcpy(dataArray,src->dataArray,batch*size*sizeof(float));
   }

   /**
   * 返回一张图对应的数据
   */
   float *getData(int index){
      if(index>batch){
         a_error("错误:索引溢出 :%d",index);
      }
      return  (float*)(dataArray+index*size);
   }

//   float   *getData(int index,int ch){
//      if(index>=batch || ch>=self->channels){
//         a_error("错误:索引溢出 :batch:%d %d channels:%d %d",batch,index,channels,ch);
//      }
//      float *subData=dataArray+index*size+w*h*ch;
//      return (float*)subData;
//   }


   void resize(int w,int h){
      if(readOnly){
         a_error("不能resize。");
         return;
      }
      self->w=w;
      self->h=h;
      size=w*h*channels;
      destroyFunc((char*)dataArray);
      dataArray=DnnUtils.alignAlloc(batch*size*sizeof(float));//(float*)malloc(size*sizeof(float));
   }

   void resize(int w,int h,int channels){
      if(readOnly){
         a_error("不能resize。");
         return;
      }
      self->w=w;
      self->h=h;
      self->channels=channels;
      size=w*h*channels;
      destroyFunc((char*)dataArray);
      dataArray=DnnUtils.alignAlloc(batch*size*sizeof(float));//(float*)malloc(size*sizeof(float));
   }

   void resize(int size){
      if(readOnly){
         a_error("不能resize。");
         return;
      }
      self->size=size;
      destroyFunc((char*)dataArray);
      dataArray=DnnUtils.alignAlloc(batch*size*sizeof(float));//(float*)malloc(size*sizeof(float));
   }

   float *getDataArray(){
      return dataArray;
   }

   aboolean compare(NData *obj){
      if(obj==NULL)
         return FALSE;
      if(w!=obj->getWidth() || h!=obj->getHeight() || channels!=obj->getChannels() || batch!=obj->getBatch())
         return FALSE;
      return TRUE;
   }


   void copy(NData *dest,int index,int destOffset){
      if(index>batch){
         a_error("数组下标溢出。batch:%d index:%d",batch,index);
         return;
      }
      char *destData=(char*)dest->getData(index)+destOffset*sizeof(float);
      if(destOffset>size){
         a_error("出错了----destData size :%d src size:%d destOffset:%d\n",dest->getSize(),size,destOffset);
      }
      char *srcData=(char*)dataArray+index*size*sizeof(float)+destOffset*sizeof(float);
      memcpy(destData,srcData,(size-destOffset)*sizeof(float));
   }

   /**
   * 把本身数据复制到dest
   */
   aboolean copy(NData *dest){
      aboolean ret= compare(dest);
      if(dest->readOnly){
         a_error("目标是只读对象。");
         return FALSE;
      }
      if(!ret){
         a_warning("不能复制数据到dest。参数不匹配！");
         return FALSE;
      }
      if(size!=dest->getSize()){
         a_error("目标是与源的大小不匹配。dest:%d src:%d",dest->getSize(),size);
         return FALSE;
      }
      float *destArray=dest->getDataArray();
      memcpy(destArray,dataArray,batch*size*sizeof(float));
      return TRUE;
   }

   /**
   * 所有数据设为零
   */
   void  setZero(){
      memset(dataArray,0,batch*size*sizeof(float));
   }

   //原型 scale_bias blas.h convolutional_layer.c
   //用scales更新data
   void scale(float *scales){
      int i,j,b;
      int spatial=w*h;
//#pragma omp parallel for num_threads( needThreads ) private( b,i,j)
      for(b = 0; b < batch; ++b){
         float *data=dataArray+b*size;
         for(i = 0; i < channels; ++i)
            for(j = 0; j < spatial; ++j)
               data[i*spatial + j] *= scales[i];
      }
   }

   public$  void  scaleBias(float *scales){
      scale(scales);
   }

   void scaleAvx(float *scales){
      int i,j,b;
      int spatial=w*h;
   #pragma omp parallel for num_threads( needThreads ) private( b,i,j)
      for(b = 0; b < batch; ++b){
         float *data=dataArray+b*size;
         __m256 result,mmscales;
         for(i = 0; i < channels; ++i){
            __m256 mmscales = _mm256_set1_ps(scales[i]);
            for(j = 0; j < size-8; j+=8){
               result = _mm256_loadu_ps(&data[i*spatial + j]);
               result= _mm256_mul_ps(result,mmscales);
               _mm256_storeu_ps(&data[i*spatial + j],result);
            }
            for(; j < spatial; j++)
               data[i*spatial + j] *= scales[i];
         }
      }
   }

   /**
   * 本身数据乘上scale后再加偏置
   * 把两个操作合成一个可以提升性能
   */
   void scaleAndAddBias(float *scales,float *biases){
      int i,j,b;
      int spatial=w*h;
   #pragma omp parallel for num_threads( needThreads ) private( b,i,j ) schedule(dynamic,1)
      for(b = 0; b < batch; ++b){
         float *data=dataArray+b*size;
         for(i = 0; i < channels; ++i)
            for(j = 0; j < spatial; ++j)
               data[i*spatial + j] =  data[i*size + j] *scales[i]+biases[i];
      }
   }

   void checkNan(char *info,int layer){
      if(5>3)
         return;
      int b,n;
      for (b = 0; b < self->batch; ++b) {
         float *item=dataArray+b*size;
         for(n = 0; n < size; ++n){
            if(isnan(item[n]) || isinf(item[n])){
               printf("%s 层数:%d data b:%d n:%d isnan %f\n",info,layer,b,n,item[n]);
               exit(0);
            }
         }
      }
   }

   void checkNan(char *info,int layer,int index){
      if(5>3)
         return;
      int b,n;
      float *item=dataArray+index*size;
      for(n = 0; n < size; ++n){
         if(isnan(item[n])|| isinf(item[n])){
            printf("%s 层数:%d data b:%d n:%d isnan %f\n",info,layer,index,n,item[n]);
            exit(0);
         }
      }
   }

   /**
    * 求和 原型 sum_array utils.h utils.c
    */
   public$ float sum(){
      int i;
      float sums=0;
      int total = batch*size;
      for (i = 0; i < total; ++i)
         sums+=dataArray[i];
      return sums;
   }


   char *toString(){
      sprintf(toString,"w:%d h:%d batch:%d channels:%d size:%d\n",w,h,batch,channels,size);
      return toString;
   }

   NData *clone(){
      NData *newData=NULL;
      if(w>0 || h>0 || channels>0)
         newData=new$ NData(w,h,channels,batch);
      else
         newData=new$ NData(size,batch);
      if(!copy(newData)){
         newData->unref();
         a_error("克隆 NData出错。");
      }
      return newData;
   }

   ~NData(){
      //printf("NData 释放内存:%p destroyFunc:%p batch:%d readOnly:%d ref:%d\n",self,destroyFunc,batch,readOnly,getRefCount());
      if(readOnly)
         return;
      if(dataArray!=NULL){
         destroyFunc (dataArray);
         dataArray=NULL;
      }
   }
};

