#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/util/AKeyFile.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/System.h>
#include <aet/time/Time.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>

#include "DataFactory.h"
#include "mtcs/MtcsIm2Col.h"
#include "mtcs/MtcsGemm.h"
#include "mtcs/MtcsData.h"
#include "mtcs/MtcsDeltaData.h"
#include "mtcs/MtcsIOData.h"
#include "mtcs/MtcsConvKernel.h"
#include "mtcs/MtcsNormData.h"
#include "mtcs/MtcsScaleData.h"
#include "mtcs/MtcsBiasData.h"

#include "cnnmicro.h"

impl$ DataFactory{

   static DataFactory *getInstance(){
      static DataFactory *singleton = NULL;
      if (!singleton){
         singleton =new$ DataFactory();
         singleton->useMtcs=use_mtcs;
      }
      return singleton;
   }

   public$ NData *createData(int size,int batch){
      if(useMtcs)
         return new$ MtcsData(size,batch);
      return new$ NData(size,batch);
   }

   public$ NormData *createNormData(int w,int h,int channels,int batch){
      if(useMtcs)
         return new$ MtcsNormData(w,h,channels,batch);
      return new$ NormData(w,h,channels,batch);
   }

   public$ ScaleData *createScaleData(int size){
      if(useMtcs)
         return new$ MtcsScaleData(size);
      return new$ ScaleData(size);
   }

   public$ DeltaData *createDeltaData(int w,int h,int channels,int batch){
      if(useMtcs)
         return new$ MtcsDeltaData(w,h,channels,batch);
      return new$ DeltaData(w,h,channels,batch);
   }

   public$ DeltaData *createDeltaData(int size,int batch){
      if(useMtcs)
         return new$ MtcsDeltaData(size,batch);
      return new$ DeltaData(size,batch);
   }

   public$ InputData *createInputData(int size,int batch){
      if(useMtcs)
         return new$ MtcsIOData(size,batch);
      return new$ IOData(size,batch);
   }

   public$ InputData *createInputData(int w,int h,int channels,int batch){
      if(useMtcs)
         return new$ MtcsIOData(w,h,channels,batch);
      return new$ IOData(w,h,channels,batch);
   }


   public$ OutputData *createOutputData(int w,int h,int channels,int batch){
      if(useMtcs)
         return new$ MtcsIOData(w,h,channels,batch);
      return new$ IOData(w,h,channels,batch);
   }

   public$ OutputData *createOutputData(int size,int batch){
      if(useMtcs)
         return new$ MtcsIOData(size,batch);
      return new$ IOData(size,batch);
   }


   public$ WeightData *createConvKernel(int ksize,int channels,int filters,int pad,int stride){
      if(useMtcs)
         return new$ MtcsConvKernel(ksize,channels,filters,pad,stride);
      return new$ ConvKernel(ksize,channels,filters,pad,stride);

   }

   public$ BiasData *createBiasData(int size){
      if(useMtcs)
         return new$ MtcsBiasData(size);
      return new$ BiasData(size);

   }

   public$ Im2Col *createIm2Col(){
      if(useMtcs)
         return new$ MtcsIm2Col();
      return new$ Im2Col();

   }

   public$ Gemm *createGemm(){
      if(useMtcs)
         return new$ MtcsGemm();
      return new$ Gemm();

   }

   public$ int *createIndexes(int batch,int size){
      int *indexes=NULL;
      if(useMtcs){
         indexes=(int *)MtcsMem.malloc(batch*size*sizeof(int),TRUE);
      }else{
         indexes=(int *)calloc(batch*size,sizeof(int));
      }
      return indexes;
   }


};

