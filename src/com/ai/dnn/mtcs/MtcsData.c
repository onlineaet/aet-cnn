#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/mtcs/MtcsMem.h>
#include "MtcsData.h"
#include "../DnnUtils.h"

impl$ MtcsData{

   public$ MtcsData(int size,int batch){
      self->batch=batch;
      self->size=size;
      dataArray = (float*)MtcsMem.malloc(batch*size*sizeof(float),TRUE);
      destroyFunc=MtcsMem.free;
      hostLoss = malloc(batch*size*sizeof(float));

   }

   public$ float sum(){
      MtcsMem.memcpy(hostLoss,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      int i;
      float sums=0;
      int total = batch*size;
      for (i = 0; i < total; ++i)
         sums+=hostLoss[i];
      return sums;
   }


   ~MtcsData(){
      if(hostLoss){
         free(hostLoss);
         hostLoss=NULL;
      }
   }
};

