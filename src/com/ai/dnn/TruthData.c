#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/mtcs/MtcsMem.h>

#include  "TruthData.h"
#include  "DnnUtils.h"


impl$ TruthData{

   TruthData(int batch,int classes,aboolean useMtcs){
      self->batch=batch;
      self->classes = classes;
      self->useMtcs=useMtcs;
      if(useMtcs)
         self->dataArray=(float*)MtcsMem.calloc(batch*classes*sizeof(float),TRUE);
      else
         self->dataArray=(float*)calloc(batch*classes,sizeof(float));
   }

   void setData(float *data){
      if(useMtcs){
         MtcsMem.memcpy(dataArray,data,batch*classes*sizeof(float),MtcsCpyKind.HOST2DEV);
      }else{
         memcpy(dataArray,data,batch*classes*sizeof(float));
      }
   }

   float *getData(int index){
      return dataArray+index*classes;
   }

   float *getData(int index,int sub){
      if(index>=batch)
         a_error("溢出 index:%d batch:%d",index,batch);
      if(sub>=truths[index].count)
         a_error("溢出 count:%d sub:%d",truths[index].count,sub);
      return truths[index].values[sub];
   }

   public$ void setBox(int count,int index){
      boxes[index]=count;
   }

   public$ int getBoxCount(int index){
      return boxes[index];
   }

   public$ float *getDataArray(){
      return dataArray;
   }


   ~TruthData(){
      if(useMtcs)
         MtcsMem.free(dataArray);
      else
         a_free(dataArray);
   }


};

