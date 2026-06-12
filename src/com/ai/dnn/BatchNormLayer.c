#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsStream.h>

#include "BatchNormLayer.h"
#include "DataFactory.h"
#include "mtcs/MtcsTool.h"

impl$ BatchNormLayer{

   /**
   * 构造函数
   * @param batch 一个batch包含图片的张数
   * @param w 图片的高度
   * @param h 图片的宽度
   * @param c 图片的通道数
   * @return
   */
   BatchNormLayer(int batch,int w,int h,int c){
      fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
      self->type = LayerType.BATCHNORM;
      self->batch = batch;
      setInputDimen(w,h,c);
      setOutputDimen(w,h,c);
      outputData=DataFactory.getInstance()->createOutputData(w,h,c,batch);
      deltaData=DataFactory.getInstance()->createDeltaData(w,h,c,batch);
      self->inputs = w*h*c;
      self->outputs = self->inputs;
      scaleData=DataFactory.getInstance()->createScaleData(c);
      scaleData->init(1.0);
      biasData=DataFactory.getInstance()->createBiasData(c);
      normData=DataFactory.getInstance()->createNormData(1,1,c,1);
   }


   void forward(NetworkState state){
      forwardNorm(state);
   }

   void backward(NetworkState state){
      backwardNorm(state);
   }

   void loadWeights(FILE *fp){
      int channels=inputDimen.channels;
      if(devType==DeviceType.MTCS){
         fread(biasData->getBias(), sizeof(float), channels, fp);
         fread(scaleData->getScales(), sizeof(float), channels, fp);
         fread(normData->getRollingMean(), sizeof(float), channels, fp);
         fread(normData->getRollingVariance(), sizeof(float), channels, fp);
      }else{
         float *temp = malloc(workspace_size);
         MtcsCpyKind kind=MtcsCpyKind.HOST2DEV;
         MtcsStream *stream= MtcsTool.getStream();
         fread(temp, sizeof(float),channels, fp);
         MtcsMem.memcpyAsync(biasData->getBias(),temp,channels*sizeof(float),kind,stream);
         fread(temp, sizeof(float),channels, fp);
         MtcsMem.memcpyAsync(scaleData->getScales(),temp,channels*sizeof(float),kind,stream);
         fread(temp, sizeof(float),channels, fp);
         MtcsMem.memcpyAsync(normData->getRollingMean(),temp,channels*sizeof(float),kind,stream);
         fread(temp, sizeof(float),channels, fp);
         MtcsMem.memcpyAsync(normData->getRollingVariance(),temp,channels*sizeof(float),kind,stream);
         free(temp);
      }
   }


   //原型 save_batchnorm_weights parser.c
   void saveWeights(FILE *fp){
      if(devType==DeviceType.MTCS){
         if(!saveData.biases)
            saveData.biases=malloc(biasData->getSize()*sizeof(float));
         if(!saveData.scales)
            saveData.scales=malloc(scaleData->size*sizeof(float));
         if (saveData.rollMean)
            saveData.rollMean=malloc(normData->channels*sizeof(float));
         if (saveData.rollVariance)
            saveData.rollVariance=malloc(normData->channels*sizeof(float));

         MtcsStream *stream=MtcsTool.getStream();
         MtcsCpyKind kind=MtcsCpyKind.DEV2HOST;
         MtcsMem.memcpyAsync(saveData.biases,biasData->getBias(),biasData->getSize()*sizeof(float),kind,stream);
         MtcsMem.memcpyAsync(saveData.scales,scaleData->getScales(),scaleData->size*sizeof(float),kind,stream);
         MtcsMem.memcpyAsync(saveData.rollMean,normData->getRollingMean(),
               normData->channels*sizeof(float),kind,stream);
         MtcsMem.memcpyAsync(saveData.rollVariance,normData->getRollingVariance(),
               normData->channels*sizeof(float),kind,stream);
         stream->sync();
         fwrite(saveData.biases, sizeof(float),biasData->getSize(), fp);
         fwrite(saveData.scales, sizeof(float), scaleData->size, fp);
         fwrite(saveData.rollMean, sizeof(float), normData->channels, fp);
         fwrite(saveData.rollVariance, sizeof(float),normData->channels, fp);
      }else{
         fwrite(biasData->getBias(), sizeof(float),biasData->getSize(), fp);
         fwrite(scaleData->getScales(), sizeof(float), scaleData->size, fp);
         fwrite(normData->getRollingMean(), sizeof(float), normData->channels, fp);
         fwrite(normData->getRollingVariance(), sizeof(float),normData->channels, fp);
      }
   }


};

