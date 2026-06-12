#ifndef __COM_AI_DNN_NORM_DATA_H__
#define __COM_AI_DNN_NORM_DATA_H__

#include <aet.h>
#include  "NData.h"

package$ com.ai.dnn;

/**
 * 规一化数据,
 */
public$ class$ NormData{

   public$ static aboolean isInValidNumber(float value){
      return (isnan(value) || isinf(value));
   }
   protected$ int w, h,channels;
   protected$ int batch;
   protected$ float * mean;//每个通道的平均值
   protected$ float * variance;

   float * rolling_mean;
   float * rolling_variance;

   float *x;//输出数据output赋值到x
   float *x_norm;
   protected$ int needThreads;//omp需要的线程数

   public$ NormData(int w,int h,int channels,int batch);
   public$ float *getMean();
   public$ float *getVariance();
   public$ float *getRollingMean();
   public$ float *getRollingVariance();
   public$ void   copyToXNorm(NData*outData);
   public$ void   copyToX(NData*outData);
   public$ void   resize(int w,int h);
   public$ void   copyToMean(float *src);
   public$ void   copyToVarinace(float *src);
   public$ float *getX(int index,int ch);
   public$ float *getX();
   public$ float *getXNorm();
   public$ void   calcVariance(float *outputData);
   public$ void   calcMean(float *outputData);
   public$ int    getWidth();
   public$ int    getHeight();
   public$ int    getChannels();
   public$ int    getBatch();
   public$ void   normalize(NData *outputData,float *scales,float *bias);

   //用滚动平均值和滚动方差生成归一化数据
   public$ void   normalizeByRoll(NData *outputData,float *scales,float *biases);
   public$ void   calcMeanAndVariance(NData *outputData,aboolean merge);
   public$ void   scalAxpyRollMeanAndRollVariance();
   public$ float *getNorm(int index);
   public$ void   calcVarianceAndMean(float *outputData);
   public$ void   createX();
   public$ void   createMean();
   public$ void   createVariance();

};




#endif /* __N_MEM_H__ */

