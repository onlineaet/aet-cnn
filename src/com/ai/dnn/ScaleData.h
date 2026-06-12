#ifndef __COM_AI_DNN_SCALE_DATA_H__
#define __COM_AI_DNN_SCALE_DATA_H__

#include <aet.h>
#include <aet/util/AArray.h>

#include  "NData.h"
#include  "NormData.h"

package$ com.ai.dnn;

/**
 * 伸缩数据
 */
public$ class$ ScaleData{
   float * scales;
   float * scale_updates;
   float * ema;
   int size;
   public$        ScaleData(int size);
   public$ void   init(float value);
   public$ float *getScales();
   public$ float *getUpdates();
   public$ float *getEma();
   public$ void   scaleUpdates(float scale);
   public$ void   addUpdatesToScale(float alpha);
   //原型 backward_scale_cpu blas.h batchnorm_layer.c
   public$ void   backwardScale(NData *delta,NormData *normData);
   public$ void   createEma();
   public$ void   createUpdates();
   public$ int    getSize();
   public$ int    read(FILE *fp);
   public$ ScaleData *clone();
};




#endif /* __N_MEM_H__ */

