

#ifndef __COM_AI_DNN_WEIGHT_DATA_H__
#define __COM_AI_DNN_WEIGHT_DATA_H__

#include <aet.h>
#include "Activation.h"

package$ com.ai.dnn;

// parser.h
typedef enum {
    NO_WEIGHTS, PER_FEATURE, PER_CHANNEL
} WEIGHTS_TYPE_T;

// parser.h
typedef enum {
    NO_NORMALIZATION, RELU_NORMALIZATION, SOFTMAX_NORMALIZATION
} WEIGHTS_NORMALIZATION_T;

/**
 * 权重数据
 */
public$ class$ WeightData {
   // parser.h
   public$ enum$ WEIGHTS_TYPE_T{
      NO_WEIGHTS, PER_FEATURE, PER_CHANNEL
   };
   float  *weight_updates;
   float  *weights;
   float  *ema;
   float  *weight_deform;
   int     size;
   public$        WeightData(int size);
   public$ float *getWeights();
   public$ float *getUpdates();
   public$ float *getDeform();
   public$ int    getSize();
   public$ float *getEma();

   /**
   * 缩放updates
   * 原型  scal_cpu(l.nweights, momentum, l.weight_updates, 1);
   */
   public$ void   scaleUpdates(float scale);
   public$ void   checkNan(char *info);
   public$ void   createEma();
   public$ void   createUpdates();
   public$ int    read(FILE *fp);
   public$ void   setValue(float value);
   public$ void   setUpdates(float *newUpdatas);
   /**
   *把weights加到weight_updates上
   *原型 axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
   */
   public$ void  addWeightToUpdates(float alpha);
   /**
   *把weight_updates加到weights上
   *原型 axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
   */
   public$ void   addUpdatesToWeight(float alpha);
};

/**
 * 卷积核
 */
public$ class$ ConvKernel extends$ WeightData{
   auint   ksize;
   aushort channels;
   auint   pad;
   auint   stride;
   auint   filters;
   float   *rot180Data;//权重转180度后的数据
   public$ ConvKernel(int ksize,int channels,int filters,int pad,int stride);
   public$ ConvKernel(int nweights);
   float *getKernel(int index);
   float *getData(int filter,int channel);
   float *createRot180Matrix();
   void initData(ActivationType activation);
   void testprint();

};


#endif /* __N_MEM_H__ */

