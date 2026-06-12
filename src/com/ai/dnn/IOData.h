

#ifndef __COM_AI_DNN_IO_DATA_H__
#define __COM_AI_DNN_IO_DATA_H__

#include <math.h>
#include <aet.h>
#include "NData.h"
#include "Activation.h"

package$ com.ai.dnn;

interface$ InputData{
   public$ void   setImageData(float *imageData);
   public$ float *getData(int index);
   public$ float *getDataArray();
   public$ void   binarize(InputData *binary);
   public$ int    getSize();
};

interface$ OutputData{
   public$ float *getData(int index);
   public$ float *getDataArray();
   public$ float  sum();
   public$ void   resize(int w,int h);
   public$ int    getSize();

   // void normalize(float *mean, float *variance);
   public$ void addBias(float *biases);
   //原型 activate_array_cpu_custom gemm.h gemm.c
   public$ void activate(const ActivationType type);
   //原型 activate_array_swish activations.h activations.c
   public$ void activateArraySwish(OutputData * activation_input);
   //原型 activate_array_mish activations.h activations.c
   public$ void activateArrayMish(OutputData *activation_input);
   //原型 activate_array_hard_mish activations.h activations.c
   public$ void activateArrayHardMish(OutputData *activation_input);
   //原型 activate_array_normalize_channels activations.h activations.c
   public$ void activateArrayNormalizeChannels();
   //原型 activate_array_normalize_channels_softmax activations.h activations.c
   public$ void activateArrayNormalizeChannelsSoftmax(int use_max_val);
};

/**
 * 层的输出数据、输入数据以及误差数据的抽象。
 */
public$ class$ IOData extends$ NData implements$ InputData,OutputData {

   public$ IOData(int w,int h,int channels,int batch);
   public$ IOData(int size,int batch);

};




#endif /* __N_MEM_H__ */

