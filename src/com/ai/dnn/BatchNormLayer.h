

#ifndef __COM_AI_DNN_BATCH_NORM_LAYER_H__
#define __COM_AI_DNN_BATCH_NORM_LAYER_H__

#include <aet.h>

#include  "NNetwork.h"


package$ com.ai.dnn;


/**
 * Deep Neural Networks,
 */
public$ class$ BatchNormLayer extends$ NLayer{
   struct{
      float *biases;
      float *scales;
      float *rollMean;
      float *rollVariance;
   }saveData;
   BatchNormLayer(int batch,int w,int h,int c);

};




#endif /* __N_MEM_H__ */

