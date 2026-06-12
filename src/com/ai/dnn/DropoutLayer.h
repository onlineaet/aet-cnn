#ifndef __COM_AI_DNN_DROPOUT_LAYER_H__
#define __COM_AI_DNN_DROPOUT_LAYER_H__

#include <aet.h>
#include  "NLayer.h"
#include  "Activation.h"

package$ com.ai.dnn;

/**
 * Deep Neural Networks,
 */
public$ class$ DropoutLayer extends$ NLayer{

   float probability;
   int dropblock;
   float dropblock_size_rel;;
   int dropblock_size_abs;
   float scale;
   float *rand;
   DropoutLayer(int batch, int inputs, float probability, int dropblock,
         float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c);
};




#endif

