

#ifndef __COM_AI_DNN_SHORTCUT_LAYER_H__
#define __COM_AI_DNN_SHORTCUT_LAYER_H__

#include <aet.h>
#include  "NLayer.h"
#include  "Activation.h"


package$ com.ai.dnn;



/**
 * Deep Neural Networks,
 */
public$ class$ ShortcutLayer extends$ NLayer{

   private$ float alpha;
   private$ float beta;
   int n;
   int train;
   ActivationType activation;
   //from某层来的数据
   WEIGHTS_TYPE_T weights_type;
   WEIGHTS_NORMALIZATION_T weights_normalization;
   int nweights;
   OutputData * activation_input;
   NLayer **inputLayers;
   //n层数
   ShortcutLayer(int batch, int n, NLayer **inputLayers, int w, int h, int c,
           float **layers_output_gpu, float **layers_delta_gpu,
           WEIGHTS_TYPE_T weights_type, WEIGHTS_NORMALIZATION_T weights_normalization,
           ActivationType activation, int train);

   void setAlpha(float alpha);
   void setBeta(float beta);
   public$ int getFirstInputLayerIndex();

};




#endif /* __N_MEM_H__ */

