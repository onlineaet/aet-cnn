

#ifndef __COM_AI_DNN_SOFT_MAX_LAYER_H__
#define __COM_AI_DNN_SOFT_MAX_LAYER_H__

#include <aet.h>

#include  "NLayer.h"
#include  "OutputLayer.h"


package$ com.ai.dnn;



/**
 * 计算用的数据，把图像数据转成了float,
 */
public$ class$ SoftMaxLayer extends$ NLayer implements$ OutputLayer{
   private$ int groups;
   private$ float temperature;
   private$ int spatial;
   private$ int noloss;
   private$ NData *loss;
   private$ float cost;
   tree *softMaxTree;
   SoftMaxLayer(int batch, int w,int h,int c,float temperature,int spatial,int noloss,
            int inputs, int groups,tree *softMaxTree);
};




#endif /* __N_MEM_H__ */






