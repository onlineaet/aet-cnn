

#ifndef __COM_AI_DNN_UPSAMPLE_LAYER_H__
#define __COM_AI_DNN_UPSAMPLE_LAYER_H__

#include <aet.h>
#include  "NLayer.h"
#include  "Activation.h"


package$ com.ai.dnn;



/**
 * Deep Neural Networks,
 */
public$ class$ UpsampleLayer extends$ NLayer{

    private$ aboolean downSample; //0是上采样 1是下采样
    private$ int stride;//采样步幅
    private$ float scale;//像数值缩放
    UpsampleLayer(int batch, int w, int h, int c, int stride,float scale);
};




#endif /* __N_MEM_H__ */

