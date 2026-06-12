

#ifndef __COM_AI_DNN_MTCS_MAX_POOL_LAYER_H__
#define __COM_AI_DNN_MTCS_MAX_POOL_LAYER_H__

#include <aet.h>

#include  "../MaxPoolLayer.h"

package$ com.ai.dnn.mtcs;

/**
 * 卷积运算
 */
public$ class$ MtcsMaxPoolLayer extends$ MaxPoolLayer {
   private$ InputData *input_antialiasing;
   public$ MtcsMaxPoolLayer (int batch, int h, int w, int c, int size, int stride_x, int stride_y,
               int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train);

};




#endif /* __COM_AI_DNN_MTCS_MAX_POOL_LAYER_H__ */

