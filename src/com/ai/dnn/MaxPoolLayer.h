#ifndef __COM_AI_DNN_MAX_POOL_LAYER_H__
#define __COM_AI_DNN_MAX_POOL_LAYER_H__

#include <aet.h>

#include  "NLayer.h"

package$ com.ai.dnn;

/**
 * 计算用的数据，把图像数据转成了float,
 */
public$ class$ MaxPoolLayer extends$ NLayer{

    int stride;
    int size ;
    int pad ;
    int avgpool;
    int antialiasing;
    int stride_x;
    int stride_y;
    int out_channels;
    float bflops;
    int maxpool_depth;
    NLayer *input_layer;
    int *indexes;
    int maxpool_zero_nonmax;
    Gemm *gemm;
    MaxPoolLayer (int batch, int h, int w, int c, int size, int stride_x, int stride_y,
                int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train);
};




#endif /* __N_MEM_H__ */






