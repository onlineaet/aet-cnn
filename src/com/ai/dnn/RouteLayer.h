

#ifndef __COM_AI_DNN_ROUTE_LAYER_H__
#define __COM_AI_DNN_ROUTE_LAYER_H__

#include <aet.h>
#include  "NLayer.h"
#include  "Activation.h"

package$ com.ai.dnn;
/**
 * 将几个输入层拼接在一起，例如：输入层1: 26*26*256   输入层2： 26*26*128， 则route输出为：26*26*(256+128)；
 */
public$ class$ RouteLayer extends$ NLayer{

    int   * input_layers;//关联层的序号
    int *input_sizes;
    int   layerCount;
    NLayer **inputLayers;
    int groups;
    int group_id;
    int wait_stream_id;
    int stream;
    RouteLayer(int batch, int layerCount,NLayer **layers, int groups, int group_id, apointer network);

    void updataDimension();

};




#endif /* __N_MEM_H__ */

