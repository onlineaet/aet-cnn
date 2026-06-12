

#ifndef __COM_AI_DNN_N_DETECTION_H__
#define __COM_AI_DNN_N_DETECTION_H__

#include <aet.h>

#include "Box.h"

package$ com.ai.dnn;


/**
 * 侦查; 探测; 发现; 察觉;
 * Deep Neural Networks,
 */
public$ class$ NDetection{
	Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
};




#endif /* __N_MEM_H__ */

