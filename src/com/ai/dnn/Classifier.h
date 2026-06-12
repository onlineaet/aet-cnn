

#ifndef __COM_AI_DNN_CLASSIFIER_H__
#define __COM_AI_DNN_CLASSIFIER_H__

#include <aet.h>

#include "NNetwork.h"
#include "Business.h"

package$ com.ai.dnn;


/**
 * 分类
 * 一幅图像归类,比如:该图片是人脸或数字1
 */
public$ class$ Classifier extends$ Business{
    private$ aboolean isTrain;
    private$ char *function;

    Classifier(int ngpus,char *dataCfg,char *cfgFile,char *weightFile,int argc,char **argv);

};




#endif /* __N_MEM_H__ */

