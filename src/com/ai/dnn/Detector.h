

#ifndef __COM_AI_DETECTOR_H__
#define __COM_AI_DETECTOR_H__

#include <aet.h>

#include "NNetwork.h"
#include "Business.h"


package$ com.ai.dnn;


/**
 * 检测
 */
public$ class$ Detector extends$ Business{
    private$ char *function;
    Detector(int ngpus,char *datacfg,char *cfgFile, char *weightFile,int argc,char **argv);
    const char *getFunction();
 };




#endif /* __N_MEM_H__ */

