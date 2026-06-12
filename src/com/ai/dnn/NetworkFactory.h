

#ifndef __COM_AI_NETWORK_FACTORY_H__
#define __COM_AI_NETWORK_FACTORY_H__

#include <aet.h>
#include "NLayer.h"
#include "NNetwork.h"
#include "ConfigFile.h"

package$ com.ai.dnn;
/**
 * 创建网络的工厂
 */
public$ class$ NetworkFactory{
    private$ ConfigFile *cfgFile;
    private$ aboolean useMtcs;
    private$ NetworkFactory();
    static NetworkFactory *getInstance();
    public$ NNetwork      *createNetwork(char *cfgFile,int batch,int time_steps);
    public$ NNetwork      *createTrainNetwork(char *cfgFileName);

};




#endif /* __N_MEM_H__ */

