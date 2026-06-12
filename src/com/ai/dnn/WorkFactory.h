

#ifndef __COM_AI_WORK_FACTORY_H__
#define __COM_AI_WORK_FACTORY_H__

#include <aet.h>
#include "Business.h"


package$ com.ai.dnn;


/**
 * 根据参数，生成不同的工作。如分类，检测，语义分割等。
 */
public$ class$ WorkFactory{
    public$ static WorkFactory *getInstance();
    public$ Business *build(int argc, char **argv);

};




#endif /* __N_MEM_H__ */

