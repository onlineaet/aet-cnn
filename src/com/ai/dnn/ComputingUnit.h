/*
 * ComputingUnit.c
 *
 *  Created on: 2022年6月1日
 *      Author: sns
 */


#ifndef __COM_AI_DNN_COMPUTING_UNIT_H__
#define __COM_AI_DNN_COMPUTING_UNIT_H__

#include <aet.h>


package$ com.ai.dnn;

public$ interface$ ComputingUnit{
   void excuse(int start,int end,apointer userData);
   int  getNeedThreadCount();
   int  getCircleCount();
};

#endif /* __COM_AI_DNN_COMPUTING_UNIT_H__ */
