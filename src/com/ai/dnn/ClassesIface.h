/*
 * ComputingUnit.c
 *
 *  Created on: 2022年6月1日
 *      Author: sns
 */


#ifndef __COM_AI_DNN_CLASSES_IFACE_H__
#define __COM_AI_DNN_CLASSES_IFACE_H__

#include <aet.h>


package$ com.ai.dnn;

public$ interface$ ClassesIface{
   int  getClasses();
   int  getCoords();
   float *getEmbeddingOutput();
   int  getEmbeddingSize();
};

#endif /* __COM_AI_DNN_CLASSES_IFACE_H__ */
