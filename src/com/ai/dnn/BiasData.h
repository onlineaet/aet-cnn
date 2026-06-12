#ifndef __COM_AI_DNN_BIAS_DATA_H__
#define __COM_AI_DNN_BIAS_DATA_H__

#include <aet.h>

#include <aet/util/AArray.h>
#include  "NData.h"

package$ com.ai.dnn;

/**
 * 规一化数据,
 */
public$ class$ BiasData{
   float * bias;
   float * updates;//偏差的梯度 darknet中就是bias_updates
   float * ema;
   int size;
   public$           BiasData(int size);
   public$ void      calcGrad(NData *deltaData);
   public$ float    *getBias();
   public$ float    *getUpdates();
   public$ float    *getEma();
   public$ void      createEma();//只有train才需要创建ema数据
   public$ void      createUpdates();
   public$ void      clear();
   public$ BiasData *clone();
   public$ void      setBiasValue(float value);
   public$ int       read(FILE *fp);
   /**
    *把updates加到bias上
    *原型 axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    */
   public$ void       addUpdatesToBias(float alpha);//把梯度加到bias上
   public$ void       scaleUpdates(float scale);//缩放梯度
   public$ int        getSize();
};




#endif /* __N_MEM_H__ */

