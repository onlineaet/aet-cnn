#ifndef __COM_AI_DNN_DELTA_DATA_H__
#define __COM_AI_DNN_DELTA_DATA_H__

#include <aet.h>

#include  "NData.h"
#include  "NormData.h"

package$ com.ai.dnn;

/**
 * 计算用的数据，把图像数据转成了float,
 */
public$ class$ DeltaData extends$ NData{

    protected$ float *mean; //误差平均值
    protected$ float *variance; //误差均方差
    public$ DeltaData(int w,int h,int channels,int batch);
    public$ DeltaData(int size,int batch);
    //原型 mean_delta_cpu blas.h batchnorm_layer.c
    public$ void      calcMean(float *variance);
    public$ void      calcVariance(NormData *normData);
    //原型 normalize_delta_cpu blas.h batchnorm_layer.c
    public$ void       normalize(NormData *normData);
    public$ float     *getMean();
    public$ float     *getVariance();
    public$ void      checkMeanVariance();
    public$ DeltaData *clone();
};


#endif /* __N_MEM_H__ */

