#ifndef __COM_AI_DATA_FACTORY_H__
#define __COM_AI_DATA_FACTORY_H__

#include <aet.h>

#include "NormData.h"
#include "ScaleData.h"
#include "DeltaData.h"
#include "WeightData.h"
#include "Im2Col.h"
#include "Gemm.h"
#include "IOData.h"
#include "BiasData.h"


package$ com.ai.dnn;
/**
 * 创建网络的工厂
 */
public$ class$ DataFactory{
    aboolean useMtcs;
    private$ DataFactory();
    public$ static DataFactory *getInstance();
    public$ NData     *createData(int size,int batch);
    public$ NormData  *createNormData(int w,int h,int channels,int batch);
    public$ ScaleData *createScaleData(int size);

    public$ DeltaData  *createDeltaData(int w,int h,int channels,int batch);
    public$ DeltaData  *createDeltaData(int size,int batch);
    public$ InputData  *createInputData(int size,int batch);
    public$ InputData  *createInputData(int w,int h,int channels,int batch);
    public$ OutputData *createOutputData(int w,int h,int channels,int batch);
    public$ OutputData *createOutputData(int size,int batch);
    public$ BiasData   *createBiasData(int size);
    public$ int        *createIndexes(int batch,int size);

    public$ WeightData *createConvKernel(int ksize,int channels,int filters,int pad,int stride);
    public$ Im2Col     *createIm2Col();
    public$ Gemm       *createGemm();

};




#endif /* __N_MEM_H__ */

