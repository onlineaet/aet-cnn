

#ifndef __COM_AI_DNN_N_DATA_H__
#define __COM_AI_DNN_N_DATA_H__

#include <math.h>
#include <aet.h>

package$ com.ai.dnn;

/**
 * 层的输出数据、输入数据以及误差数据的抽象。
 */
public$ class$ NData{

    public$ enum$ DataType{
        CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA,
        WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA,
        LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
    };

    public$ static aboolean isInValidNumber(float value){
        return (isnan(value) || isinf(value));
    }
    int w, h,channels;
    int size;//单个数据的大小
    int batch;
    float *dataArray;
    char  toString[255];
    protected$ ADestroyNotify destroyFunc;//清除内存data的方法
    protected$ aboolean readOnly;
    private$ aboolean isAlign;//数据是否对齐
    protected$ int needThreads;//omp需要的线程数

    public$ NData(int w,int h,int channels,int batch);
    public$ NData(int w,int h,int channels,int batch,aboolean alloc,ADestroyNotify freeFunc);
    public$ NData(int w,int h,int channels,int batch,aboolean align);//分配内存，并且是否对齐
    public$ NData(int size,int batch);//没有w,h,channels,只有size和batch
    public$ NData(int size,float *data);

    public$  int     getSize();//单个数据的大小
    public$  int     getWidth();
    public$  int     getHeight();
    public$  int     getChannels();
    public$  int     getBatch();
    public$ float   *getDataArray();
    public$ float   *getData(int index);
   // public$ float   *getData(int index,int channels);
    public$ void     resize(int w,int h);
    public$ void     resize(int w,int h,int c);
    public$ void     resize(int size);
    public$ void     setData(float *data,int index);
    public$ void     setData(NData *src);
    public$ aboolean compare(NData *obj);
    public$ aboolean copy(NData *dest);
    public$ void     setZero();//dataArray设为0
    public$ void     copy(NData *dest,int index,int destOffset);
    //原型 scale_bias blas.h convolutional_layer.c
    public$ void     scale(float *scales);
    public$ void     scaleBias(float *scales);
    /**
     * 求和 原型 sum_array utils.h utils.c
     */
    public$ float   sum();
    public$ char   *toString();
    public$ void    checkNan(char *info,int layer);
    public$ void    checkNan(char *info,int layer,int index);
    public$ void    scaleAndAddBias(float *scales,float *biases);
    public$ NData  *clone();


};




#endif /* __N_MEM_H__ */

