#ifndef __COM_AI_DNN_TRUTH_DATA_H__
#define __COM_AI_DNN_TRUTH_DATA_H__

#include <aet.h>

package$ com.ai.dnn;


/**
 * 图片的标注框
 * batch：图片数
 */
public$ class$ TruthData{
    int batch; //图片个数
    struct{
        int count;
        float *values[100];
    }truths[512]; //一张图对应一个truths元素 ,每个元素可以有100个标注框、类别

    aboolean useMtcs;
    float *dataArray;
    int *boxes;
    int classes;
    TruthData(int batch,int classes,aboolean useMtcs);
    public$ float  *getData(int index,int sub);
    public$ void    setData(float *data);
    public$ float  *getData(int index);
    public$ void    setBox(int count,int index);
    public$ int     getBoxCount(int index);
    public$ float  *getDataArray();


};




#endif /* __N_MEM_H__ */

