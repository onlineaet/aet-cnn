

#ifndef __COM_AI_DNN_BOX_LABEL_H__
#define __COM_AI_DNN_BOX_LABEL_H__

#include <aet.h>



package$ com.ai.dnn;


public$ class$ BoxLabel{
    public$ static void randomizeSwap(BoxLabel **boxes, int count);
    public$ static void correctBoxes(BoxLabel **boxes, int count, float dx, float dy, float sx, float sy, int flip);
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
    public$ void correct(float dx, float dy, float sx, float sy, int flip);
    public$ float *encode();


};




#endif /* __N_MEM_H__ */

