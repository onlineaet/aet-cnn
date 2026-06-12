#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/util/ARandom.h>

#include  "BoxLabel.h"
#include  "DnnUtils.h"


impl$ BoxLabel{

    static void randomizeSwap(BoxLabel **boxes, int count){
        int i;
        for(i = 0; i < count; ++i){
            BoxLabel *swap = boxes[i];
            int index = ARandom.getInstance()->nextInt()%count;
            boxes[i] = boxes[index];
            boxes[index] = swap;
        }
    }

    static void correctBoxes(BoxLabel **boxes, int count, float dx, float dy, float sx, float sy, int flip){
        int i;
        for(i = 0; i < count; ++i){
            boxes[i]->correct(dx,dy,sx,sy,flip);
        }
    }

    void correct(float dx, float dy, float sx, float sy, int flip){
            int i;
            if(self->x == 0 && self->y == 0) {
                self->x = 999999;
                self->y = 999999;
                self->w = 999999;
                self->h = 999999;
                return;
            }
            self->left   = self->left  * sx - dx;
            self->right  = self->right * sx - dx;
            self->top    = self->top   * sy - dy;
            self->bottom = self->bottom* sy - dy;

            if(flip){
                float swap = self->left;
                self->left = 1. - self->right;
                self->right = 1. - swap;
            }

            self->left =  DnnUtils.constrain(0, 1, self->left);
            self->right = DnnUtils.constrain(0, 1, self->right);
            self->top =   DnnUtils.constrain(0, 1, self->top);
            self->bottom =DnnUtils.constrain(0, 1, self->bottom);

            self->x = (self->left+self->right)/2;
            self->y = (self->top+self->bottom)/2;
            self->w = (self->right - self->left);
            self->h = (self->bottom - self->top);
            self->w = DnnUtils.constrain(0, 1, self->w);
            self->h = DnnUtils.constrain(0, 1, self->h);

    }

    float *encode(){
        float *ret=malloc(5*sizeof(float));
        ret[0]=x;
        ret[1]=y;
        ret[2]=w;
        ret[3]=h;
        ret[4]=id;
        return ret;
    }

    ~BoxLabel(){
    }

};

