#ifndef __COM_AI_CNN_IMAGE_DATA_H__
#define __COM_AI_CNN_IMAGE_DATA_H__

#include <aet.h>
#include <aet/util/AMutex.h>

#include "NImage.h"
#include "ComputingUnit.h"
#include "TruthData.h"

package$ com.ai.dnn;

// data.h
typedef enum {
    CLASSIFICATION_DATA,
    DETECTION_DATA,
    CAPTCHA_DATA,
    REGION_DATA,
    IMAGE_DATA,
    COMPARE_DATA,
    WRITING_DATA,
    SWAG_DATA,
    TAG_DATA,
    OLD_CLASSIFICATION_DATA,
    STUDY_DATA,
    DET_DATA,
    SUPER_DATA,
    LETTERBOX_DATA,
    REGRESSION_DATA,
    SEGMENTATION_DATA,
    INSTANCE_DATA,
    ISEG_DATA
} data_type;

public$ class$ ImageData implements$ ComputingUnit{

   int threads;
   char **paths;
   char *path;
   //n是每个批次的图片数 int n = net->batch * net->subdivisions * ngpus;
   int batchImageCount;
   int totalImageCount;//总的训练图片数
   char **labels;
   private$ int h;
   private$ int w;
   private$ int c; // color depth
   int out_w;
   int out_h;
   int nh;
   int nw;
   int num_boxes;
   int truth_size;
   int min, max, size;
   int classes;
   int background;
   int scale;
   int center;
   int coords;
   int mini_batch;
   int track;
   int augment_speed;
   int letter_box;
   int mosaic_bound;
   int show_imgs;
   int dontuse_opencv;
   int contrastive;
   int contrastive_jit_flip;
   int contrastive_color;
   float jitter;
   float resize;
   int use_flip;
   int gaussian_noise;
   int blur;
   int mixup;
   float label_smooth_eps;
   float angle;
   float aspect;
   float saturation;
   float exposure;
   float hue;
   data_type type;
   tree *hierarchy;

   float *batchImageData ;
   float *truthsData;
   int  imoffset;
   NImage *srcImgs[512];

   ImageData(int totalImageCount,int batchImgCount,int classes,int w,int h,int c);
   int loadData();
   void wait(int id);
   //返回一次生成的图片数
   int getImagCount();
   //原型 get_next_batch data.h data.c
   void getNextBatch(int n, int offset, InputData *X,TruthData *y);
};




#endif /* __N_MEM_H__ */

