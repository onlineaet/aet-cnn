

#ifndef __COM_AI_DNN_N_IMAGE_H__
#define __COM_AI_DNN_N_IMAGE_H__

#include <aet.h>
#include "Box.h"

package$ com.ai.dnn;

typedef struct {
   int w;
   int h;
   float scale;
   float rad;
   float dx;
   float dy;
   float aspect;
} augment_args;

/**
 * 图像
 */
public$ class$ NImage{
   public$ enum$ ImageType{
      PNG, BMP, TGA, JPG
   };

   private$ static float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

   int color;
   int width;
   int height;
   float *imgData;

   public$ static NImage *createImage(char *fileName,int w,int h);
   public$ static NImage *createImage(char *fileName,int channels);
   //原型 get_color image.h image.c
   public$ static float getColor(int c, int x, int max);
   /**
   * 生成带颜色的图像，如果大小与实际不符，缩放
   */
   public$ static NImage *createImage(char *fileName,int w,int h,int c);
   //原型 make_empty_image image.h image.c
   public$ static NImage *makeEmptyImage(int w, int h, int c);
   //原型 title_images image.c
   public$ static NImage *createTitleImage(NImage *a, NImage *b, int dx);

   public$  NImage(int w, int h, int c);
   public$  NImage(float *data,int w, int h, int c);
   public$ void     fill(float s);
   public$ int      getColor();
   public$ void     place(int w,int h,int dx,int dy,NImage *canvas);//平移
   //原型 random_distort_image image.h image.c
   public$ void     randomDistort(float hue,float saturation,float exposure);//随机扭曲
   //原型 flip_image image.h image.c
   public$ void     flip();//翻转
   public$ int      getWidth();
   public$ int      getHeight();
   public$ void     distort(float hue, float sat, float val);
   public$ void     setPixel(int x, int y, int c, float val);
   //直接设rgb,避免调用三次setPixel
   public$ void     setPixel(int x, int y, float r,float g,float b);

   //原型 save_image imae.h image.c
   public$ void     save(const char *fileName);
   //原型 show_image imae.h image.c
   public$ void     show(const char *name);
   public$ float   *getImageData(int channel);
   //原型 resize_image image.h image.c
   public$ NImage  *resize(int w, int h);
   public$ NImage  *centerCrop(int w, int h);
   public$ NImage  *crop(int dx, int dy, int w, int h);
   //原型 random_augment_image image.h image.c
   public$ NImage  *randomAugment(float angle, float aspect, int low, int high, int size);
   //原型 rotate_crop_image image.h image.c
   public$ NImage  *rotateCrop(float rad, float s, int w, int h, float dx, float dy, float aspect);
   //原型 copy_image image.h image.c
   public$ NImage  *copy();
   //原型 get_pixel_extend image.h image.c
   public$  float   getPixelExtend(int x, int y, int c);
   private$ float   getPixel(int x, int y, int c);
   private$ void    addPixel(int x, int y, int c, float val);
   //public$  void    checkNan(char *info);
   public$  NImage *letterboxImage(int w, int h);
   //原型 threshold_image image.h image.c
   public$ NImage * thresholdImage(float thresh);
   public$ void     embed(NImage *dest, int dx, int dy);
   public$ float   *getData();
   public$ int      getSize();
   //原型 draw_box_width image.h image.c
   public$ void     drawBoxWidth(int x1, int y1, int x2, int y2, int w, float r, float g, float b);
   //原型 draw_weighted_label image.h image.c
   public$ void     drawWeightedLabel(int r, int c, NImage *label, const float *rgb, const float alpha);
   //原型 draw_detections_v3 image.h image.c
   public$ void     drawDetectionsV3(Detection **dets, int num,
               float thresh, char **names, NImage **alphabet, int classes, int ext_output);
   //原型 composite_image image.c
   public$ void      composite(NImage *dest, int dx, int dy);
   //原型 border_image image.c
   public$ NImage   *borderImage(int border);
   public$ aboolean  resize(int w, int h,int c,float *dest);
   public$ void      reload(char *fileName);
   public$ NImage   *resizeMin(int w);

};




#endif /* __N_MEM_H__ */

