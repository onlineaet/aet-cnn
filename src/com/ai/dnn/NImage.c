#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/util/ARandom.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/AThread.h>
#include <aet/lang/AString.h>
#include "DnnUtils.h"
#include "NImage.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"


static inline   float three_way_max(float a, float b, float c)
{
       return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

static inline float three_way_min(float a, float b, float c)
{
        return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

// compare to sort detection** by bbox.x
static int compare_by_lefts(const void *a_ptr, const void *b_ptr)
{
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det->bbox.x - a->det->bbox.w/2) - (b->det->bbox.x - b->det->bbox.w/2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
static int compare_by_probs(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det->prob[a->best_class] - b->det->prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

static inline void set_pixel(int x, int y, int c, float val,float *data,int width,int height,int color)
{
   if (x < 0 || y < 0 || c < 0 || x >= width || y >= height || c >= color)
      return;
   a_assert(x < width && y < height && c <color);
   data[c*height*width + y*width + x] = val;
}

private$ inline float get_pixel(int x, int y, int c,float *imgData,int width,int height,int color)
{
   a_assert(x < width && y < height && c <color);
   return imgData[c*height*width + y*width + x];
}


impl$ NImage{

   NImage(int w, int h, int c){
      self->width=w;
      self->height=h;
      self->color=c;
      imgData=a_malloc0(h*w*c*sizeof(float));
   }

   NImage(float *data,int w, int h, int c){
      self(w,h,c);
      memcpy(imgData,data,w*h*c*sizeof(float));
   }

   //原型 embed_image image.h image.c
   void embed(NImage *dest, int dx, int dy){
      int x,y,k;
      for(k = 0; k < self->color; ++k){
         for(y = 0; y < self->height; ++y){
            for(x = 0; x < self->width; ++x){
               float val = getPixel/*!get_pixel*/(x,y,k);
               dest->setPixel/*!set_pixel*/(dx+x, dy+y, k, val);
            }
         }
      }
   }

   NImage *letterboxImage(int w, int h){
      int new_w = self->width;
      int new_h = self->height;
      if (((float)w / self->width) < ((float)h / self->height)) {
         new_w = w;
         new_h = (self->height * w) /self->width;
      }else {
         new_h = h;
         new_w = (self->width * h) / self->height;
      }
      NImage *resized = resize/*!resize_image*/(new_w, new_h);
      NImage *boxed =new$ NImage(w, h, self->color);
      boxed->fill/*!fill_image*/(.5);
      //int i;
      //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
      resized->embed/*!embed_image*/(boxed, (w - new_w) / 2, (h - new_h) / 2);
      resized->unref();
      return boxed;
   }

   public$ float *getData(){
      return imgData;
   }

   public$ int getSize(){
      return width*height*color;
   }

   void fill(float s){
      int i;
      int pixs=height*width*color;
      for(i = 0; i < pixs; ++i)
         imgData[i] = s;
   }

   private$ inline float getPixel(int x, int y, int c){
      a_assert(x < width && y < height && c <color);
      return imgData[c*height*width + y*width + x];
   }

   private$  inline void getPixel(int x, int y, float *r,float *g,float *b){
      a_assert(x < width && y < height);
      int wh=height*width;
      int yw=y*width+x;
      *r=imgData[yw];
      *g=imgData[wh+yw];
      *b=imgData[2*wh+yw];
   }

   //原型 get_pixel_extend image.h image.c
   float getPixelExtend(int x, int y, int c){
      if(x < 0 || x >= width || y < 0 || y >= height)
         return 0;
      if(c < 0 || c >= color)
         return 0;
      return getPixel(x, y, c);
   }

   float bilinearInterpolate(float x, float y, int c){
      int ix = (int) floorf(x);
      int iy = (int) floorf(y);

      float dx = x - ix;
      float dy = y - iy;
      float val = (1-dy) * (1-dx) * getPixelExtend(ix, iy, c) +
                     dy     * (1-dx) * getPixelExtend(ix, iy+1, c) +
                     (1-dy) *   dx   * getPixelExtend(ix+1, iy, c) +
                     dy     *   dx   * getPixelExtend(ix+1, iy+1, c);
      return val;
   }

   void setPixel(int x, int y, int c, float val){
      if (x < 0 || y < 0 || c < 0 || x >= width || y >= height || c >= color)
         return;
      a_assert(x < width && y < height && c <color);
      self->imgData[c*height*width + y*width + x] = val;
   }

   void setPixel(int x, int y, float r,float g,float b){
      if (x < 0 || y < 0 || x >= width || y >= height)
         return;
      a_assert(x < width && y < height);
      int wh=height*width;
      int yz=y*width + x;
      self->imgData[yz] = r;
      self->imgData[wh+yz] = g;
      self->imgData[2*wh+yz] = b;

   }

   private$ inline void addPixel(int x, int y, int c, float val){
      assert(x < width && y < height && c < color);
      self->imgData[c*height*width + y*width + x] += val;
   }

   /**
   * 平移图片
   */
   void place(int w,int h,int dx,int dy,NImage *canvas){
      int x, y, c;
      for(c = 0; c < color; ++c){
         for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
               float rx = ((float)x / w) * width;
               float ry = ((float)y / h) * height;
               float val = bilinearInterpolate(rx, ry, c);
               canvas->setPixel(x + dx, y + dy, c, val);
            }
         }
      }
   }

   // http://www.cs.rit.edu/~ncs/color/t_convert.html
   void rgbToHsv(){
      a_assert(color == 3);
      int i, j;
      float r, g, b;
      float h, s, v;
      for(j = 0; j < height; ++j){
         for(i = 0; i < width; ++i){
            /*
            r = getPixel(i , j, 0);//red
            g = getPixel(i , j, 1);//green
            b = getPixel(i , j, 2);//black
            */
            getPixel(i,j,&r,&g,&b);
            float max = three_way_max(r,g,b);
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0){
               s = 0;
               h = 0;
            }else{
               s = delta/max;
               if(r == max){
                  h = (g - b) / delta;
               } else if (g == max) {
                  h = 2 + (b - r) / delta;
               } else {
                  h = 4 + (r - g) / delta;
               }
               if (h < 0)
                  h += 6;
               h = h/6.;
            }
            /*
            setPixel(i, j, 0, h);
            setPixel(i, j, 1, s);
            setPixel(i, j, 2, v);
            */
            setPixel(i,j,h,s,v);

         }
      }
   }

   void hsvToRgb(){
      a_assert(color == 3);
      int i, j;
      float r, g, b;
      float h, s, v;
      float f, p, q, t;
      for(j = 0; j < height; ++j){
         for(i = 0; i < width; ++i){
            /*
            h = 6 * getPixel(i , j, 0);
            s = getPixel(i , j, 1);
            v = getPixel(i , j, 2);
            */
            getPixel(i,j,&h,&s,&v);
            h=6*h;

            if (s == 0) {
               r = g = b = v;
            } else {
               int index = floor(h);
               f = h - index;
               p = v*(1-s);
               q = v*(1-s*f);
               t = v*(1-s*(1-f));
               if(index == 0){
                  r = v; g = t; b = p;
               } else if(index == 1){
                  r = q; g = v; b = p;
               } else if(index == 2){
                  r = p; g = v; b = t;
               } else if(index == 3){
                  r = p; g = q; b = v;
               } else if(index == 4){
                  r = t; g = p; b = v;
               } else {
                  r = v; g = p; b = q;
               }
            }
            /*
            setPixel(i, j, 0, r);
            setPixel(i, j, 1, g);
            setPixel(i, j, 2, b);
            */
            setPixel(i,j,r,g,b);
         }
      }
   }


   void scaleChannel(int c, float v){
      int i, j;
      for(j = 0; j < height; ++j){
         for(i = 0; i < width; ++i){
            //float pix = getPixel(i, j, c);
            float pix = get_pixel(i, j, c,imgData,width,height,color);

            pix = pix*v;
            //setPixel(i, j, c, pix);
            set_pixel(i, j, c, pix,imgData,width,height,color);

         }
      }
   }

   void constrainImage(){
      int i;
      int pixs=width*height*color;
      for(i = 0; i < pixs; ++i){
         if(imgData[i] < 0)
            imgData[i] = 0;
         if(imgData[i] > 1)
            imgData[i] = 1;
      }
   }

   /**
   * 扭曲
   */
   void distort(float hue, float sat, float val){
      rgbToHsv();
      scaleChannel(1, sat);
      scaleChannel(2, val);
      int i;
      int pixs=width*height;
      for(i = 0; i < pixs; ++i){
         imgData[i] =imgData[i]+ hue;
         if (imgData[i]> 1)
            imgData[i] -= 1;
         if (imgData[i] < 0)
            imgData[i] += 1;
      }
      hsvToRgb();
      constrainImage();
   }

   /**
   * 随机扭曲
   */
   //原型 random_distort_image image.h image.c
   void randomDistort(float hue, float saturation, float exposure){
      float dhue =DnnUtils.randUniformStrong(-hue,hue);
      float dsat =DnnUtils.randScale(saturation);
      float dexp =DnnUtils.randScale(exposure);
      distort(dhue, dsat, dexp);
   }

   /**
   * 原型 flip_image image.h image.c
   * 翻转
   */
   void flip(){
      int i,j,k;
      int halfW=width/2;
      for(k = 0; k <color; ++k){
         for(i = 0; i <height; ++i){
            for(j = 0; j < halfW; ++j){
               int index = j + width*(i + height*(k));
               int flip = (width - j - 1) +width*(i + height*(k));
               float swap = imgData[flip];
               imgData[flip] = imgData[index];
               imgData[index] = swap;
            }
         }
      }
   }

   int getColor(){
      return color;
   }

   int getWidth(){
      return width;
   }

   int getHeight(){
      return height;
   }

   float *getImageData(int channel){
      if(channel>=self->color){
         a_error("通道数不正确。channel:%d color:%d",channel,self->color);
      }
      return imgData+channel* self->width* self->height;
   }

   //原型 save_image_options image.c
   void saveImageOptions(const char *name, ImageType  f, int quality){
       char buff[256];
       //sprintf(buff, "%s (%d)", name, windows);
       if (f == ImageType.PNG)
          sprintf(buff, "%s.png", name);
       else if (f == ImageType.BMP)
          sprintf(buff, "%s.bmp", name);
       else if (f == ImageType.TGA)
          sprintf(buff, "%s.tga", name);
       else if (f == ImageType.JPG)
          sprintf(buff, "%s.jpg", name);
       else
          sprintf(buff, "%s.png", name);
       unsigned char* data = (unsigned char*)xcalloc(width * height * color, sizeof(unsigned char));
       int i, k;
       for (k = 0; k < color; ++k) {
           for (i = 0; i < width*height; ++i) {
               data[i*color + k] = (unsigned char)(255 * imgData[i + k*width*height]);
           }
       }
       int success = 0;
       if (f == ImageType.PNG)
          success = stbi_write_png(buff, width, height, color, data, width*color);
       else if (f == ImageType.BMP)
          success = stbi_write_bmp(buff, width, height, color, data);
       else if (f == ImageType.TGA)
          success = stbi_write_tga(buff, width, height, color, data);
       else if (f == ImageType.JPG)
          success = stbi_write_jpg(buff, width, height, color, data, quality);
       free(data);
       if (!success)
          fprintf(stderr, "Failed to write image %s\n", buff);
   }

   //原型 save_image imae.h image.c
   void save(const char *name){
      saveImageOptions(name, ImageType.JPG, 80);
   }

   //原型 show_image imae.h image.c
   void show(const char *name){
       fprintf(stderr, "Not compiled with OpenCV, saving to %s.jpg instead\n", name);
       save(name);
   }

   /**
   * 剪裁并生成新图像。
   */
   public$ NImage *crop(int dx, int dy, int w, int h){
      NImage *cropped=new$ NImage(w,h,self->color);
      int i, j, k;
      for(k = 0; k < color; ++k){
         for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
               int r = j + dy;
               int c = i + dx;
               float val = 0;
               r = DnnUtils.constrain(r, 0, height-1);
               c = DnnUtils.constrain(c, 0, width-1);
               if (r >= 0 && r < height && c >= 0 && c < width) {
                  //val = getPixel(c, r, k);
                  val = get_pixel(c,r,k,imgData,width,height,color);
               }
               //cropped->setPixel(i, j, k, val);
               set_pixel(i,j,k,val,cropped->imgData,cropped->width,cropped->height,cropped->color);
            }
         }
      }
      return cropped;
   }

   /**
   * 从中心剪裁
   */
   public$ NImage *centerCrop(int w, int h){
      int m = (self->width< self->height) ? self->width: self->height;
      NImage *c = crop((self->width - m) / 2, (self->height - m)/2, m, m);
      NImage *r = c->resize(w, h);
      c->unref();
      return r;
   }

   //原型 resize_image image.h image.c
   private$ NImage *resizeImage(int w, int h){
      NImage *resized=new$ NImage(w,h,color);
      NImage *part=new$ NImage(w,height,color);
      int r, c, k;
      float w_scale = (float)(width - 1) / (w - 1);
      float h_scale = (float)(height - 1) / (h - 1);
      for(k = 0; k < color; ++k){
         for(r = 0; r < height; ++r){
            for(c = 0; c < w; ++c){
               float val = 0;
               if(c == w-1 || width == 1){
                  //val = getPixel(width-1, r, k);
                  val = get_pixel(width-1, r, k,imgData,width,height,color);
               }else{
                  float sx = c*w_scale;
                  int ix = (int) sx;
                  float dx = sx - ix;
                  //val = (1 - dx) * getPixel(ix, r, k) + dx * getPixel(ix+1, r, k);
                  val = (1 - dx) * get_pixel(ix, r, k,imgData,width,height,color)
                        + dx * get_pixel(ix+1, r, k,imgData,width,height,color);

               }
               //part->setPixel(c, r, k, val);
               set_pixel(c,r,k,val,part->imgData,part->width,part->height,part->color);
            }
         }
      }
      for(k = 0; k < self->color; ++k){
         for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
               //float val = (1-dy) * part->getPixel(c, iy, k);
               float val = (1-dy) *get_pixel(c, iy, k,part->imgData,part->width,part->height,part->color);

               //resized->setPixel(c, r, k, val);
               set_pixel(c, r, k, val,resized->imgData,resized->width,resized->height,resized->color);

            }
            if(r == h-1 || self->height== 1)
               continue;
            for(c = 0; c < w; ++c){
              // float val = dy * part->getPixel(c, iy+1, k);
               float val = dy *get_pixel(c, iy+1, k,part->imgData,part->width,part->height,part->color);
               resized->addPixel(c, r, k, val);
            }
         }
      }
      part->unref();
      return resized;
   }

   /**
   * 把原图复制到新的大小图像上。
   */
   //原型 resize_image image.h image.c
   public$ NImage *resize(int w, int h){
      if (width == w && height == h)
         return copy();
      return resizeImage(w,h);
   }

   //resize 数据到 dest
   public$ aboolean resize(int w, int h,int c,float *dest){
      if (width == w && height == h && color==c){
         memcpy(dest,imgData,w*h*c*sizeof(float));
         return TRUE;
      }
      printf("resize 大小不匹配- %d %d %d %d %d %d\n",w,h,c,width,height,color);
      NImage *im=resizeImage(w, h);
      memcpy(dest,im->imgData,w*h*c*sizeof(float));
      im->unref();
      return TRUE;
   }

   //原型 rotate_crop_image image.h image.c
   public$ NImage *rotateCrop(float rad, float s, int w, int h, float dx, float dy, float aspect){
      int x, y, c;
      float cx = self->width/2.0;
      float cy = self->height/2.0;
      NImage *rot = new$ NImage(w, h, self->color);
      float cr=cos(rad);
      float sr=sin(rad);
      float dx1=dx/s*aspect;
      float dy1= dy/s;
      for(c = 0; c < self->color; ++c){
         for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
               float rx = cr*((x - w/2.)/s*aspect + dx1) - sr*((y - h/2.)/s + dy1) + cx;
               float ry = sr*((x - w/2.)/s*aspect + dx1) + cr*((y - h/2.)/s + dy1) + cy;
               float val = bilinearInterpolate(rx, ry, c);
               //rot->setPixel(x, y, c, val);
               set_pixel(x,y,c,val,rot->imgData,rot->width,rot->height,rot->color);
            }
         }
      }
      return rot;
   }


//   void checkNan(char *info){
//      int b,n;
//      float *item=imgData;
//      int size=width*height*color;
//      for(n = 0; n < size; ++n){
//         if(isnan(item[n])){
//            printf("nimage checkNan %s n:%d isnan %f\n",info,n,item[n]);
//            exit(0);
//         }
//      }
//   }

   public$ static NImage *createImage(char *fileName,int channels){
      int w, h, c;
      unsigned char *data = stbi_load(fileName, &w, &h, &c, channels);
      if (!data) {
         printf("生成图片出错:%s\n",fileName);
         return new$ NImage (10, 10, 3);
      }
      if(channels)
         c = channels;
      int i,j,k;
      NImage *im = new$ NImage(w, h, c);
      float *imageData=im->getData();
      for(k = 0; k < c; ++k){
         for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
               int dst_index = i + w*j + w*h*k;
               int src_index = k + c*i + c*w*j;
               imageData[dst_index] = (float)data[src_index]/255.;
            }
         }
      }
      free(data);
      return im;
   }

   /**
    * 生成RGB图像，如果大小与实际不符，缩放
    */
   public$ static NImage *createImage(char *fileName,int w,int h){
      NImage *out =createImage(fileName,3);
      if((h && w) && (h != out->getHeight() || w != out->getWidth())){
         NImage *resized = out->resize(w, h);
         out->unref();
         out = resized;
      }
      return out;
   }

   /**
    * 生成带颜色的图像，如果大小与实际不符，缩放
    */
   public$ static NImage *createImage(char *fileName,int w,int h,int c){
      NImage *out =createImage(fileName,c);
      if((h && w) && (h != out->getHeight() || w != out->getWidth())){
         NImage *resized = out->resize(w, h);
         out->unref();
         out = resized;
      }
      return out;
   }

   public$ void reload(char *fileName){
      int w, h, c;
      unsigned char *data = stbi_load(fileName, &w, &h, &c, color);
      if (!data) {
         a_error("生成图片出错:%s\n",fileName);
      }
      if(w!=width || h!=height || c!=color){
         a_error("生成图片出错 大小不符:%s\n",fileName);
      }

      int i,j,k;
      for(k = 0; k < c; ++k){
         for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
               int dst_index = i + w*j + w*h*k;
               int src_index = k + c*i + c*w*j;
               imgData[dst_index] = (float)data[src_index]/255.0;
            }
         }
      }
      free(data);
   }

   //原型 get_color image.h image.c
   static float getColor(int c, int x, int max){
      float ratio = ((float)x/max)*5;
      int i = floor(ratio);
      int j = ceil(ratio);
      ratio -= i;
      float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
      //printf("%f\n", r);
      return r;
   }

   //原型 draw_box_bw image.c
   void drawBoxBw(int x1, int y1, int x2, int y2, float brightness){
      //normalize_image(a);
      int i;
      if (x1 < 0)
         x1 = 0;
      if (x1 >= width)
         x1 = width - 1;
      if (x2 < 0)
         x2 = 0;
      if (x2 >= width)
         x2 =width - 1;

      if (y1 < 0)
         y1 = 0;
      if (y1 >= height)
         y1 = height - 1;
      if (y2 < 0)
         y2 = 0;
      if (y2 >= height)
         y2 = height - 1;

      for (i = x1; i <= x2; ++i) {
         imgData[i + y1*width + 0 * width*height] = brightness;
         imgData[i + y2*width + 0 * width*height] = brightness;
      }
      for (i = y1; i <= y2; ++i) {
         imgData[x1 + i*width + 0 * width*height] = brightness;
         imgData[x2 + i*width + 0 * width*height] = brightness;
      }
   }

   //原型 draw_box_width_bw image.c
   void drawBoxWidthBw(int x1, int y1, int x2, int y2, int w, float brightness){
      int i;
      for (i = 0; i < w; ++i) {
         float alternate_color = (w % 2) ? (brightness) : (1.0 - brightness);
         drawBoxBw/*!draw_box_bw*/(x1 + i, y1 + i, x2 - i, y2 - i, alternate_color);
      }
   }

   //原型 draw_box image.c
   void drawBox(int x1, int y1, int x2, int y2, float r, float g, float b){
      //normalize_image(a);
      int i;
      if(x1 < 0)
         x1 = 0;
      if(x1 >= width)
         x1 = width-1;
      if(x2 < 0)
         x2 = 0;
      if(x2 >= width)
         x2 = width-1;

      if(y1 < 0)
         y1 = 0;
      if(y1 >= height)
         y1 = height-1;
      if(y2 < 0)
         y2 = 0;
      if(y2 >= height)
         y2 = height-1;

      for(i = x1; i <= x2; ++i){
         imgData[i + y1*width + 0*width*height] = r;
         imgData[i + y2*width + 0*width*height] = r;

         imgData[i + y1*width + 1*width*height] = g;
         imgData[i + y2*width + 1*width*height] = g;

         imgData[i + y1*width + 2*width*height] = b;
         imgData[i + y2*width + 2*width*height] = b;
      }
      for(i = y1; i <= y2; ++i){
         imgData[x1 + i*width + 0*width*height] = r;
         imgData[x2 + i*width + 0*width*height] = r;

         imgData[x1 + i*width + 1*width*height] = g;
         imgData[x2 + i*width + 1*width*height] = g;

         imgData[x1 + i*width + 2*width*height] = b;
         imgData[x2 + i*width + 2*width*height] = b;
      }
   }
   //原型 draw_box_width image.h image.c
   void drawBoxWidth(int x1, int y1, int x2, int y2, int w, float r, float g, float b){
      int i;
      for(i = 0; i < w; ++i){
         drawBox/*!draw_box*/(x1+i, y1+i, x2-i, y2-i, r, g, b);
      }
   }

   //原型 make_empty_image image.h image.c
   static NImage *makeEmptyImage(int w, int h, int c){
      NImage *img=new$ NImage();
      img->imgData = NULL;
      img->height = h;
      img->width = w;
      img->color = c;
      return img;
   }

   //原型 copy_image image.h image.c
   NImage * copy(){
      NImage *copy=new$ NImage(width,height,color);
      memcpy(copy->imgData,imgData,width*height*color*sizeof(float));
      return copy;
   }

   //原型 composite_image image.c
   void composite(NImage *dest, int dx, int dy){
      int x,y,k;
      for(k = 0; k < color; ++k){
         for(y = 0; y < height; ++y){
            for(x = 0; x < width; ++x){
               float val = getPixel/*!get_pixel*/(x, y, k);
               float val2 = dest->getPixelExtend/*!get_pixel_extend*/(dx+x, dy+y, k);
               dest->setPixel/*!set_pixel*/(dx+x, dy+y, k, val * val2);
            }
         }
      }
   }

   //原型 title_images image.c
   static NImage *createTitleImage(NImage *a, NImage *b, int dx){
      if(a->width == 0)
         return b->copy();
      NImage *c =new$ NImage(a->width + b->width+ dx, (a->height > b->height)
      ? a->height : b->height, (a->color > b->color) ? a->color : b->color);
      c->fill/*! fill_cpu*/(1.0);
      a->embed/*!embed_image*/(c, 0, 0);
      b->composite/*!composite_image*/(c, a->width + dx, 0);
      return c;
   }

   //原型 border_image image.c
   NImage *borderImage(int border){
      NImage *b = new$ NImage(width + 2*border, height + 2*border, color);
      int x,y,k;
      for(k = 0; k < b->color; ++k){
         for(y = 0; y < b->height; ++y){
            for(x = 0; x < b->width; ++x){
               float val = getPixelExtend/*!_pixel_extend*/(x - border, y - border, k);
               if(x - border < 0 || x - border >= width || y - border < 0 || y - border >= height)
                  val = 1;
               b->setPixel/*!set_pixel*/(x, y, k, val);
            }
         }
      }
      return b;
   }

   //原型 get_label_v3 image.c
   NImage *getLabelV3(NImage **characters, char *string, int size){
      size = size / 10;
      if (size > 7)
         size = 7;
      NImage *label =makeEmptyImage/*!make_empty_image*/(0, 0, 0);
      while (*string) {
         NImage **items=(NImage **)characters[size];
         NImage *l = items[(int)*string]/*!characters[size][(int)*string];*/;
         NImage *n = createTitleImage/*!tile_images*/(label, l, -size - 1 + (size + 1) / 2);
         label->unref();/*!free_image(label);*/
         label = n;
         ++string;
      }
      NImage *b = label->borderImage/*!border_image*/(label->height*.05);
      label->unref();
      return b;
   }

   //原型 draw_weighted_label image.h image.c
   void drawWeightedLabel(int r, int c, NImage *label, const float *rgb, const float alpha){
      int w = label->width;
      int h = label->height;
      if (r - h >= 0)
      r = r - h;

      int i, j, k;
      for (j = 0; j < h && j + r < height; ++j) {
         for (i = 0; i < w && i + c < width; ++i) {
            for (k = 0; k < label->color; ++k) {
               float val1 = label->getPixel/*!get_pixel*/(i, j, k);
               float val2 = getPixel(i + c, j + r, k);
               float val_dst = val1 * rgb[k] * alpha + val2 * (1 - alpha);
               setPixel(i + c, j + r, k, val_dst);
            }
         }
      }
   }

   //原型 threshold_image image.h image.c
   NImage * thresholdImage(float thresh){
      int i;
      NImage *t = new$ NImage(width,height,color);
      int len=width*height*color;
      for(i = 0; i < len; ++i)
      t->imgData[i] = imgData[i]>thresh ? 1 : 0;
      return t;
   }

   //原型 draw_detections_v3 image.h image.c
   void drawDetectionsV3(Detection **dets, int num,
         float thresh, char **names, NImage **alphabet, int classes, int ext_output){
      static int frame_id = 0;
      frame_id++;
      printf("draw_detections_v3 00 num:%d thresh:%f classes:%d ext_output:%d\n",num,thresh,classes,ext_output);
      int selected_detections_num;
      detection_with_class* selected_detections =Detection.getActualDetections/*! get_actual_detections*/(dets,
                  num, thresh, &selected_detections_num, names);
      printf("draw_detections_v3 11 num:%d thresh:%f classes:%d ext_output:%d %d\n",num,thresh,classes,ext_output,selected_detections_num);

      // text output
      qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
      int i;
      for (i = 0; i < selected_detections_num; ++i) {
         const int best_class = selected_detections[i].best_class;
         printf("%s: %.0f%%", names[best_class],    selected_detections[i].det->prob[best_class] * 100);
         if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                     round((selected_detections[i].det->bbox.x - selected_detections[i].det->bbox.w / 2)*self->width),
                     round((selected_detections[i].det->bbox.y - selected_detections[i].det->bbox.h / 2)*self->height),
                     round(selected_detections[i].det->bbox.w*self->width), round(selected_detections[i].det->bbox.h*self->height));

         else
            printf("\n");
         int j;
         for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det->prob[j] > thresh && j != best_class) {
               printf("%s: %.0f%%", names[j], selected_detections[i].det->prob[j] * 100);

               if (ext_output)
                  printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                           round((selected_detections[i].det->bbox.x - selected_detections[i].det->bbox.w / 2)*self->width),
                           round((selected_detections[i].det->bbox.y - selected_detections[i].det->bbox.h / 2)*self->height),
                           round(selected_detections[i].det->bbox.w*self->width),
                           round(selected_detections[i].det->bbox.h*self->height));
               else
                  printf("\n");
            }
         }
      }

      // image output
      qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
      for (i = 0; i < selected_detections_num; ++i) {
         int width = height * .002;
         if (width < 1)
            width = 1;

         /*
         if(0){
         width = pow(prob, 1./2.)*10+1;
         alphabet = 0;
         }
         */
         //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
         int offset = selected_detections[i].best_class * 123457 % classes;
         float red =getColor/*!get_color*/(2, offset, classes);
         float green = getColor/*!get_color*/(1, offset, classes);
         float blue = getColor/*!get_color*/(0, offset, classes);
         float rgb[3];

         //width = prob*20+2;
         rgb[0] = red;
         rgb[1] = green;
         rgb[2] = blue;
         Box b = selected_detections[i].det->bbox;
         //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

         int left = (b.x - b.w / 2.)*width;
         int right = (b.x + b.w / 2.)*width;
         int top = (b.y - b.h / 2.)*height;
         int bot = (b.y + b.h / 2.)*height;

         if (left < 0)
            left = 0;
         if (right > width- 1)
            right = width - 1;
         if (top < 0)
            top = 0;
         if (bot > height - 1)
            bot = height - 1;

         //int b_x_center = (left + right) / 2;
         //int b_y_center = (top + bot) / 2;
         //int b_width = right - left;
         //int b_height = bot - top;
         //sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

         // you should create directory: result_img
         //static int copied_frame_id = -1;
         //static image copy_img;
         //if (copied_frame_id != frame_id) {
         //    copied_frame_id = frame_id;
         //    if (copy_img.data) free_image(copy_img);
         //    copy_img = copy_image(im);
         //}
         //image cropped_im = crop_image(copy_img, left, top, right - left, bot - top);
         //static int img_id = 0;
         //img_id++;
         //char image_name[1024];
         //int best_class_id = selected_detections[i].best_class;
         //sprintf(image_name, "result_img/img_%d_%d_%d_%s.jpg", frame_id, img_id, best_class_id, names[best_class_id]);
         //save_image(cropped_im, image_name);
         //free_image(cropped_im);

         if (color == 1) {
            drawBoxWidthBw/*!draw_box_width_bw*/(left, top, right, bot, width, 0.8);    // 1 channel Black-White
         }else{
            drawBoxWidth/*!draw_box_width*/(left, top, right, bot, width, red, green, blue); // 3 channels RGB
         }
         if (alphabet) {
            char labelstr[4096] = { 0 };
            strcat(labelstr, names[selected_detections[i].best_class]);
            char prob_str[10];
            sprintf(prob_str, ": %.2f", selected_detections[i].det->prob[selected_detections[i].best_class]);
            strcat(labelstr, prob_str);
            int j;
            for (j = 0; j < classes; ++j) {
               if (selected_detections[i].det->prob[j] > thresh && j != selected_detections[i].best_class) {
                  strcat(labelstr, ", ");
                  strcat(labelstr, names[j]);
               }
            }
            NImage *label = getLabelV3/*!get_label_v3*/(alphabet, labelstr, (height*.02));
            //draw_label(im, top + width, left, label, rgb);
            drawWeightedLabel/*!draw_weighted_label*/(top + width, left, label, rgb, 0.7);
            label->unref();
         }
         if (selected_detections[i].det->mask) {
            NImage *mask = new$ NImage/*!float_to_image*/(selected_detections[i].det->mask,14, 14, 1);
            NImage *resized_mask =mask->resize/*!resize_image*/(b.w*width, b.h*height);
            NImage *tmask = resized_mask->thresholdImage/*!threshold_image*/(0.5);
            tmask->embed/*!embed_image*/(self, left, top);
            mask->unref();
            resized_mask->unref();
            tmask->unref();
         }
      }
      free(selected_detections);
   }

   //原型 random_augment_image image.h image.c
   NImage *randomAugment (float angle, float aspect, int low, int high, int size){
      aspect = DnnUtils.randScale/*!rand_scale*/(aspect);
      int r = DnnUtils.randInt/*!rand_int*/(low, high);
      int min = (self->height < self->width*aspect) ? self->height : self->width*aspect;
      float scale = (float)r / min;

      float rad = DnnUtils.randUniform/*!rand_uniform*/(-angle, angle) * 2.0 * M_PI / 360.;

      float dx = (self->width*scale/aspect - size) / 2.;
      float dy = (self->height*scale - size) / 2.;
      if(dx < 0)
         dx = 0;
      if(dy < 0)
         dy = 0;
      dx = DnnUtils.randUniform/*!rand_uniform*/(-dx, dx);
      dy = DnnUtils.randUniform/*!rand_uniform*/(-dy, dy);
      NImage *crop = rotateCrop(rad, scale, size, size, dx, dy, aspect);
      return crop;
   }

   //原型 resize_min image.h image.c
   NImage *resizeMin(int min){
      int w = width;
      int h = height;
      if(w < h){
         h = (h * min) / w;
         w = min;
      } else {
         w = (w * min) / h;
         h = min;
      }
      if(w == width && h == height)
         return self;
      NImage *resized = resize(w, h);
      return resized;
   }

   ~NImage(){
      if(imgData){
         a_free(imgData);
      }
   }

};


