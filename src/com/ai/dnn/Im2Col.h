

#ifndef __COM_AI_CNN_IM_2_COL_H__
#define __COM_AI_CNN_IM_2_COL_H__

#include <aet.h>

package$ com.ai.dnn;

public$ class$ Im2Col{
   //原型 im2col_cpu_ext im2col.h im2col.c
   public$ void im2col(const float* data_im, const int channels,
       const int height, const int width, const int kernel_h, const int kernel_w,
       const int pad_h, const int pad_w,
       const int stride_h, const int stride_w,
       const int dilation_h, const int dilation_w,
       float* data_col);
   //原型 im2col_cpu im2col.h im2col.c
   public$ void im2col(float* data_im,
        int channels,  int height,  int width,
        int ksize,  int stride, int pad, float* data_col);
   //原型 im2col_cpu_custom_bin gemm.h gemm.c
   void im2colCustomBin(float* data_im,
       int channels, int height, int width,
       int ksize, int stride, int pad, float* data_col, int bit_align);
   //原型  col2im_gpu_ext col2im.h col2im.c
   public$ void col2im(const float* data_col, const int channels,
       const int height, const int width, const int kernel_h, const int kernel_w,
       const int pad_h, const int pad_w, const int stride_h,
       const int stride_w, const int dilation_h, const int dilation_w,
       float* data_im);
   //原型 col2im_cpu col2im.h col2im.c
   public$ void col2im(float* data_col,int channels,  int height,  int width,
            int ksize,  int stride, int pad, float* data_im);
   //原型 im2col_get_pixel im2col.h im2col.c
   float getPixel(float *im, int height, int width, int channels,
                           int row, int col, int channel, int pad);

};

#endif /* __COM_AI_CNN_IM_2_COL_H__ */
