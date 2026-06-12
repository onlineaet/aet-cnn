#include "Im2Col.h"

static inline void set_bit(unsigned char *const dst, size_t index)
{
    size_t dst_i = index / 8;
    int dst_shift = index % 8;
    dst[dst_i] |= 1 << dst_shift;
    //dst[dst_i] |= 1 << (8 - dst_shift);
}


impl$ Im2Col{

   // Function uses casting from int to unsigned to compare if value of
   // parameter a is greater or equal to zero and lower than value of
   // parameter b. The b parameter is of type signed and is always positive,
   // therefore its value is always lower than 0x800... where casting
   // negative value of a parameter converts it to value higher than 0x800...
   // The casting allows to use one condition instead of two.
   inline  static int is_a_ge_zero_and_a_lt_b(int a, int b){
      return (unsigned)(a) < (unsigned)(b);
   }

   //原型 im2col_get_pixel im2col.h im2col.c
   float getPixel(float *im, int height, int width, int channels,
                           int row, int col, int channel, int pad){
       row -= pad;
       col -= pad;

       if (row < 0 || col < 0 ||
           row >= height || col >= width) return 0;
       return im[col + width*(row + height*channel)];
   }

   //From Berkeley Vision's Caffe!
   //https://github.com/BVLC/caffe/blob/master/LICENSE
   //原型 im2col_cpu im2col.h im2col.c
   void im2col(float* data_im,
        int channels,  int height,  int width,
        int ksize,  int stride, int pad, float* data_col){
       int c,h,w;
       int height_col = (height + 2*pad - ksize) / stride + 1;
       int width_col = (width + 2*pad - ksize) / stride + 1;

       int channels_col = channels * ksize * ksize;
       for (c = 0; c < channels_col; ++c) {
           int w_offset = c % ksize;
           int h_offset = (c / ksize) % ksize;
           int c_im = c / ksize / ksize;
           for (h = 0; h < height_col; ++h) {
               for (w = 0; w < width_col; ++w) {
                   int im_row = h_offset + h * stride;
                   int im_col = w_offset + w * stride;
                   int col_index = (c * height_col + h) * width_col + w;
                   data_col[col_index] = getPixel(data_im, height, width, channels,
                           im_row, im_col, c_im, pad);
               }
           }
       }
   }


   // https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
   //原型 im2col_cpu_ext im2col.h im2col.c
   void im2col(const float* data_im, const int channels,
       const int height, const int width, const int kernel_h, const int kernel_w,
       const int pad_h, const int pad_w,
       const int stride_h, const int stride_w,
       const int dilation_h, const int dilation_w,
       float* data_col){
       const int output_h = (height + 2 * pad_h -
           (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
       const int output_w = (width + 2 * pad_w -
           (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
       const int channel_size = height * width;
       int channel, kernel_row, kernel_col, output_rows, output_col;
       for (channel = channels; channel--; data_im += channel_size) {
           for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
               for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                   int input_row = -pad_h + kernel_row * dilation_h;
                   for (output_rows = output_h; output_rows; output_rows--) {
                       if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                           for (output_col = output_w; output_col; output_col--) {
                               *(data_col++) = 0;
                           }
                       }else{
                           int input_col = -pad_w + kernel_col * dilation_w;
                           for (output_col = output_w; output_col; output_col--) {
                               if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                   *(data_col++) = data_im[input_row * width + input_col];
                               }
                               else {
                                   *(data_col++) = 0;
                               }
                               input_col += stride_w;
                           }
                       }
                       input_row += stride_h;
                   }
               }
           }
       }
   }

   //原型  col2im_gpu_ext col2im.h col2im.c


   void col2im_add_pixel(float *im, int height, int width, int channels,
                           int row, int col, int channel, int pad, float val){
       row -= pad;
       col -= pad;
       if (row < 0 || col < 0 ||
           row >= height || col >= width) return;
       im[col + width*(row + height*channel)] += val;
   }
   //This one might be too, can't remember.
   //原型 col2im_cpu col2im.h col2im.c
   void col2im(float* data_col,int channels,  int height,  int width,
            int ksize,  int stride, int pad, float* data_im){
       int c,h,w;
       int height_col = (height + 2*pad - ksize) / stride + 1;
       int width_col = (width + 2*pad - ksize) / stride + 1;

       int channels_col = channels * ksize * ksize;
       for (c = 0; c < channels_col; ++c) {
           int w_offset = c % ksize;
           int h_offset = (c / ksize) % ksize;
           int c_im = c / ksize / ksize;
           for (h = 0; h < height_col; ++h) {
               for (w = 0; w < width_col; ++w) {
                   int im_row = h_offset + h * stride;
                   int im_col = w_offset + w * stride;
                   int col_index = (c * height_col + h) * width_col + w;
                   float val = data_col[col_index];
                   col2im_add_pixel(data_im, height, width, channels,
                           im_row, im_col, c_im, pad, val);
               }
           }
       }
   }
   // ----------------------------------------
   void caffe_set(const int N, const float alpha, float* Y) {
       if (alpha == 0) {
           memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
           return;
       }
       int i;
       for (i = 0; i < N; ++i) {
           Y[i] = alpha;
       }
   }

   // https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
   //原型  col2im_gpu_ext col2im.h col2im.c
   void col2im(const float* data_col, const int channels,
       const int height, const int width, const int kernel_h, const int kernel_w,
       const int pad_h, const int pad_w,
       const int stride_h, const int stride_w,
       const int dilation_h, const int dilation_w,
       float* data_im)
   {
       caffe_set(height * width * channels, 0.0F, data_im);
       const int output_h = (height + 2 * pad_h -
           (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
       const int output_w = (width + 2 * pad_w -
           (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
       const int channel_size = height * width;
       int channel, kernel_row, kernel_col, output_rows, output_col;
       for (channel = channels; channel--; data_im += channel_size) {
           for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
               for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                   int input_row = -pad_h + kernel_row * dilation_h;
                   for (output_rows = output_h; output_rows; output_rows--) {
                       if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                           data_col += output_w;
                       }
                       else {
                           int input_col = -pad_w + kernel_col * dilation_w;
                           for (output_col = output_w; output_col; output_col--) {
                               if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                   data_im[input_row * width + input_col] += *data_col;
                               }
                               data_col++;
                               input_col += stride_w;
                           }
                       }
                       input_row += stride_h;
                   }
               }
           }
       }
   }

   //From Berkeley Vision's Caffe!
   //https://github.com/BVLC/caffe/blob/master/LICENSE
   //原型 im2col_cpu_custom_bin gemm.h gemm.c
   void im2colCustomBin(float* data_im,
       int channels, int height, int width,
       int ksize, int stride, int pad, float* data_col, int bit_align){
       int c;
       const int height_col = (height + 2 * pad - ksize) / stride + 1;
       const int width_col = (width + 2 * pad - ksize) / stride + 1;
       const int channels_col = channels * ksize * ksize;

       // optimized version
       if (height_col == height && width_col == width && stride == 1 && pad == 1){
           int new_ldb = bit_align;
           #pragma omp parallel for
           for (c = 0; c < channels_col; ++c) {
               int h, w;
               int w_offset = c % ksize;
               int h_offset = (c / ksize) % ksize;
               int c_im = c / ksize / ksize;
               for (h = pad; h < height_col - pad; ++h) {
                   for (w = pad; w < width_col - pad - 8; w += 1) {
                       int im_row = h_offset + h - pad;
                       int im_col = w_offset + w - pad;
                       //int col_index = (c * height_col + h) * width_col + w;
                       int col_index = c * new_ldb + h * width_col + w;

                       float val = data_im[im_col + width*(im_row + height*c_im)];
                       if (val > 0) set_bit((unsigned char*)data_col, col_index);
                   }

                   for (; w < width_col - pad; ++w) {
                       int im_row = h_offset + h - pad;
                       int im_col = w_offset + w - pad;
                       //int col_index = (c * height_col + h) * width_col + w;
                       int col_index = c * new_ldb + h * width_col + w;

                       //data_col[col_index] = data_im[im_col + width*(im_row + height*c_im)];
                       float val = data_im[im_col + width*(im_row + height*c_im)];
                       if (val > 0) set_bit((unsigned char*)data_col, col_index);
                   }
               }

               {
                   w = 0;
                   for (h = 0; h < height_col; ++h) {
                       int im_row = h_offset + h;
                       int im_col = w_offset + w;
                       //int col_index = (c * height_col + h) * width_col + w;
                       int col_index = c * new_ldb + h * width_col + w;

                       //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       float val = getPixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       if (val > 0) set_bit((unsigned char*)data_col, col_index);
                   }
               }

               {
                   w = width_col - 1;
                   for (h = 0; h < height_col; ++h) {
                       int im_row = h_offset + h;
                       int im_col = w_offset + w;
                       //int col_index = (c * height_col + h) * width_col + w;
                       int col_index = c * new_ldb + h * width_col + w;

                       //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       float val = getPixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       if (val > 0) set_bit((unsigned char*)data_col, col_index);
                   }
               }

               {
                   h = 0;
                   for (w = 0; w < width_col; ++w) {
                       int im_row = h_offset + h;
                       int im_col = w_offset + w;
                       //int col_index = (c * height_col + h) * width_col + w;
                       int col_index = c * new_ldb + h * width_col + w;

                       //data_col[col_index] = getPixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       float val = getPixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       if (val > 0) set_bit((unsigned char*)data_col, col_index);
                   }
               }

               {
                   h = height_col - 1;
                   for (w = 0; w < width_col; ++w) {
                       int im_row = h_offset + h;
                       int im_col = w_offset + w;
                       //int col_index = (c * height_col + h) * width_col + w;
                       int col_index = c * new_ldb + h * width_col + w;

                       //data_col[col_index] = getPixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       float val = getPixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                       if (val > 0) set_bit((unsigned char*)data_col, col_index);
                   }
               }
           }

       }
       else {
           printf("\n Error: is no non-optimized version \n");
       }
   }


};
