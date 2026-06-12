

#ifndef __COM_AI_DNN_CONVOLUTIONAL_LAYER_H__
#define __COM_AI_DNN_CONVOLUTIONAL_LAYER_H__

#include <aet.h>

#include  "NNetwork.h"
#include  "ComputingUnit.h"
#include  <aet/util/AMutex.h>
#include  "WeightData.h"
#include "Im2Col.h"


package$ com.ai.dnn;

/**
 * 卷积层
 */
public$ class$ ConvolutionalLayer extends$ NLayer{

   int filters;
   WeightData *binWeightData;
   int flipped;
   float dot;
   int stride;
   int stride_x;
   int stride_y;
   int ksize ;
   int pad ;
   int binary;
   int xnor;
   int antialiasing;
   int stream;
   int wait_stream_id;
   int assisted_excitation;
   ConvolutionalLayer  *share_layer;
   ConvolutionalLayer  *input_layer;

   int index;
   int groups;
   int n;
   int use_bin_output;
   int steps;
   int dilation;
   int nweights;
   //float *binary_weights;
   char *cweights;
   float *scales;
   InputData *binary_input;
   int bit_align;
   float *mean_arr;
   auint32 *bin_re_packed_input;
   int lda_align;
   char *t_bit_input;
   OutputData *activation_input;

   int sway;
   int rotate;
   int stretch;
   int stretch_sway;
   float angle;
   int grad_centr;
   float reverse;
   int coordconv;
   int B1;
   int B2;
   int eps;
   ActivationType activation;
   int adam;
   float *m;
   float *v;
   float *bias_m;
   float *scale_m;
   float *bias_v;
   float *scale_v;
   int new_lda;
   int align_bit_weights_size;
   char *align_bit_weights;
   //原型 bin_conv_shortcut_in_gpu
   float *bin_conv_shortcut_in;
   //原型 bin_conv_shortcut_out_gpu
   float *bin_conv_shortcut_out;
   Im2Col *im2Col;
   Gemm *gemm;

   ConvolutionalLayer(int batch, int steps, int h, int w, int c,
            int n, int groups, int size, int stride_x, int stride_y, int dilation,
            int padding, ActivationType activation, int batch_normalize, int binary,
            int xnor, int adam, int use_bin_output, int index, int antialiasing,
            ConvolutionalLayer *share_layer, int assisted_excitation, int deform, int train);
   //原型 free_convolutional_batchnorm convolutional_layer.h convolutional_layer.c
   public$ void freeBatchnorm();
   //原型 binary_align_weights convolutional_layer.h convolutional_layer.c
   public$ void binaryAlignWeights();
   //原型 convolutional_out_height convolutional_layer.h convolutional_layer.c
   public$ int getOutHeight();
   //原型 convolutional_out_height convolutional_layer.h convolutional_layer.c
   public$ int getOutWidth();
   //原型 swap_binary convolutional_layer.h convolutional_layer.c
   void swapBinary();

   protected$ void transposeMatrix(float *a, int rows, int cols);

   //原型 binarize_weights convolutional_layer.h convolutional_layer.c
   public$ static void binarizeWeights(float *weights, int n, int size, float *binary);


   //原型 assisted_excitation_forward convolutional_layer.h convolutional_layer.c
   public$ void assistedExcitationForward(NetworkState state);
   //原型 fuse_conv_batchnorm darknet.h network.c
   public$ void fuseConvBatchnorm();

};




#endif /* __N_MEM_H__ */

