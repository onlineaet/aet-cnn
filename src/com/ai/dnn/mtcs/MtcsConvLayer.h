

#ifndef __COM_AI_DNN_MTCS_CONV_LAYER_H__
#define __COM_AI_DNN_MTCS_CONV_LAYER_H__

#include <aet.h>

#include  "../ConvolutionalLayer.h"
#include  "MtcsIm2Col.h"
#include  "../Im2Col.h"
#include  "MtcsGemm.h"



package$ com.ai.dnn.mtcs;

/**
 * 卷积运算
 */
public$ class$ MtcsConvLayer extends$ ConvolutionalLayer {
   private$ MtcsIm2Col *im2ColMTCS;
   private$ Im2Col *im2ColCPU;
   private$ MtcsGemm *gemmMTCS;
   private$ InputData *input_antialiasing;
   private$ float *m_cbn_avg;
   private$ float *v_cbn_avg;
   struct{
      float *weights;
      float *biases;
      float *weightUpdates;
      float *biasUpdates;
      float *scales;
      float *rollMean;
      float *rollVariance;
      float *m;
      float *v;
      float *biasEma;
      float *scaleEma;
      float *weightEma;
   }saveData;

   public$ MtcsConvLayer(int batch, int steps, int h, int w, int c,
            int n, int groups, int size, int stride_x, int stride_y, int dilation,
            int padding, ActivationType activation, int batch_normalize, int binary,
            int xnor, int adam, int use_bin_output, int index, int antialiasing,
            ConvolutionalLayer *share_layer, int assisted_excitation, int deform, int train);

};




#endif /* __N_MEM_H__ */

