#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>

#include  "NLayer.h"
#include  "NNetwork.h"
#include  "DnnUtils.h"
#include  "mtcs/MtcsNormData.h"
#include  "mtcs/MtcsTool.h"

impl$ NLayer{

   static LayerType getLayerType(char * type){
      if (strcmp(type, "shortcut")==0) return LayerType.SHORTCUT;
      if (strcmp(type, "crop")==0) return LayerType.CROP;
      if (strcmp(type, "cost")==0) return LayerType.COST;
      if (strcmp(type, "detection")==0) return LayerType.DETECTION;
      if (strcmp(type, "region")==0) return LayerType.REGION;
      if (strcmp(type, "yolo")==0) return LayerType.YOLO;
      if (strcmp(type, "iseg")==0) return LayerType.ISEG;
      if (strcmp(type, "conv")==0
      || strcmp(type, "convolutional")==0) return LayerType.CONVOLUTIONAL;
      if (strcmp(type, "deconv")==0
      || strcmp(type, "deconvolutional")==0) return LayerType.DECONVOLUTIONAL;
      if (strcmp(type, "activation")==0) return LayerType.ACTIVE;
      if (strcmp(type, "logistic")==0) return LayerType.LOGXENT;
      if (strcmp(type, "l2norm")==0) return LayerType.L2NORM;
      if (strcmp(type, "net")==0
      || strcmp(type, "network")==0) return LayerType.NETWORK;
      if (strcmp(type, "crnn")==0) return LayerType.CRNN;
      if (strcmp(type, "gru")==0) return LayerType.GRU;
      if (strcmp(type, "lstm") == 0) return LayerType.LSTM;
      if (strcmp(type, "rnn")==0) return LayerType.RNN;
      if (strcmp(type, "conn")==0
      || strcmp(type, "connected")==0) return LayerType.CONNECTED;
      if (strcmp(type, "max")==0
      || strcmp(type, "maxpool")==0) return LayerType.MAXPOOL;
      if (strcmp(type, "reorg")==0) return LayerType.REORG;
      if (strcmp(type, "avg")==0
      || strcmp(type, "avgpool")==0) return LayerType.AVGPOOL;
      if (strcmp(type, "dropout")==0) return LayerType.DROPOUT;
      if (strcmp(type, "lrn")==0
      || strcmp(type, "normalization")==0) return LayerType.NORMALIZATION;
      if (strcmp(type, "batchnorm")==0) return LayerType.BATCHNORM;
      if (strcmp(type, "soft")==0
      || strcmp(type, "softmax")==0) return LayerType.SOFTMAX;
      if (strcmp(type, "route")==0) return LayerType.ROUTE;
      if (strcmp(type, "upsample")==0) return LayerType.UPSAMPLE;

      return LayerType.BLANK;
   }

   /**
    * 根据类型返回名称
    */
   static char *getLayerString(LayerType a){
      switch(a){
         case LayerType.CONVOLUTIONAL:
            return "convolutional";
         case LayerType.ACTIVE:
            return "activation";
         case LayerType.DECONVOLUTIONAL:
            return "deconvolutional";
         case LayerType.CONNECTED:
            return "connected";
         case LayerType.RNN:
            return "rnn";
         case LayerType.GRU:
            return "gru";
         case LayerType.LSTM:
            return "lstm";
         case LayerType.CRNN:
            return "crnn";
         case LayerType.MAXPOOL:
            return "maxpool";
         case LayerType.REORG:
            return "reorg";
         case LayerType.AVGPOOL:
            return "avgpool";
         case LayerType.SOFTMAX:
            return "softmax";
         case LayerType.DETECTION:
            return "detection";
         case LayerType.REGION:
            return "region";
         case LayerType.YOLO:
            return "yolo";
         case LayerType.DROPOUT:
            return "dropout";
         case LayerType.CROP:
            return "crop";
         case LayerType.COST:
            return "cost";
         case LayerType.ROUTE:
            return "route";
         case LayerType.SHORTCUT:
            return "shortcut";
         case LayerType.NORMALIZATION:
            return "normalization";
         case LayerType.BATCHNORM:
            return "batchnorm";
         default:
            break;
      }
      return "none";
   }

   /**
    * 构造函数
    */
   NLayer(){
      orderNumber=-1;
      normData=NULL;
      outputData=NULL;
      batch_normalize=0;
      deltaData=NULL;
      scaleData=NULL;
      biasData=NULL;
      setInputDimen(0,0,0);
      setOutputDimen(0,0,0);
   }

   void resize(int w, int h){

   }

   /**
   * 前向生成的数据作规一化处理
   * z-score标准化
   * A=v1,v2,v3,...,vn
   * 公式 v'=(vi-A的平均值)/σa(A的均方差)
   * 由于卷积层后跟的是非线性激活函数，而通过归一化改变了数据的分布，可能使得原本工作在非线性激活区的数据跑到线性激活区了，
   * 为了解决该问题，darknet框架在z-score标准化的计算公式中加入了一个缩放因子beta和偏移量b。
   * 最终卷积后对输出数据归一化的公式是
   *  v'=β*((vi-A的平均值)/σa(A的均方差))+b
   *  β和b这两个都是超参数，即这两个参数是通过训练得到的。
   */
   //原型 forward_batchnorm_layer batchnorm_layer.h batchnorm_layer.c
   void forwardNormCPU(NetworkState state){
      if(self->type == LayerType.BATCHNORM){
         ((IOData*)state.input)->copy((IOData *)outputData);
      }

      if(self->type == LayerType.CONNECTED){
         self->outputDimen.channels=self->outputs;
         self->outputDimen.w=self->outputDimen.h=1;
      }
      if(state.train){
         normData->calcMeanAndVariance((IOData*)outputData,TRUE);
         normData->scalAxpyRollMeanAndRollVariance();
         normData->normalize((IOData*)outputData,scaleData->getScales(),biasData->getBias());
      }else{
         /*! 三个函数调用，合并成一个normalizeByRoll
         normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
         scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
         add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
         */
         normData->normalizeByRoll((IOData*)outputData,scaleData->getScales(),biasData->getBias());
      }
   }

   void forwardNormMTCS(NetworkState state){
      NNetwork *net=(NNetwork *)network;

      if(self->type == LayerType.BATCHNORM){
         ((IOData*)state.input)->copy((IOData *)outputData);
      }

       if (net->adversarial) {
          /*! 三个函数调用，合并成一个normalizeByRoll
          normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
          scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
          add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
          */
          normData->normalizeByRoll((IOData*)outputData,scaleData->getScales(),biasData->getBias());
          return;
       }

       if (state.train) {
           if (batch_normalize == 2) {
              if(self->type!=LayerType.CONVOLUTIONAL){
                  a_error("只有卷积层 batch_normalize =2");
                  return;
               }
               /* simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.x_gpu)
               * 这句原来在state.train下面 不需要了 normalize 会复制数据到x 和 x_norm
               */
               normData->copyToX((NData*)outputData);
               normData->calcMean(outputData->getDataArray());
               const int minibatch_index = net->current_subdivision + 1;
               const int max_minibatch_index = net->subdivisions;
               const float alpha = 0.01;
               int inverse_variance = 0;
               float *mcbn =getMCbn();
               float *vcbn =getVCbn();
               a_assert(mcbn!=NULL && vcbn!=NULL);
               ((MtcsNormData*)normData)->fastVcbn(outputData->getDataArray(),  minibatch_index, max_minibatch_index,
                     mcbn, vcbn,alpha, inverse_variance, 0.00001);
               ((MtcsNormData*)normData)->normalize(outputData->getDataArray(),scaleData->getScales(),
                     biasData->getBias(),inverse_variance, .00001f);
               normData->copyToXNorm((NData*)outputData);


           } else {
               /*!
               fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
               fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

               scal_ongpu(l.out_c, .99, l.rolling_mean_gpu, 1);
               axpy_ongpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
               scal_ongpu(l.out_c, .99, l.rolling_variance_gpu, 1);
               axpy_ongpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

               copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
               normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
               copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

               scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
               add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
               */

               normData->calcMeanAndVariance((IOData*)outputData,TRUE);
               normData->scalAxpyRollMeanAndRollVariance();
               normData->normalize((IOData*)outputData,scaleData->getScales(),biasData->getBias());
           }
       } else {
          /*!
           normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
           scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
           add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
           */
           normData->normalizeByRoll((IOData*)outputData,scaleData->getScales(),biasData->getBias());
       }
   }

   void forwardNorm(NetworkState state){
      if(devType==DeviceType.MTCS)
         forwardNormMTCS(state);
      else
         forwardNormCPU(state);
   }

   //原型 backward_batchnorm_layer batchnorm_layer.h batchnorm_layer.c
   void backwardNormCPU(NetworkState state){
      /*! backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);*/
      scaleData->backwardScale(deltaData,normData);
      //printf("backwardNorm----00 \n");
      /*!   scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);*/
      deltaData->scale(scaleData->getScales());
     // deltaData->checkNan("backwardNorm 00 误差数据",orderNumber);
      //printf("backwardNorm----11 \n");

      /*!mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);*/
      deltaData->calcMean(normData->getVariance());
     // deltaData->checkNan("backwardNorm 11 误差数据",orderNumber);

      /*!variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);*/
      deltaData->calcVariance(normData);
     // deltaData->checkNan("backwardNorm 22 误差数据",orderNumber);
      //printf("backwardNorm----22 \n");

      /*!normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);*/
      deltaData->normalize(normData);
     // deltaData->checkNan("backwardNorm 33 误差数据",orderNumber);
     // printf("backwardNorm----33 \n");

      /*!
      if(l.type == BATCHNORM)
      copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
      */
      if(self->type == LayerType.BATCHNORM){
         //copy_cpu(self->outputs*self->batch, self->delta, 1, net->delta, 1);
         DeltaData *prevDelta=  state.delta;
         deltaData->copy((NData*)prevDelta);
      }
   }

   //原型 backward_batchnorm_layer_gpu
   void backwardNormMTCS(NetworkState state){
      NNetwork *net=(NNetwork *)network;
      if (net->adversarial) {
         /*!inverse_variance_ongpu(l.out_c, l.rolling_variance_gpu, l.variance_gpu, 0.00001);*/
         ((MtcsNormData *)normData)->inverseVariance();
         /*!scale_bias_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);*/
         deltaData->scaleBias(normData->variance);
         /*!scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);*/
         deltaData->scaleBias(scaleData->getScales());
         return;
      }

      if (!state.train) {
         /*!simple_copy_ongpu(l.out_c, l.rolling_mean_gpu, l.mean_gpu);*/
         MtcsTool.simpleCopy(outputDimen.channels,normData->rolling_mean,normData->mean);
         /*!simple_copy_ongpu(l.out_c, l.rolling_variance_gpu, l.variance_gpu);*/
         MtcsTool.simpleCopy(outputDimen.channels,normData->rolling_variance,normData->variance);
      }

      /*!backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);*/
      biasData->calcGrad(deltaData);
      /*!backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);*/
      scaleData->backwardScale(deltaData,normData);
      /*!scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);*/
      deltaData->scaleBias(scaleData->getScales());
      /*!fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);*/
      deltaData->calcMean(normData->getVariance());
      /*!fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);*/
      deltaData->calcVariance(normData);
      /*!normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu,
         l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);*/
      deltaData->normalize(normData);

      if(self->type == LayerType.BATCHNORM)
         /*!simple_copy_ongpu(l.outputs*l.batch, l.delta_gpu, state.delta);*/
         deltaData->copy(state.delta);

      if (net->try_fix_nan) {
         /*!fix_nan_and_inf(l.scale_updates_gpu, l.n);*/
         MtcsTool.fixNanAndInf(scaleData->getUpdates(),scaleData->size);
         /*!fix_nan_and_inf(l.bias_updates_gpu, l.n);*/
         MtcsTool.fixNanAndInf(biasData->updates,biasData->size);
      }
   }

   void backwardNorm(NetworkState state){
      if(devType==DeviceType.MTCS)
         backwardNormMTCS(state);
      else
         backwardNormCPU(state);
   }

   size_t getWorkSize(){
      return workspace_size;
   }

   void setNetwork(apointer network){
      self->network=network;
   }


   void setOrderNumber(int orderNumber){
      self->orderNumber=orderNumber;
   }

   int  getOrderNumber(){
      return orderNumber;
   }

   OutputData *getOutputData(){
      return outputData;
   }

   /**
   * 实现NLayer的抽象方法initDelta
   */
   void initDelta(){
      if(deltaData)
         deltaData->setZero();
   }

   void setInputDimen(int w,int h,int channels){
      inputDimen.w=w;
      inputDimen.h=h;
      inputDimen.channels=channels;
   }

   void setOutputDimen(int w,int h,int channels){
      outputDimen.w=w;
      outputDimen.h=h;
      outputDimen.channels=channels;
   }

   void setInputDimen(int w,int h){
      inputDimen.w=w;
      inputDimen.h=h;
   }

   void setOutputDimen(int w,int h){
      outputDimen.w=w;
      outputDimen.h=h;
   }

   void saveWeights(FILE *fp){

   }
   void  saveWeightsEma(FILE *fp){

   }
   void loadWeights(FILE *fp){

   }

   void setReceptive(int *w,int *h,int *wScale,int *hScale){
      int dilation = getDilation();
      int stride=getStride();
      int size=getSize();
      dilation = max_val_cmp(1, dilation);
      stride = max_val_cmp(1,stride);
      size = max_val_cmp(1, size);

      int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
      increase_receptive = max_val_cmp(0, increase_receptive);
      *w += increase_receptive * (*wScale);
      *h += increase_receptive * (*hScale);
      *wScale *=stride;
      *hScale *=stride;
      self->receptive.w = *w;
      self->receptive.h = *h;
      self->receptive.wScale = *wScale;
      self->receptive.hScale = *hScale;
   }

   /**
    * 4个层有dilation(膨胀),它们是:connected,conv_lstm,convolutional,crnn
    */
   public$ int getDilation(){
      return 0;
   }

   public$ int getStride(){
      return 0;
   }

   public$ int getSize(){
      return 0;
   }

   public$ NLayer *getInputLayer(){
      return NULL;
   }

   public$ aboolean isAntialiasing(){
      return FALSE;
   }

   void update (int batch, float learning_rate_init, float momentum, float decay){

   }

   public$ void  update(int batch, float learning_rate_init, float momentum, float decay, float loss_scale){

   }

   //目前只有MtcsConvLayer覆盖
   public$ float *getMCbn(){
      return NULL;
   }

   public$ float *getVCbn(){
      return NULL;
   }

   /**
    * 析构函数
    */
   ~NLayer(){

   }

};

void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves)
{
    int j;
    for(j = 0; j < n; ++j){
        int parent = hier->parent[j];
        if(parent >= 0){
            predictions[j] *= predictions[parent];
        }
    }
    if(only_leaves){
        for(j = 0; j < n; ++j){
            if(!hier->leaf[j]) predictions[j] = 0;
        }
    }
}
