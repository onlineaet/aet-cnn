

#ifndef __COM_AI_DNN_N_LAYER_H__
#define __COM_AI_DNN_N_LAYER_H__

#include <aet.h>
#include <aet/time/Time.h>
#include  "Activation.h"
#include  "NormData.h"
#include  "DeltaData.h"
#include  "BiasData.h"
#include  "WeightData.h"
#include  "ScaleData.h"
#include  "DataFactory.h"
#include  "TruthData.h"
#include  "IOData.h"

package$ com.ai.dnn;

// network.h
typedef struct _NetworkState{
  // float *truth[512];//分类数
   TruthData *truth;//真实数据
   InputData *input;
   DeltaData *delta;
   float *workspace;
   int train;
   int index;
} NetworkState;

/**
 * 损失函数loss function 也叫代价函数 cost function
 */
typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} CostType;

// tree.h
typedef struct tree {
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

/**
 * Deep Neural Networks,
 */
public$ abstract$ class$ NLayer{
   /**
   * 网络层的类型
   */
   public$ enum$ LayerType{
      CONVOLUTIONAL,
      DECONVOLUTIONAL,
      CONNECTED,
      MAXPOOL,
      LOCAL_AVGPOOL,
      SOFTMAX,
      DETECTION,
      DROPOUT,
      CROP,
      ROUTE,
      COST,
      NORMALIZATION,
      AVGPOOL,
      SHORTCUT,
      ACTIVE,
      RNN,
      GRU,
      LSTM,
      CONV_LSTM,
      CRNN,
      BATCHNORM,
      NETWORK,
      XNOR,
      REGION,
      YOLO,
      ISEG,
      REORG,
      UPSAMPLE,
      LOGXENT,
      L2NORM,
      BLANK,
      GAUSSIAN_YOLO,
      CONTRASTIVE
   };
   public$ enum$ DeviceType{
     CPU,AVX,MTCS
   };

   public$ static LayerType getLayerType(char * type);
   public$ static char     *getLayerString(LayerType a);

   LayerType type;
   ActivationType activationType;
   CostType costType;
   apointer network;

   int batch_normalize;
   int batch;
   int inputs; //输入大小
   int outputs;//输出大小
   float bflops;//计算量

   int truth;
   float smooth;
   float jitter;
   float learning_rate_scale;
   int   random;

   int orderNumber;//层在网络中的序号

   int onlyforward;
   int stopbackward;
   int dontload;
   int dontsave;
   int dontloadscales;
   int numload;

   size_t workspace_size;
   OutputData *outputData; //输出数据
   protected$ NormData *normData;//归一化数据
   protected$ DeltaData *deltaData; //误差数据
   BiasData *biasData;//偏差数据 y=wx+b 偏差就是线性中的b
   WeightData *weightData;//权重数据
   ScaleData *scaleData;

   struct {
      int w,h,channels;
   }outputDimen;

   struct {
      int w,h,channels;
   }inputDimen;

   //dark-alex新加的
   int keep_delta_gpu;
   int use_bin_output;
   struct {
      int w,h,wScale,hScale;
   }receptive;
   float clip;
   int dynamic_minibatch;
   int dont_update;
   int burnin_update;
   int train_only_bn;
   int train;
   int deform;
   int t;

   DeviceType devType;
   public$ abstract$ void forward(NetworkState state);
   public$ abstract$ void backward(NetworkState state);
   //原型 forward_batchnorm_layer batchnorm_layer.h batchnorm_layer.c
   public$ void    forwardNorm(NetworkState state);
   public$ void    backwardNorm(NetworkState state);
   //原型 update darknet.h
   public$ void    update (int batch, float learning_rate_init, float momentum, float decay);
   public$ void    update(int batch, float learning_rate_init, float momentum, float decay, float loss_scale);

   public$ size_t  getWorkSize();
   public$ void    setNetwork(apointer network);
   protected$ void setOrderNumber(int orderNumber);//第几层的序号
   protected$ int  getOrderNumber();
   public$ OutputData *getOutputData();
   public$ void initDelta();

   public$ void setInputDimen(int w,int h,int channels);
   public$ void setOutputDimen(int w,int h,int channels);
   public$ void setInputDimen(int w,int h);
   public$ void setOutputDimen(int w,int h);
   public$ void loadWeights(FILE *fp);
   public$ void saveWeights(FILE *fp);
   public$ void saveWeightsEma(FILE *fp);


   public$ void resize(int w, int h);

   public$ int getDilation();
   public$ int getStride();
   public$ int getSize();
   public$ void setReceptive(int *w,int *h,int *wScale,int *hScale);
   public$ NLayer *getInputLayer();
   public$ aboolean isAntialiasing();
   //目前只有MtcsConvLayer覆盖
   public$ float *getMCbn();
   public$ float *getVCbn();

};

void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);



#endif /* __N_LAYER_H__ */

