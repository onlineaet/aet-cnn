

#ifndef __COM_AI_DNN_N_NETWORK_H__
#define __COM_AI_DNN_N_NETWORK_H__

#include <aet.h>
#include  <aet/util/AKeyFile.h>
#include  "NLayer.h"
#include  "TruthData.h"
#include  "DeltaData.h"
#include  "ImageData.h"
#include  "ConvolutionalLayer.h"

package$ com.ai.dnn;


/**
 * 神经网络。
 * 由层，参数组成
 * momentum 动量能够在一定程度上解决这个问题。momentum 动量是依据物理学的势能与动能之间能量转换原理提出来的。
 * 当 momentum 动量越大时，其转换为势能的能量也就越大，就越有可能摆脱局部凹域的束缚，进入全局凹域。momentum 动量主
 * 要用在权重更新的时候。 一般，神经网络在更新权值时，采用如下公式: w = w - learning_rate * dw
 * 引入momentum后，采用如下公式：v = mu * v - learning_rate * dw   w = w + v
 * 其中，v初始化为0，mu是设定的一个超变量，最常见的设定值是0.9。可以这样理解上式：如果上次的momentum(v)与这次的
 * 负梯度方向是相同的，那这次下降的幅度就会加大，从而加速收敛。
 */

public$ class$ NNetwork{
   public$ enum$ LearningRatePolicy{
      CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM, SGDR
   };
   int batch;
   size_t seen;
   aint64 cur_iteration;
   int subdivisions;
   float badlabels_rejection_percentage;
   NLayer **layers;
   int layerCount;//多少层
   LearningRatePolicy policy;
 //  int imgWidth,imgHeight;//图像的宽，32的倍数
   TruthData *truthData;//标注框，分类
   float learning_rate;
   float momentum;// SGD动量大小
   float decay;   //权重衰减正则项，防止过拟合
   float gamma;
   float scale;
   float power;
   int time_steps;
   int step;
   int max_batches;
   float *scales;
   int   *steps;
   int    num_steps;
   int    burn_in;
   int num_boxes;
   float *seq_scales;
   tree *hierarchy;

   int adam;
   float B1;
   float B2;
   float eps;

   int inputs;
   int outputs;
   int notruth;
   int h, w, c;
   int random;
   int max_crop;
   int min_crop;
   int flip;
   int blur;
   int gaussian_noise;
   int mixup;
   int mosaic_bound;
   int contrastive;
   int contrastive_jit_flip;
   int contrastive_color;
   int unsupervised;
   aboolean train;
   float cost;
   float label_smooth_eps;
   int resize_step;
   int attention;
   float max_chart_loss;
   float angle;
   float aspect;
   float saturation;
   float exposure;
   float hue;

   private$ int classes;//类别数

   int show_receptive_field;
   int dynamic_minibatch;
   int letter_box;
   int benchmark_layers;

   int adversarial;
   float adversarial_lr;
   int *total_bbox;
   int *rewritten_bbox;
   float loss_scale;
   int optimized_memory;
   size_t workspace_size_limit;

   float learning_rate_min;
   int batches_per_cycle;
   int batches_cycle_mult;
   int track;
   int augment_speed;
   int init_sequential_subdivisions;
   int sequential_subdivisions;
   int try_fix_nan;
   int weights_reject_freq;
   int equidistant_point;
   float num_sigmas_reject_badlabels;
   float ema_alpha;
   float *badlabels_reject_threshold;
   float *delta_rolling_max;
   float *delta_rolling_avg;
   float *delta_rolling_std;

   void *cuda_graph;
   void *cuda_graph_exec;
   int cudnn_half;

   int use_cuda_graph;
   int *cuda_graph_ready;
   int current_subdivision;



   //临时内存，矩阵乘法中的b 减少内存分配次数
   float *workspace;

   aboolean useMtcs;

   InputData *inputData;

   NNetwork(int layerCount);

   NLayer  *getOutputLayer();
   //原型 get_network_output_size network.h network.c
   int      getOuputSize();
   void     setInitInputsAndOutputs(aboolean train);
   void     loadWeights(char *weightFile);
   void     saveWeights(char *fileName);
   //原型 int get_current_batch network.h network.c
   size_t   getCurrentBatch();
   int      resize(int w, int h);
   NLayer  *getLayer(int index);
   NLayer  *getNext(NLayer *at);
   NLayer  *getLastLayer();

   float    getCurrentRate();
   int      getLayerCount();
   //设置类别数
   public$ void setClasses(int classes);
   public$ int  getClasses();
   //原型 fuse_conv_batchnorm darknet.h network.c
   void fuseConvBatchnorm();
   //原型 calculate_binary_weights darknet.h network.c
   void calculateBinaryWeights();
   //原型 forward_network network.h network.cc
   void forward(NetworkState state);
   //原型 backward_network network.h network.cc
   void backward(NetworkState state);
   //原型 update_network network.h network.c
   void update();

   //原型 get_network_output network.h network.cc
   OutputData *getOutput();
   //原型 get_network_boxes darknet.h network.c
   Detection **getBoxes(int w, int h, float thresh, float hier,int *map, int relative, int *num, int letter);
   //原型 detection_to_json darknet.h network.c
   char *detectionToJson(Detection **dets, int nboxes, int classes, char **names, long long int frame_id, char *filename);
   //原型 make_network_boxes darknet.h network.c
   Detection **makeBoxes(float thresh, int *num);
   //原型 get_current_iteration network.h network.c
   public$ aint64    getCurrentIteration();
   //原型 train_network network.h network.c
   public$ float     train(ImageData *data);
   public$ OutputData *predict(NImage *input);
   //原型 network_predict darknet.h network.c
   public$ OutputData *predict(InputData *input);
   //原型 get_network_cost network.h network.c
   public$ float     getNetworkCost();
   //原型 is_ema_initialized network.h network.c
   public$ int       isEmaInitialized();
   public$ aboolean  tryFixNan();
   public$ aboolean  adversarial();
   public$ int       getWorkspaceSize();
   //原型 get_sequence_value network.h network.c
   public$ int       getSequenceValue();


};




#endif /* __N_MEM_H__ */

