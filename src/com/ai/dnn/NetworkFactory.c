#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/util/AKeyFile.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/System.h>
#include <aet/time/Time.h>
#include <aet/mtcs/MtcsMem.h>

#include "NetworkFactory.h"
#include "NNetwork.h"
#include "ConvolutionalLayer.h"
#include "mtcs/MtcsConvLayer.h"
#include "mtcs/MtcsMaxPoolLayer.h"

#include "ShortcutLayer.h"
#include "RouteLayer.h"
#include "UpsampleLayer.h"
#include "YoloLayer.h"
#include "MaxPoolLayer.h"
#include "AvgPoolLayer.h"
#include "SoftMaxLayer.h"
#include "DropoutLayer.h"
#include "NLayer.h"
#include "Activation.h"
#include "DnnUtils.h"
#include "CostLayer.h"
#include "cnnmicro.h"

typedef struct _size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    int train;
    NNetwork *net;
} size_params;

impl$ NetworkFactory{

   static NetworkFactory *getInstance(){
      static NetworkFactory *singleton = NULL;
      if (!singleton){
         singleton =new$ NetworkFactory();
         singleton->useMtcs = use_mtcs;
      }
      return singleton;
   }

   NetworkFactory(){
      cfgFile=NULL;
   }

   void setTrainOnlyBn(NNetwork *net){
      int count= net->getLayerCount();
      int train_only_bn = 0;
      int i;
      for(i=count-1;i>=0;--i){
         NLayer *layer=net->getLayer(i);
         if (layer->train_only_bn)
         train_only_bn = layer->train_only_bn;  // set l.train_only_bn for all previous layers
         if(train_only_bn){
            layer->train_only_bn = train_only_bn;
            if(layer->type == LayerType.CONV_LSTM){
               fprintf(stderr,"setTrainOnlyBn 类型是 CONV_LSTM 未实现。\n");
            }else if(layer->type == LayerType.CRNN){
               fprintf(stderr,"setTrainOnlyBn 类型是 CRNN 未实现。\n");
            }
         }
      }
   }

   ConvolutionalLayer *createConvolutional(char *group, size_params *params){
      int n = cfgFile->optionFindInt(group,  "filters",1);
      int groups = cfgFile->optionFindInt(group,  "groups", 1);
      int size = cfgFile->optionFindInt(group,  "size",1);
      int stride = -1;
      //int stride = option_find_int(options, "stride",1);
      int stride_x =cfgFile->optionFindIntQuiet(group,  "stride_x", -1);
      int stride_y =cfgFile->optionFindIntQuiet(group,  "stride_y", -1);
      if (stride_x < 1 || stride_y < 1) {
         stride = cfgFile->optionFindInt(group,  "stride", 1);
         if (stride_x < 1)
            stride_x = stride;
         if (stride_y < 1)
            stride_y = stride;
      }else {
         stride = cfgFile->optionFindInt(group,  "stride", 1);
      }
      int dilation = cfgFile->optionFindIntQuiet(group,  "dilation", 1);
      int antialiasing = cfgFile->optionFindIntQuiet(group,  "antialiasing", 0);
      if (size == 1)
         dilation = 1;
      int pad =cfgFile->optionFindIntQuiet(group,  "pad",0);
      int padding = cfgFile->optionFindIntQuiet(group,  "padding",0);
      if(pad)
         padding = size/2;

      ActivationType activation = cfgFile->getActivation(group,"activation", "logistic");

      int assisted_excitation = cfgFile->optionFindFloatQuiet(group, "assisted_excitation", 0);

      int share_index = cfgFile->optionFindIntQuiet(group,  "share_index", -1000000000);
      ConvolutionalLayer *share_layer = NULL;
      if(share_index >= 0)
         share_layer = (ConvolutionalLayer *)params->net->layers[share_index];
      else if(share_index != -1000000000)
         share_layer = (ConvolutionalLayer *)params->net->layers[params->index + share_index];

      int batch,h,w,c;
      h = params->h;
      w = params->w;
      c = params->c;
      batch=params->batch;
      if(!(h && w && c))
         a_error("Layer before convolutional layer must output image.\n");
      int batch_normalize =cfgFile->optionFindIntQuiet(group, "batch_normalize", 0);
      int cbn = cfgFile->optionFindIntQuiet(group, "cbn", 0);
      if (cbn)
         batch_normalize = 2;
      int binary = cfgFile->optionFindIntQuiet(group, "binary", 0);
      int xnor = cfgFile->optionFindIntQuiet(group,"xnor", 0);
      int use_bin_output = cfgFile->optionFindIntQuiet(group, "bin_output", 0);
      int sway = cfgFile->optionFindIntQuiet(group, "sway", 0);
      int rotate = cfgFile->optionFindIntQuiet(group, "rotate", 0);
      int stretch = cfgFile->optionFindIntQuiet(group,"stretch", 0);
      int stretch_sway = cfgFile->optionFindIntQuiet(group, "stretch_sway", 0);
      if ((sway + rotate + stretch + stretch_sway) > 1) {
         printf(" Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer \n");
         exit(0);
      }
      int deform = sway || rotate || stretch || stretch_sway;
      if (deform && size == 1) {
         printf(" Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer \n");
         exit(0);
      }
      ConvolutionalLayer *layer=NULL;
      if(useMtcs)
         layer = new$ MtcsConvLayer(batch,1,h,w,c,n,groups,size,
            stride_x,stride_y,dilation,padding,activation, batch_normalize,
            binary, xnor, params->net->adam, use_bin_output, params->index,
            antialiasing, share_layer, assisted_excitation, deform, params->train);
      else
         layer = new$ ConvolutionalLayer(batch,1,h,w,c,n,groups,size,
            stride_x,stride_y,dilation,padding,activation, batch_normalize,
            binary, xnor, params->net->adam, use_bin_output, params->index,
            antialiasing, share_layer, assisted_excitation, deform, params->train);

      layer->flipped = cfgFile->optionFindIntQuiet(group, "flipped", 0);
      layer->dot = cfgFile->optionFindIntQuiet(group, "dot", 0);
      layer->sway = sway;
      layer->rotate = rotate;
      layer->stretch = stretch;
      layer->stretch_sway = stretch_sway;
      layer->angle = cfgFile->optionFindFloatQuiet(group, "angle", 15);
      layer->grad_centr = cfgFile->optionFindIntQuiet(group,"grad_centr", 0);
      layer->reverse = cfgFile->optionFindFloatQuiet(group, "reverse", 0);
      layer->coordconv = cfgFile->optionFindIntQuiet(group, "coordconv", 0);

      layer->stream = cfgFile->optionFindIntQuiet(group,"stream", -1);
      layer->wait_stream_id = cfgFile->optionFindIntQuiet(group,"wait_stream", -1);
      if(params->net->adam){
         layer->B1 = params->net->B1;
         layer->B2 = params->net->B2;
         layer->eps = params->net->eps;
      }
      layer->setNetwork((apointer)params->net);
      return layer;
   }

   ShortcutLayer *createShortcut(char *group, size_params *params){
      ActivationType activation = cfgFile->getActivation(group,"activation", "linear");
      char *weights_type_str = cfgFile->getStr(group,"weights_type","none");
      WEIGHTS_TYPE_T weights_type = NO_WEIGHTS;
      if(strcmp(weights_type_str, "per_feature") == 0 || strcmp(weights_type_str, "per_layer") == 0)
         weights_type = PER_FEATURE;
      else if (strcmp(weights_type_str, "per_channel") == 0)
         weights_type = PER_CHANNEL;
      else if (strcmp(weights_type_str, "none") != 0) {
         printf("Error: Incorrect weights_type = %s \n Use one of: none, per_feature, per_channel \n", weights_type_str);
         getchar();
         exit(0);
      }

      char *weights_normalization_str = cfgFile->getStr(group,"weights_normalization","none");
      WEIGHTS_NORMALIZATION_T weights_normalization = NO_NORMALIZATION;
      if (strcmp(weights_normalization_str, "relu") == 0 || strcmp(weights_normalization_str, "avg_relu") == 0)
         weights_normalization = RELU_NORMALIZATION;
      else if (strcmp(weights_normalization_str, "softmax") == 0)
         weights_normalization = SOFTMAX_NORMALIZATION;
      else if (strcmp(weights_type_str, "none") != 0) {
         printf("Error: Incorrect weights_normalization = %s \n Use one of: none, relu, softmax \n", weights_normalization_str);
         getchar();
         exit(0);
      }
      // char *l = option_find(options, "from");
      char *l =cfgFile->getStr(group,"from",0);
      int len = strlen(l);
      if (!l)
         a_error("Route Layer must specify input layers: from = ...");
      int n = 1;
      int i;
      for (i = 0; i < len; ++i) {
         if (l[i] == ',')
            ++n;
      }

      NLayer **layers=(NLayer**)calloc(n,sizeof(NLayer *));

      float **layers_output_gpu = (float **)calloc(n, sizeof(float *));
      float **layers_delta_gpu = (float **)calloc(n, sizeof(float *));

      for (i = 0; i < n; ++i) {
         int index = atoi(l);
         l = strchr(l, ',') + 1;
         if (index < 0)
            index = params->index + index;
         layers[i]=(NLayer*)params->net->getLayer(index);
      }


      ShortcutLayer *layer = new$ ShortcutLayer(params->batch, n, layers,
                                      params->w, params->h, params->c,
                                      layers_output_gpu, layers_delta_gpu, weights_type,
                                      weights_normalization, activation, params->train);
                                      layer->setNetwork((apointer)params->net);

      layer->setNetwork((apointer)params->net);
      free(layers_output_gpu);
      free(layers_delta_gpu);
      for (i = 0; i < n; ++i) {
         NLayer *from =layers[i];
         a_assert(params->w ==from->outputDimen.w && params->h == from->outputDimen.h);
         if (params->w != from->outputDimen.w
               || params->h != from->outputDimen.h
               || params->c != from->outputDimen.channels)
            fprintf(stderr, " (%4d x%4d x%4d) + (%4d x%4d x%4d) \n",
                  params->w, params->h, params->c, from->outputDimen.w, from->outputDimen.h, from->outputDimen.channels);
      }

      return layer;
   }

   RouteLayer *createRoute(char *group, size_params *params){
      char *l = cfgFile->getStr(group,"layers",TRUE);
      if(!l)
         a_error("Route Layer must specify input layers");
      int len = strlen(l);
      int n = 1;
      int i;
      for(i = 0; i < len; ++i){
         if (l[i] == ',')
            ++n;
      }

     // int* layers = (int*)xcalloc(n, sizeof(int));
     // int* sizes = (int*)xcalloc(n, sizeof(int));
      NLayer **layers=(NLayer **)xcalloc(n,sizeof(NLayer *));
      for(i = 0; i < n; ++i){
         int index = atoi(l);
         l = strchr(l, ',')+1;
         if(index < 0)
            index = params->index + index;
         //layers[i] = index;
        // sizes[i] = params->net->layers[index]->outputs;
         layers[i]=params->net->layers[index];
      }
      int batch = params->batch;
      int groups = cfgFile->optionFindIntQuiet(group, "groups", 1);
      int group_id = cfgFile->optionFindIntQuiet(group,"group_id", 0);

      RouteLayer *layer = new$ RouteLayer(batch, n, layers, groups, group_id,params->net);

      /*!
      ConvolutionalLayer *first = params->net->layers[layers[0]];
      layer.out_w = first.out_w;
      layer.out_h = first.out_h;
      layer.out_c = first.out_c;
      for(i = 1; i < n; ++i){
      int index = layers[i];
      convolutional_layer next = params.net.layers[index];
      if(next.out_w == first.out_w && next.out_h == first.out_h){
      layer.out_c += next.out_c;
      }else{
      fprintf(stderr, " The width and height of the input layers are different. \n");
      layer.out_h = layer.out_w = layer.out_c = 0;
      }
      }
      layer.out_c = layer.out_c / layer.groups;

      layer.w = first.w;
      layer.h = first.h;
      layer.c = layer.out_c;
      */
      layer->updataDimension();
      layer->stream = cfgFile->optionFindIntQuiet(group, "stream", -1);
      layer->wait_stream_id = cfgFile->optionFindIntQuiet(group,  "wait_stream", -1);

      if (n > 3)
         fprintf(stderr, " \t    ");
      else if (n > 1)
         fprintf(stderr, " \t            ");
      else
         fprintf(stderr, " \t\t            ");

      fprintf(stderr, "           ");
      if (layer->groups > 1)
         fprintf(stderr, "%d/%d", layer->group_id, layer->groups);
      else
         fprintf(stderr, "   ");
      fprintf(stderr, " -> %4d x%4d x%4d \n", layer->outputDimen.w, layer->outputDimen.h, layer->outputDimen.channels);
      return layer;
   }


   UpsampleLayer * createUpsample(char *group, size_params *params){
      int stride = cfgFile->optionFindInt(group, "stride",2);
      float scale = cfgFile->optionFindFloatQuiet(group, "scale", 1);
      UpsampleLayer *layer = new$ UpsampleLayer(params->batch, params->w, params->h, params->c, stride,scale);
      layer->setNetwork((apointer)params->net);
      return layer;
   }

   DropoutLayer *createDropoutLayer(char *group, size_params *params){
      float probability = cfgFile->optionFindFloat(group,  "probability", .2);
      int dropblock = cfgFile->optionFindIntQuiet(group, "dropblock", 0);
      float dropblock_size_rel = cfgFile->optionFindFloatQuiet(group, "dropblock_size_rel", 0);
      int dropblock_size_abs = cfgFile->optionFindFloatQuiet(group,"dropblock_size_abs", 0);
      if (dropblock_size_abs > params->w || dropblock_size_abs > params->h) {
         printf(" [dropout] - dropblock_size_abs = %d that is bigger than layer size %d x %d \n",
                  dropblock_size_abs, params->w, params->h);
         dropblock_size_abs = min_val_cmp(params->w, params->h);
      }
      if (dropblock && !dropblock_size_rel && !dropblock_size_abs) {
         printf(" [dropout] - None of the parameters (dropblock_size_rel or dropblock_size_abs) \
               are set, will be used: dropblock_size_abs = 7 \n");
         dropblock_size_abs = 7;
      }
      if (dropblock_size_rel && dropblock_size_abs) {
         printf(" [dropout] - Both parameters are set, only the parameter will be used: dropblock_size_abs = %d \n",
               dropblock_size_abs);
         dropblock_size_rel = 0;
      }
      DropoutLayer *layer = new$ DropoutLayer(params->batch, params->inputs, probability,
            dropblock, dropblock_size_rel, dropblock_size_abs, params->w, params->h, params->c);
      layer->outputDimen.w = params->w;
      layer->outputDimen.h = params->h;
      layer->outputDimen.channels= params->c;
      return layer;
   }

   int *parse_yolo_mask(char *a, int *num){
      int *mask = 0;
      if (a) {
         int len = strlen(a);
         int n = 1;
         int i;
         for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
         }
         mask = (int*)xcalloc(n, sizeof(int));
         for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
         }
         *num = n;
      }
      return mask;
   }

   /**
   * 如果没有配置mask，num等于total
   * 否则，num等于mask的维数
   */

   float *get_classes_multipliers(char *cpc, const int classes, const float max_delta){
      float *classes_multipliers = NULL;
      if (cpc) {
      int classes_counters = classes;
      int *counters_per_class = parse_yolo_mask(cpc, &classes_counters);
      if (classes_counters != classes) {
         printf(" number of values in counters_per_class = %d doesn't match with classes = %d \n", classes_counters, classes);
         a_error("Error!");
      }
      float max_counter = 0;
      int i;
      for (i = 0; i < classes_counters; ++i) {
         if (counters_per_class[i] < 1)
            counters_per_class[i] = 1;
         if (max_counter < counters_per_class[i])
            max_counter = counters_per_class[i];
      }
      classes_multipliers = (float *)calloc(classes_counters, sizeof(float));
      for (i = 0; i < classes_counters; ++i) {
         classes_multipliers[i] = max_counter / counters_per_class[i];
         if(classes_multipliers[i] > max_delta)
            classes_multipliers[i] = max_delta;
      }
      free(counters_per_class);
      printf(" classes_multipliers: ");
      for (i = 0; i < classes_counters; ++i) printf("%.1f, ", classes_multipliers[i]);
         printf("\n");
      }
      return classes_multipliers;
   }


   YoloLayer *createYolo(char *group, size_params *params){
      int classes =cfgFile->optionFindInt(group,  "classes", 20);
      int total = cfgFile->optionFindInt(group,  "num", 1);
      int num = total;
      char *a = cfgFile->getStr(group, "mask", 0);
      int *mask = parse_yolo_mask(a, &num);
      int max_boxes =cfgFile->optionFindIntQuiet(group,"max", 200);
      YoloLayer *layer =new$ YoloLayer(params->batch, params->w, params->h, num, total, mask, classes,max_boxes);

      if (layer->outputs != params->inputs) {
         a_error("Error: l.outputs == params.inputs, filters= in the [convolutional]-layer doesn't \
               correspond to classes= or mask= in [yolo]-layer");
      }
      layer->show_details = cfgFile->optionFindIntQuiet(group, "show_details", 1);
      layer->max_delta = cfgFile->optionFindFloatQuiet(group, "max_delta", FLT_MAX);   // set 10
      char *cpc = cfgFile->getStr(group, "counters_per_class", 0);
      layer->classes_multipliers = get_classes_multipliers(cpc, classes, layer->max_delta);

      layer->label_smooth_eps = cfgFile->optionFindFloatQuiet(group, "label_smooth_eps", 0.0f);
      layer->scale_x_y = cfgFile->optionFindFloatQuiet(group,"scale_x_y", 1);
      layer->objectness_smooth = cfgFile->optionFindIntQuiet(group,"objectness_smooth", 0);
      layer->new_coords = cfgFile->optionFindIntQuiet(group, "new_coords", 0);
      layer->iou_normalizer = cfgFile->optionFindFloatQuiet(group, "iou_normalizer", 0.75);
      layer->obj_normalizer = cfgFile->optionFindFloatQuiet(group,"obj_normalizer", 1);
      layer->cls_normalizer = cfgFile->optionFindFloatQuiet(group, "cls_normalizer", 1);
      layer->delta_normalizer =cfgFile->optionFindFloatQuiet(group, "delta_normalizer", 1);
      char *iou_loss = cfgFile->getStr(group, "iou_loss", "mse");   //  "iou");

      if (strcmp(iou_loss, "mse") == 0)
         layer->iou_loss = MSE;
      else if (strcmp(iou_loss, "giou") == 0)
         layer->iou_loss = GIOU;
      else if (strcmp(iou_loss, "diou") == 0)
         layer->iou_loss = DIOU;
      else if (strcmp(iou_loss, "ciou") == 0)
         layer->iou_loss = CIOU;
      else
         layer->iou_loss = IOU;
      fprintf(stderr, "[yolo] params: iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f,\
             cls_norm: %2.2f, delta_norm: %2.2f, scale_x_y: %2.2f\n",
            iou_loss, layer->iou_loss, layer->iou_normalizer, layer->obj_normalizer,
            layer->cls_normalizer, layer->delta_normalizer, layer->scale_x_y);

      char *iou_thresh_kind_str = cfgFile->getStr(group, "iou_thresh_kind", "iou");
      if (strcmp(iou_thresh_kind_str, "iou") == 0)
         layer->iou_thresh_kind = IOU;
      else if (strcmp(iou_thresh_kind_str, "giou") == 0)
         layer->iou_thresh_kind = GIOU;
      else if (strcmp(iou_thresh_kind_str, "diou") == 0)
         layer->iou_thresh_kind = DIOU;
      else if (strcmp(iou_thresh_kind_str, "ciou") == 0)
         layer->iou_thresh_kind = CIOU;
      else {
         fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
         layer->iou_thresh_kind = IOU;
      }

      layer->beta_nms =  cfgFile->optionFindFloatQuiet(group, "beta_nms", 0.6);
      char *nms_kind = cfgFile->getStr(group, "nms_kind", "default");
      if (strcmp(nms_kind, "default") == 0)
         layer->nms_kind = DEFAULT_NMS;
      else {
         if (strcmp(nms_kind, "greedynms") == 0)
            layer->nms_kind = GREEDY_NMS;
         else if (strcmp(nms_kind, "diounms") == 0)
            layer->nms_kind = DIOU_NMS;
         else
            layer->nms_kind = DEFAULT_NMS;
         printf("nms_kind: %s (%d), beta = %f \n", nms_kind, layer->nms_kind, layer->beta_nms);
      }

      layer->jitter = cfgFile->optionFindFloat(group, "jitter", .2);
      layer->resize =  cfgFile->optionFindFloatQuiet(group, "resize", 1.0);
      layer->focal_loss = cfgFile->optionFindIntQuiet(group, "focal_loss", 0);

      layer->ignore_thresh = cfgFile->optionFindFloat(group, "ignore_thresh", .5);
      layer->truth_thresh = cfgFile->optionFindFloat(group, "truth_thresh", 1);
      layer->iou_thresh =  cfgFile->optionFindFloatQuiet(group, "iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
      layer->random =  cfgFile->optionFindFloatQuiet(group,"random", 0);

      layer->track_history_size = cfgFile->optionFindIntQuiet(group, "track_history_size", 5);
      layer->sim_thresh =  cfgFile->optionFindFloatQuiet(group, "sim_thresh", 0.8);
      layer->dets_for_track = cfgFile->optionFindIntQuiet(group, "dets_for_track", 1);
      layer->dets_for_show = cfgFile->optionFindIntQuiet(group, "dets_for_show", 1);
      layer->track_ciou_norm =  cfgFile->optionFindFloatQuiet(group, "track_ciou_norm", 0.01);
      int embedding_layer_id = cfgFile->optionFindIntQuiet(group, "embedding_layer", 999999);
      if (embedding_layer_id < 0)
         embedding_layer_id = params->index + embedding_layer_id;
      if (embedding_layer_id != 999999) {
         printf(" embedding_layer_id = %d, ", embedding_layer_id);
         YoloLayer *le = (YoloLayer *)params->net->layers[embedding_layer_id];
         if(le->type!=LayerType.YOLO)
            a_error("未实现对非YOLO的处理:%s\n",NLayer.getLayerString(le->type));
         layer->embedding_layer_id = embedding_layer_id;
         layer->embedding_output = (float*)xcalloc(le.batch * le->outputs, sizeof(float));
         layer->embedding_size = le->anchorNumbers / layer->anchorNumbers;
         printf(" embedding_size = %d \n", layer->embedding_size);
         if (le->anchorNumbers % layer->anchorNumbers != 0) {
            printf(" Warning: filters=%d number in embedding_layer=%d isn't divisable by number of anchors %d \n",
                        le->anchorNumbers, embedding_layer_id, layer->anchorNumbers);
         }
      }

      char *map_file = cfgFile->getStr(group, "map", 0);
      if (map_file)
         layer->map = DnnUtils.readMap(map_file);

      a = cfgFile->getStr(group, "anchors", 0);
      if (a) {
         int len = strlen(a);
         int n = 1;
         int i;
         for (i = 0; i < len; ++i) {
            if (a[i] == '#')
               break;
            if (a[i] == ',')
               ++n;
         }
         float *biases=layer->biasData->getBias();
         for (i = 0; i < n && i < total*2; ++i) {
            float bias = atof(a);
            biases[i] = bias;
            a = strchr(a, ',') + 1;
         }
      }
      return layer;
   }


   MaxPoolLayer *createMaxPool(char *group, size_params *params){
      int stride = cfgFile->optionFindInt(group, "stride",1);
       int stride_x = cfgFile->optionFindIntQuiet(group, "stride_x", stride);
       int stride_y =cfgFile->optionFindIntQuiet(group, "stride_y", stride);
       int size = cfgFile->optionFindInt(group, "size",stride);
       int padding = cfgFile->optionFindIntQuiet(group, "padding", size-1);
       int maxpool_depth = cfgFile->optionFindIntQuiet(group, "maxpool_depth", 0);
       int out_channels =cfgFile->optionFindIntQuiet(group, "out_channels", 1);
       int antialiasing = cfgFile->optionFindIntQuiet(group, "antialiasing", 0);
       const int avgpool = 0;

       int batch,h,w,c;
       h = params->h;
       w = params->w;
       c = params->c;
       batch=params->batch;
       if(!(h && w && c))
          a_error("Layer before [maxpool] layer must output image.");
       MaxPoolLayer *layer;
       if(useMtcs)
          layer=new$  MtcsMaxPoolLayer(batch, h, w, c, size, stride_x, stride_y, padding,
                maxpool_depth, out_channels, antialiasing, avgpool, params->train);
       else
          layer = new$ MaxPoolLayer(batch, h, w, c, size, stride_x, stride_y, padding,
                maxpool_depth, out_channels, antialiasing, avgpool, params->train);
       layer->maxpool_zero_nonmax = cfgFile->optionFindIntQuiet(group, "maxpool_zero_nonmax", 0);
       return layer;
   }

   AvgPoolLayer *createAvgPool(char *group, size_params *params){
      int batch,w,h,c;
      w = params->w;
      h = params->h;
      c = params->c;
      batch=params->batch;
      if(!(h && w && c))
         a_error("Layer before avgpool layer must output image.");
      AvgPoolLayer *layer = new$ AvgPoolLayer(batch,w,h,c);
      if(useMtcs)
         layer->devType = DeviceType.MTCS;
      return layer;
   }

   SoftMaxLayer  *createSoftMax(char *group, size_params *params){
      int groups = cfgFile->optionFindInt(group, "groups",1);
      float temperature = cfgFile->optionFindFloat(group, "temperature",1);
      int spatial = cfgFile->optionFindInt(group, "spatial",0);
      int noloss = cfgFile->optionFindInt(group, "noloss",0);
      char *tree_file = cfgFile->getStr(group, "tree", 0);
      tree *softMaxTree =NULL;// read_tree(tree_file);
      int w = params->w;
      int h = params->h;
      int c = params->c;
      int batch=params->batch;
      SoftMaxLayer *layer = new$ SoftMaxLayer(batch,w,h,c,temperature,spatial,noloss,params->inputs,groups,softMaxTree);
      if(useMtcs)
           layer->devType = DeviceType.MTCS;
      return layer;
   }

   CostLayer *createCost(char *group, size_params *params){
       char *type_s = cfgFile->getStr(group,  "type", "sse");
       CostType type = CostLayer.getCostType/*!get_cost_type*/(type_s);
       float scale = cfgFile->optionFindFloatQuiet(group, "scale",1);
       float ratio = cfgFile->optionFindFloatQuiet(group, "ratio",0);
       CostLayer *layer = new$ CostLayer(params->batch, params->inputs, type, scale,ratio);
       return layer;
   }

   /**
   * 创建网络层。
   */
   NLayer *createLayer(LayerType type,char *group,size_params *params,int last_stop_backward){
      NLayer *layer=NULL;
      if(type == NLayer.LayerType.CONVOLUTIONAL){
         layer = createConvolutional(group,params);
      }else if(type == NLayer.LayerType.SHORTCUT){
         layer=createShortcut(group,params);
      }else if(type == NLayer.LayerType.ROUTE){
         layer=createRoute(group,params);
         RouteLayer *route=(RouteLayer *)layer;
         int k;
         for (k = 0; k < route->layerCount; ++k) {
            NLayer *layer=route->inputLayers[k];
            layer->use_bin_output = 0;
            if (params->index >= last_stop_backward)
               layer->keep_delta_gpu = 1;
         }
      }else if(type == NLayer.LayerType.UPSAMPLE){
         layer=createUpsample(group,params);
      }else if(type == NLayer.LayerType.YOLO){
         layer=createYolo(group,params);
         layer->keep_delta_gpu = 1;
      }else if(type == NLayer.LayerType.MAXPOOL){
         layer=createMaxPool(group,params);
      }else if(type==NLayer.LayerType.AVGPOOL){
         layer=createAvgPool(group,params);
      }else if(type==NLayer.LayerType.SOFTMAX){
         layer=createSoftMax(group,params);
      }else if(type==NLayer.LayerType.COST){
         layer=createCost(group,params);
      }else if(type == NLayer.LayerType.DROPOUT){
         layer=createDropoutLayer(group,params);
         //用上一层的输出数据和误差数据
         NLayer *up=params->net->layers[params->index-1];
         layer->outputData=up->outputData->ref();
         layer->deltaData = up->deltaData->ref();
         //printf("create dropout %p up:%p %d up->output:%p\n",layer,up,params->index-1,up->outputData);
      }else{
         a_error("还未处理的网络层类型:%d",type);
         return NULL;
      }
      layer->setNetwork((apointer)params->net);
      layer->truth = cfgFile->optionFindIntQuiet(group, "truth", 0);
      layer->onlyforward = cfgFile->optionFindIntQuiet(group, "onlyforward", 0);
      layer->stopbackward = cfgFile->optionFindIntQuiet(group, "stopbackward", 0);
      layer->dontsave = cfgFile->optionFindIntQuiet(group, "dontsave", 0);
      layer->dontload = cfgFile->optionFindIntQuiet(group, "dontload", 0);
      layer->numload = cfgFile->optionFindIntQuiet(group, "numload", 0);
      layer->dontloadscales = cfgFile->optionFindIntQuiet(group, "dontloadscales", 0);
      layer->learning_rate_scale = cfgFile->optionFindIntQuiet(group, "learning_rate", 1);
      layer->smooth = cfgFile->optionFindFloatQuiet(group, "smooth", 0);
      return layer;
   }

   /**
   * 从keyfile中解析参数并设到NNetwork中。
   */
   void parseNetOption(NNetwork *net){
      char *group="net";
      if(cfgFile->isFirstGroup("network"))
         group="network";

      net->max_batches = cfgFile->optionFindInt(group, "max_batches", 0);
      net->batch = cfgFile->optionFindInt(group, "batch",1);
      net->learning_rate = cfgFile->optionFindFloat(group, "learning_rate", .001);
      net->learning_rate_min = cfgFile->optionFindFloatQuiet(group, "learning_rate_min", .00001);
      net->batches_per_cycle = cfgFile->optionFindIntQuiet(group, "sgdr_cycle", net->max_batches);
      net->batches_cycle_mult = cfgFile->optionFindIntQuiet(group, "sgdr_mult", 2);
      net->momentum = cfgFile->optionFindFloat(group, "momentum", .9);
      net->decay = cfgFile->optionFindFloat(group, "decay", .0001);
      int subdivs = cfgFile->optionFindInt(group, "subdivisions",1);
      net->time_steps = cfgFile->optionFindIntQuiet(group, "time_steps",1);
      net->track = cfgFile->optionFindIntQuiet(group, "track", 0);
      net->augment_speed = cfgFile->optionFindIntQuiet(group, "augment_speed", 2);
      net->init_sequential_subdivisions = net->sequential_subdivisions
            = cfgFile->optionFindIntQuiet(group, "sequential_subdivisions", subdivs);
      if (net->sequential_subdivisions > subdivs)
         net->init_sequential_subdivisions = net->sequential_subdivisions = subdivs;
      net->try_fix_nan = cfgFile->optionFindIntQuiet(group, "try_fix_nan", 0);
      net->batch /= subdivs;          // mini_batch
      const int mini_batch = net->batch;
      net->batch *= net->time_steps;  // mini_batch * time_steps
      net->subdivisions = subdivs;    // number of mini_batches

      net->weights_reject_freq = cfgFile->optionFindIntQuiet(group, "weights_reject_freq", 0);
      net->equidistant_point = cfgFile->optionFindIntQuiet(group, "equidistant_point", 0);
      net->badlabels_rejection_percentage = cfgFile->optionFindFloatQuiet(group, "badlabels_rejection_percentage", 0);
      net->num_sigmas_reject_badlabels = cfgFile->optionFindFloatQuiet(group, "num_sigmas_reject_badlabels", 0);
      net->ema_alpha = cfgFile->optionFindFloatQuiet(group, "ema_alpha", 0);

      *net->badlabels_reject_threshold = 0;
      *net->delta_rolling_max = 0;
      *net->delta_rolling_avg = 0;
      *net->delta_rolling_std = 0;
      net->seen = 0;
      net->cur_iteration = 0;
      *net->cuda_graph_ready = 0;

      net->use_cuda_graph = cfgFile->optionFindIntQuiet(group, "use_cuda_graph", 0);
      net->loss_scale = cfgFile->optionFindFloatQuiet(group, "loss_scale", 1);
      net->dynamic_minibatch = cfgFile->optionFindIntQuiet(group, "dynamic_minibatch", 0);
      net->optimized_memory = cfgFile->optionFindIntQuiet(group, "optimized_memory", 0);
      // 1024 MB by default
      net->workspace_size_limit = (size_t)1024*1024 * cfgFile->optionFindFloatQuiet(group, "workspace_size_limit_MB", 1024);

      net->adam = cfgFile->optionFindIntQuiet(group, "adam", 0);
      if(net->adam){
         net->B1 = cfgFile->optionFindFloat(group, "B1", .9);
         net->B2 = cfgFile->optionFindFloat(group, "B2", .999);
         net->eps = cfgFile->optionFindFloat(group, "eps", .000001);
      }
      net->h = cfgFile->optionFindIntQuiet(group, "height",0);
      net->w = cfgFile->optionFindIntQuiet(group, "width",0);
      net->c = cfgFile->optionFindIntQuiet(group, "channels",0);
      net->inputs = cfgFile->optionFindIntQuiet(group, "inputs", net->h * net->w * net->c);
      net->max_crop = cfgFile->optionFindIntQuiet(group, "max_crop",net->w*2);
      net->min_crop = cfgFile->optionFindIntQuiet(group, "min_crop",net->w);
      net->flip = cfgFile->optionFindIntQuiet(group, "flip", 1);
      net->blur = cfgFile->optionFindIntQuiet(group, "blur", 0);
      net->gaussian_noise = cfgFile->optionFindIntQuiet(group, "gaussian_noise", 0);
      net->mixup = cfgFile->optionFindIntQuiet(group, "mixup", 0);
      int cutmix = cfgFile->optionFindIntQuiet(group, "cutmix", 0);
      int mosaic = cfgFile->optionFindIntQuiet(group, "mosaic", 0);
      if (mosaic && cutmix)
         net->mixup = 4;
      else if (cutmix)
         net->mixup = 2;
      else if (mosaic)
         net->mixup = 3;
      net->letter_box = cfgFile->optionFindIntQuiet(group, "letter_box", 0);
      net->mosaic_bound = cfgFile->optionFindIntQuiet(group, "mosaic_bound", 0);
      net->contrastive = cfgFile->optionFindIntQuiet(group, "contrastive", 0);
      net->contrastive_jit_flip = cfgFile->optionFindIntQuiet(group, "contrastive_jit_flip", 0);
      net->contrastive_color = cfgFile->optionFindIntQuiet(group, "contrastive_color", 0);
      net->unsupervised = cfgFile->optionFindIntQuiet(group, "unsupervised", 0);
      if (net->contrastive && mini_batch < 2) {
         a_error("Error: mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss!\n");
      }
      net->label_smooth_eps = cfgFile->optionFindFloatQuiet(group, "label_smooth_eps", 0.0f);
      net->resize_step = cfgFile->optionFindFloatQuiet(group, "resize_step", 32);
      net->attention = cfgFile->optionFindIntQuiet(group, "attention", 0);
      net->adversarial_lr = cfgFile->optionFindFloatQuiet(group, "adversarial_lr", 0);
      net->max_chart_loss = cfgFile->optionFindFloatQuiet(group, "max_chart_loss", 20.0);

      net->angle = cfgFile->optionFindFloatQuiet(group, "angle", 0);
      net->aspect = cfgFile->optionFindFloatQuiet(group, "aspect", 1);
      net->saturation = cfgFile->optionFindFloatQuiet(group, "saturation", 1);
      net->exposure = cfgFile->optionFindFloatQuiet(group, "exposure", 1);
      net->hue = cfgFile->optionFindFloatQuiet(group, "hue", 0);
      net->power = cfgFile->optionFindFloatQuiet(group, "power", 4);

      if(!net->inputs && !(net->h && net->w && net->c))
         a_error("No input parameters supplied");

      char *policy_s = cfgFile->getStr(group, "policy", "constant");
      net->policy = cfgFile->getPolicy/*!get_policy*/(policy_s);
      net->burn_in = cfgFile->optionFindIntQuiet(group, "burn_in", 0);
      if(net->policy == LearningRatePolicy.STEP){
         net->step = cfgFile->optionFindInt(group, "step", 1);
         net->scale = cfgFile->optionFindFloat(group, "scale", 1);
      } else if (net->policy == LearningRatePolicy.STEPS || net->policy == LearningRatePolicy.SGDR){
         char *l = cfgFile->getStr(group, "steps",0);
         char *p = cfgFile->getStr(group, "scales",0);
         char *s = cfgFile->getStr(group, "seq_scales",0);
         if(net->policy == LearningRatePolicy.STEPS && (!l || !p))
            a_error("STEPS policy must have steps and scales in cfg file");

         if (l) {
            int len = strlen(l);
            int n = 1;
            int i;
            for (i = 0; i < len; ++i) {
               if (l[i] == '#')
                  break;
               if (l[i] == ',')
                  ++n;
            }
            int* steps = (int*)xcalloc(n, sizeof(int));
            float* scales = (float*)xcalloc(n, sizeof(float));
            float* seq_scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
               float scale = 1.0;
               if (p) {
                  scale = atof(p);
                  p = strchr(p, ',') + 1;
               }
               float sequence_scale = 1.0;
               if (s) {
                  sequence_scale = atof(s);
                  s = strchr(s, ',') + 1;
               }
               int step = atoi(l);
               l = strchr(l, ',') + 1;
               steps[i] = step;
               scales[i] = scale;
               seq_scales[i] = sequence_scale;
            }
            net->scales = scales;
            net->steps = steps;
            net->seq_scales = seq_scales;
            net->num_steps = n;
         }
      } else if (net->policy == LearningRatePolicy.EXP){
         net->gamma = cfgFile->optionFindFloat(group, "gamma", 1);
      } else if (net->policy == LearningRatePolicy.SIG){
         net->gamma = cfgFile->optionFindFloat(group, "gamma", 1);
         net->step = cfgFile->optionFindInt(group, "step", 1);
      } else if (net->policy == LearningRatePolicy.POLY || net->policy == LearningRatePolicy.RANDOM){
         //net->power = cfgFile->optionFindFloat(group, "power", 1);
      }
   }


   /**
   * 根据配置文件创建网络
   */
   NNetwork *createNetwork(ConfigFile *cfgFile,int batch,int time_steps){
      if(!cfgFile){
         return NULL;
      }
      setConfigFile(cfgFile);
      asize groupCount=cfgFile->getGroupCount();
      NNetwork *net=new$ NNetwork(groupCount-1);
      net->useMtcs = useMtcs;
      if(!cfgFile->isNetWork()){
         a_error("First section must be [net] or [network]");
         return NULL;
      }
      parseNetOption(net);
      size_params params;
      if (batch > 0)
         params.train = 0;    // allocates memory for Inference only
      else
         params.train = 1;              // allocates memory for Inference & Training

      params.h = net->h;
      params.w = net->w;
      params.c = net->c;
      params.inputs = net->inputs;
      if (batch > 0)
         net->batch = batch;
      if (time_steps > 0)
         net->time_steps = time_steps;
      if (net->batch < 1)
         net->batch = 1;
      if (net->time_steps < 1)
         net->time_steps = 1;
      if (net->batch < net->time_steps)
         net->batch = net->time_steps;

      params.batch = net->batch;
      params.time_steps = net->time_steps;
      params.net = net;

      printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n",
            net->batch, net->batch * net->subdivisions, net->time_steps, params.train);

      int last_stop_backward = -1;
      int avg_outputs = 0;
      int avg_counter = 0;
      float bflops = 0;
      size_t workspace_size = 0;
      size_t max_inputs = 0;
      size_t max_outputs = 0;
      int receptive_w = 1, receptive_h = 1;
      int receptive_w_scale = 1, receptive_h_scale = 1;
      const int show_receptive_field =net->show_receptive_field/*!option_find_float_quiet(options, "show_receptive_field", 0)*/;

      // find l.stopbackward = cfgFile->optionFindIntQuiet(group, "stopbackward", 0);
      if (params.train == 1) {
         int gc=  cfgFile->getGroupCount();
         int i;
         for(i=1;i<gc;i++){
            char *value= cfgFile->getValue(i,"stopbackward");
            if(value!=NULL){
               int stopbackward=atoi(value);
               if(stopbackward==1){
                  last_stop_backward = i-1;
                  printf("last_stop_backward = %d \n", last_stop_backward);
               }
            }
         }
      }

      int old_params_train = params.train;
      char **groups=cfgFile->getGroups(&groupCount);
      int i=0;
      int count=0;
      NLayer ** layers=(NLayer **)net->layers;
      aint64 time=Time.monotonic();
      AError *error=NULL;
      //跳过组[net]
      fprintf(stderr, "layer     filters    size              input                output\n");
      for(i=1;i<groupCount;i++){
         params.index = count;
         char *type=groups[i];
         LayerType layerType = NLayer.getLayerType(type);
         fprintf(stderr, "%5d ", count);
         NLayer *layer=createLayer(layerType,type,&params,last_stop_backward);
         if(layer==NULL){
            a_error("创建层失败!");
         }
         layers[count++] = layer;
         layer->setOrderNumber(i-1);
         // calculate receptive field
         if(show_receptive_field){
            layer->setReceptive(&receptive_w,&receptive_h,&receptive_w_scale,&receptive_h_scale);
            int cur_receptive_w = receptive_w;
            int cur_receptive_h = receptive_h;
            fprintf(stderr, "%4d - receptive field: %d x %d \n", count, cur_receptive_w, cur_receptive_h);
         }

         layer->clip = cfgFile->optionFindFloatQuiet(type, "clip", 0);
         layer->dynamic_minibatch = net->dynamic_minibatch;
         layer->onlyforward = cfgFile->optionFindIntQuiet(type,"onlyforward", 0);
         layer->dont_update = cfgFile->optionFindIntQuiet(type, "dont_update", 0);
         layer->burnin_update = cfgFile->optionFindIntQuiet(type, "burnin_update", 0);
         layer->stopbackward = cfgFile->optionFindIntQuiet(type, "stopbackward", 0);
         layer->train_only_bn = cfgFile->optionFindIntQuiet(type, "train_only_bn", 0);
         layer->dontload = cfgFile->optionFindIntQuiet(type,"dontload", 0);
         layer->dontloadscales = cfgFile->optionFindIntQuiet(type, "dontloadscales", 0);
         layer->learning_rate_scale = cfgFile->optionFindFloatQuiet(type, "learning_rate", 1);

         //移走group。因为有重复的，所以必须移走keyfile中的当前group
         if(!cfgFile->removeGroup(type,&error)){
            if(error)
               a_error("移走组出错了:%d %s %s",i,type,error->message);
            else
               a_error("移走组出错了:%d %s",i,type);
         }


         if (layer->stopbackward == 1)
            printf(" ------- previous layers are frozen ------- \n");

         if (layer->workspace_size > workspace_size)
            workspace_size = layer->workspace_size;
         if (layer->inputs > max_inputs)
            max_inputs = layer->inputs;
         if (layer->outputs > max_outputs)
            max_outputs = layer->outputs;
         if (layer->isAntialiasing()) {
            NLayer *inLayer=layer->getInputLayer();
            params.h = inLayer->outputDimen.h;
            params.w = inLayer->outputDimen.w;
            params.c = inLayer->outputDimen.channels;
            params.inputs = inLayer->outputs;
         } else {
            params.h = layer->outputDimen.h;
            params.w = layer->outputDimen.w;
            params.c = layer->outputDimen.channels;
            params.inputs = layer->outputs;
         }

         if (layer->bflops > 0)
            bflops += layer->bflops;

         if (layer->inputDimen.w > 1 && layer->inputDimen.h  > 1) {
            avg_outputs += layer->outputs;
            avg_counter++;
         }

      }//end for
      aint64 time1=Time.monotonic();
      printf("创建网络所花时间 配置文件:%s time::%lli\n",cfgFile->getFileName(),time1-time);
      if (last_stop_backward > -1) {
         int k;
         for (k = 0; k < last_stop_backward; ++k) {
            NLayer *l = net->layers[k];
            if (l->keep_delta_gpu) {
               if (!l->deltaData) {
                  l->deltaData = DataFactory.getInstance()->createDeltaData(l->outputs,l->batch);
               }
            }
            l->onlyforward = 1;
            l->train = 0;
         }
      }

      setTrainOnlyBn(net); // set l.train_only_bn for all required layers
      // net.outputs = get_network_output_size(net);
      //  net.output = get_network_output(net);
      avg_outputs = avg_outputs / avg_counter;
      fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
      fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
      if (workspace_size) {
         if( net->useMtcs){
            net->workspace = MtcsMem.calloc(workspace_size*sizeof(float),TRUE);
         }else
            net->workspace = (float*)xcalloc(workspace_size, sizeof(float));
         printf("创建工作空间---- %d %p\n",workspace_size,net->workspace);
      }
      NLayer *last=net->getLastLayer();
      if ((net->w % 32 != 0 || net->h % 32 != 0) && (last->type ==LayerType.YOLO
      || last->type == LayerType.REGION || last->type == LayerType.DETECTION)) {
         printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
               net->w, net->h);
      }
      return net;
   }

   void setConfigFile(ConfigFile *cfg){
      if(cfgFile!=NULL){
         cfgFile->unref();
         cfgFile=NULL;
      }
      cfgFile=cfg->ref();
   }

   public$ NNetwork      *createNetwork(char *cfgFileName,int batch,int time_steps){
      ConfigFile *cfg =new$ ConfigFile(cfgFileName);
      return createNetwork(cfg,batch,time_steps);
   }

   public$ NNetwork   *createTrainNetwork(char *cfgFileName){
      ConfigFile *cfg =new$ ConfigFile(cfgFileName);
      return createNetwork(cfg,0,0);
   }

};

