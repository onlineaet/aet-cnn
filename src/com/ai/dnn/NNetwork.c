#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include <omp.h>

#include "NNetwork.h"
#include "DnnUtils.h"
#include "OutputLayer.h"
#include "ConvolutionalLayer.h"
#include "ShortcutLayer.h"
#include "YoloLayer.h"
#include "mtcs/MtcsGemm.h"


impl$ NNetwork{

   NNetwork(int layerCount){
      init(layerCount);
   }

   void init(int layerCount){
      self->layerCount = layerCount;
      self->layers=a_new(NLayer *,layerCount+1);
      self->seen = 0;
      self->cost = 0;
      self->truthData=NULL;
      self->classes=-1;
      self->cuda_graph_ready = (int*)xcalloc(1, sizeof(int));
      self->badlabels_reject_threshold = (float*)xcalloc(1, sizeof(float));
      self->delta_rolling_max = (float*)xcalloc(1, sizeof(float));
      self->delta_rolling_avg = (float*)xcalloc(1, sizeof(float));
      self->delta_rolling_std = (float*)xcalloc(1, sizeof(float));
   }

   NLayer *getOutputLayer(){
      int i;
      NLayer **layers=( NLayer **)self->layers;
      for(i = self->layerCount - 1; i >= 0; --i){
         if(layers[i]->type !=NLayer.LayerType.COST)
            break;
      }
      return layers[i];
   }

   void setInitInputsAndOutputs(aboolean train){
      NLayer *out = getOutputLayer();
      self->inputs = layers[0]->inputs;
      self->outputs = out->outputs;
      self->inputData=DataFactory.getInstance()->createInputData(w,h,c,batch);
      if(train)
        self->truthData=new$ TruthData(batch,classes,useMtcs);
   }

   //原型 get_network_output_size network.h network.c
   int getOuputSize(){
      NLayer *out = getOutputLayer();
      return out->outputs;
   }

   //原型 load_weights parser.h network.c
   void loadWeights(char *weightFile){
      if(!weightFile){
         a_error("无效的权重文件名。");
      }
      fprintf(stderr, "Loading weights from %s...", weightFile);
      fflush(stdout);
      FILE *fp = fopen(weightFile, "rb");
      if(!fp) {
         a_error("无效的权重文件名。%s",weightFile);
      }
      int cutoff = layerCount;
      int32_t major;
      int32_t minor;
      int32_t revision;
      fread(&major, sizeof(int32_t), 1, fp);
      fread(&minor, sizeof(int32_t), 1, fp);
      fread(&revision, sizeof(int32_t), 1, fp);
      if ((major * 10 + minor) >= 2) {
         printf("\n seen 64");
         auint64 iseen = 0;
         fread(&iseen, sizeof(auint64), 1, fp);
         self->seen = iseen;
      }

      self->cur_iteration = getCurrentBatch();
      printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(seen / 1000), (float)(seen / 64000));
      int transpose = (major > 1000) || (minor > 1000);
      int i;
      for(i = 0; i < layerCount && i < cutoff; ++i){
         NLayer *layer = layers[i];
         LayerType type=layer->type;
         if (layer->dontload)
            continue;
         if(type == LayerType.CONVOLUTIONAL){
            ConvolutionalLayer *cl=(ConvolutionalLayer *)layer;
            if(cl->share_layer==NULL){
               layer->loadWeights(fp);
            }
         }else if(type == LayerType.BATCHNORM){
            layer->loadWeights/*!load_batchnorm_weights*/(fp);
         }
         if (feof(fp))
            break;
      }
      fclose(fp);
   }

   //原型 save_weights_upto parser.h parser.c
   void saveWeightsUpto(char *filename, int cutoff, int save_ema){
      fprintf(stderr, "Saving weights to %s\n", filename);
      FILE *fp = fopen(filename, "wb");
      if(!fp) {
         fprintf(stderr, "Couldn't open file: %s\n", filename);
         exit(EXIT_FAILURE);
      }
      int32_t major = 0;
      int32_t minor = 2;
      int32_t revision = 5;
      fwrite(&major, sizeof(int32_t), 1, fp);
      fwrite(&minor, sizeof(int32_t), 1, fp);
      fwrite(&revision, sizeof(int32_t), 1, fp);
      seen= getCurrentIteration() * batch *subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
      fwrite(&seen, sizeof(auint64), 1, fp);

      int i;
      for(i = 0; i < layerCount && i < cutoff; ++i){
         NLayer *layer = layers[i];
         LayerType type=layer->type;
         if(type == LayerType.CONVOLUTIONAL){
            ConvolutionalLayer *cl=(ConvolutionalLayer *)layer;
            if(cl->share_layer==NULL){
               if (save_ema) {
                  layer->saveWeightsEma/*!save_convolutional_weights_ema*/(fp);
               }else {
                  layer->saveWeights/*!save_convolutional_weights*/(fp);
               }
            }
         }else if(type == LayerType.BATCHNORM){
            layer->saveWeights/*!save_convolutional_weights*/(fp);
         }
         fflush(fp);
      }
      fclose(fp);
   }

   void saveWeights(char *filename){
      saveWeightsUpto(filename, layerCount, 0);
   }

   //原型 int get_current_batch network.h network.c
   size_t getCurrentBatch(){
      size_t batch_num = (seen)/(batch*subdivisions);
      return batch_num;
   }

   int resize(int w, int h){
      int i;
      self->w = w;
      self->h = h;
      int inputs = 0;
      for (i = 0; i < self->layerCount; ++i){
         NLayer *l = self->layers[i];
         l->resize(w,h);
         inputs = l->outputs;
         w = l->outputDimen.w;
         h = l->outputDimen.h;
         if(l->type == LayerType.AVGPOOL)
            break;
      }
      NLayer *out = getOutputLayer();
      self->inputs = layers[0]->inputs;
      self->outputs = out->outputs;
      return 0;
   }

   void calcCost(){
      int i;
      float sum = 0;
      int count = 0;
      for(i = 0; i < self->layerCount; ++i){
         NLayer *layer= self->layers[i];
         if(layer varof$ OutputLayer){
            sum += ((OutputLayer *)layer)->getCost();
            ++count;
         }
      }
      self->cost = sum/count;
   }

   NLayer  *getLayer(int index){
      if(index<0 || index>=layerCount){
         a_warning("索引超出范围。index:%d 总层数:%d",index,layerCount);
         return NULL;
      }
      NLayer *ret= layers[index];
      return ret;
   }


   NLayer  *getNext(NLayer *at){
      if(at==NULL)
         return NULL;
      int i;
      for(i=0;i<layerCount;i++){
         if(layers[i]==at){
            return i>=0?layers[i+1]:NULL;
         }
      }
      return NULL;
   }

   float getCurrentRate(){
      size_t batch_num = getCurrentBatch();
      int i;
      float rate;
      if (batch_num < burn_in)
         return learning_rate * pow((float)batch_num / burn_in, power);
      switch (self->policy) {
         case LearningRatePolicy.CONSTANT:
            return learning_rate;
         case LearningRatePolicy.STEP:
            return learning_rate * pow(scale, batch_num/step);
         case LearningRatePolicy.STEPS:
            rate = learning_rate;
            for(i = 0; i < num_steps; ++i){
               if(steps[i] > batch_num)
                  return rate;
               rate *= scales[i];
            }
            return rate;
         case LearningRatePolicy.EXP:
            return learning_rate * pow(gamma, (int)batch_num);
         case LearningRatePolicy.POLY:
            return learning_rate * pow(1 - (float)batch_num / max_batches, power);
         case LearningRatePolicy.RANDOM:
            return learning_rate * pow(DnnUtils.randUniform(0,1), power);
         case LearningRatePolicy.SIG:
            return learning_rate * (1./(1.+exp(gamma*(batch_num - step))));
         default:
            a_warning("指定的学习方法%d不存在。返回learning_rate:%f\n",policy,learning_rate);
            return  learning_rate;
      }
   }

   NLayer  *getLastLayer(){
      return layers[layerCount-1];
   }

   int getLayerCount(){
      return layerCount;
   }

   public$ void setClasses(int classes){
      self->classes=classes;
   }

   public$ int  getClasses(){
      return classes;
   }

   private$ float lrelu(float src) {
      const float eps = 0.001;
      if (src > eps)
         return src;
      return eps;
   }

   //原型 fuse_conv_batchnorm darknet.h network.c
   void fuseConvBatchnorm(){
      int j;
      for (j = 0; j < layerCount; ++j) {
         NLayer *layer=layers[j];
         if (layer->type == LayerType.CONVOLUTIONAL) {
            ConvolutionalLayer *cl = (ConvolutionalLayer *)layer;
            if (cl->share_layer != NULL) {
               cl->batch_normalize = 0;
            }
            if (cl->batch_normalize) {
               cl->fuseConvBatchnorm();
               cl->freeBatchnorm/*!free_convolutional_batchnorm(l)*/();
               cl->batch_normalize = 0;
            }
         }else  if (layer->type == LayerType.SHORTCUT){
            ShortcutLayer  *sl = (ShortcutLayer *)layer;
            if(layer->weightData==NULL || layer->weightData->getWeights()==NULL || !sl->weights_normalization)
               continue;

            float *weights=layer->weightData->getWeights();
            if (sl->nweights > 0) {
               int i;
               for (i = 0; i < sl->nweights; ++i)
                  printf(" w = %f,", weights[i]);
               printf(" l->nweights = %d, j = %d \n", sl->nweights, j);
            }

            // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
            const int layer_step = sl->nweights / (sl->n + 1);    // 1 or l.c or (l.c * l.h * l.w)

            int chan, i;
            for (chan = 0; chan < layer_step; ++chan){
               float sum = 1, max_val = -FLT_MAX;
               if (sl->weights_normalization == SOFTMAX_NORMALIZATION) {
                  for (i = 0; i < (sl->n + 1); ++i) {
                     int w_index = chan + i * layer_step;
                     float w = weights[w_index];
                     if (max_val < w)
                        max_val = w;
                  }
               }

               const float eps = 0.0001;
               sum = eps;

               for (i = 0; i < (sl->n + 1); ++i) {
                  int w_index = chan + i * layer_step;
                  float w = weights[w_index];
                  if (sl->weights_normalization == RELU_NORMALIZATION)
                     sum += lrelu(w);
                  else if (sl->weights_normalization == SOFTMAX_NORMALIZATION)
                     sum += expf(w - max_val);
               }

               for (i = 0; i < (sl->n + 1); ++i) {
                  int w_index = chan + i * layer_step;
                  float w = weights[w_index];
                  if (sl->weights_normalization == RELU_NORMALIZATION)
                     w = lrelu(w) / sum;
                  else if (sl->weights_normalization == SOFTMAX_NORMALIZATION)
                     w = expf(w - max_val) / sum;
                  weights[w_index] = w;
               }
            }

            sl->weights_normalization = NO_NORMALIZATION;

         #ifdef GPU
            if (gpu_index >= 0) {
               push_shortcut_layer(*l);
            }
         #endif
         }else {
            printf(" Fusion skip layer type: %s \n", NLayer.getLayerString(layer->type));
         }
      }
   }

   //原型 calculate_binary_weights darknet.h network.c
   void calculateBinaryWeights(){
      int j;
      for (j = 0; j < layerCount; ++j) {
         NLayer *layer=layers[j];

         if (layer->type == LayerType.CONVOLUTIONAL) {
            //printf(" Merges Convolutional-%d and batch_norm \n", j);
            ConvolutionalLayer *cl = (ConvolutionalLayer *)layer;
            if (cl->xnor) {
               cl->binaryAlignWeights/*!binary_align_weights(l);*/();
               if (layers[j]->use_bin_output) {
                  cl->activation = ActivationType.LINEAR;
               }
               // fuse conv_xnor + shortcut -> conv_xnor
               if ((j + 1) < layerCount) {
                  NLayer *sc=layers[j+1];
                  if (sc->type == LayerType.SHORTCUT && sc->inputDimen.w == sc->outputDimen.w
                        && sc->inputDimen.h == sc->outputDimen.h && sc->inputDimen.channels == sc->outputDimen.channels){
                     //cl->bin_conv_shortcut_in = net.layers[net.layers[j + 1].index].output_gpu;
                     //cl->bin_conv_shortcut_out = net.layers[j + 1].output_gpu;
                     sc->type = LayerType.BLANK;
                     //sc->forward_gpu = forward_blank_layer;
                  }
               }
            }
         }
      }
      //printf("\n calculate_binary_weights Done! \n");
   }

   //原型 forward_network network.h network.cc
   void forward(NetworkState state){
      state.workspace = workspace;
      int i;
      for(i = 0; i < layerCount; ++i){
         state.index = i;
         NLayer *layer = layers[i];
         if(layer->deltaData && state.train /*!引起loss不收敛&& layer->train*/){
            //printf("进入 forward中的初始化 delta数据 i:%d layer:%p %s type:%d\n",i,layer,NLayer.getLayerString(layer->type),layer->type);
            layer->initDelta/*!scal_cpu(l.outputs * l.batch, 0, l.delta, 1)*/();
         }
         layer->forward(state);
         state.input = (InputData *)layer->getOutputData();
         //printf("前向传播结束---%d %s\n",i,NLayer.getLayerString(layer->type));
       }
   }


   //原型 get_network_output network.h network.cc
   OutputData *getOutput() {
      int i;
      for(i = layerCount-1; i > 0; --i)
         if(layers[i]->type != LayerType.COST)
            break;
      OutputData *output=layers[i]->outputData;
      return output;
   }


   //原型 network_predict darknet.h network.c
   OutputData *predict(InputData *input){
      NetworkState state={0};
      state.index = 0;
      state.input = input;
      state.truth = 0;
      state.train = 0;
      state.delta = 0;
      
      forward/*!forward_network*/(state);

      OutputData *out = getOutput/*!get_network_output(net);*/();
      return out;
   }

   public$ OutputData *predict(NImage *im){
      inputData->setImageData(im->imgData);
      return predict(inputData);
   }

   //原型 num_detections network.c
   private$ int numDetections(float thresh){
      int i;
      int s = 0;
      for (i = 0; i < layerCount; ++i) {
         NLayer *l = layers[i];
         if (l->type ==LayerType.YOLO) {
            YoloLayer *yl=(YoloLayer*)l;
            s += yl->numDetections/*!yolo_num_detections*/(thresh);
           // printf("NNetwork numDetections ---i:%d s:%d\n",i,s);
         }
         if (l->type == LayerType.GAUSSIAN_YOLO) {
            a_error("不未实现 GAUSSIAN_YOLO!");//s += gaussian_yolo_num_detections(l, thresh);
         }
         if (l->type == LayerType.DETECTION || l->type == LayerType.REGION) {
            a_error("不未实现 DETECTION REGION!");// s += l.w*l.h*l.n;
         }
      }
      return s;
   }

   //原型 make_network_boxes darknet.h network.c
   Detection **makeBoxes(float thresh, int *num){
      int i;
      NLayer *last=getLastLayer();
      for (i = 0; i < layerCount; ++i) {
         NLayer *tmp = layers[i];
         if (tmp->type == LayerType.YOLO || tmp->type == LayerType.GAUSSIAN_YOLO
         || tmp->type == LayerType.DETECTION || tmp->type == LayerType.REGION) {
            last = tmp;
            break;
         }
      }

      if(!(last varof$ ClassesIface)){
         a_error("层没有classes变量。",NLayer.getLayerString(last->type));
      }
      int classes=((ClassesIface*)last)->getClasses();
      int coords=((ClassesIface*)last)->getCoords();
      float *embeddingOutput=((ClassesIface*)last)->getEmbeddingOutput();
      int embeddingSize=((ClassesIface*)last)->getEmbeddingSize();

      int nboxes = numDetections/*!num_detections*/(thresh);
      printf("make_network_boxes 第几层:%d nboxes:%d classes:%d coords:%d\n",i,nboxes,classes,coords);

      if (num)
         *num = nboxes;
      Detection **dets = (Detection **)xcalloc(nboxes, sizeof(Detection *));
      for (i = 0; i < nboxes; ++i) {
         Detection *d=new$ Detection();
         d->prob = (float*)xcalloc(classes, sizeof(float));
         // tx,ty,tw,th uncertainty
         if(last->type == LayerType.GAUSSIAN_YOLO)
            d->uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
         else
            d->uc = NULL;
         if (coords > 4)
            d->mask = (float*)xcalloc(coords - 4, sizeof(float));
         else
            d->mask = NULL;

         if(embeddingOutput)
            d->embeddings = (float*)xcalloc(embeddingSize, sizeof(float));
         else
            d->embeddings = NULL;
         d->embedding_size = embeddingSize;
         dets[i]=d;
      }
      return dets;
   }

   //原型 fill_network_boxes network.c
   private$ void fillBoxes(int w, int h, float thresh, float hier,
      int *map, int relative,Detection **dets, int letter){
      int prev_classes = -1;
      int j;
      for (j = 0; j < layerCount; ++j) {
         NLayer *l = layers[j];
         if (l->type == LayerType.YOLO) {
            YoloLayer *yl=(YoloLayer *)l;
            int count = yl->getDetections/*!get_yolo_detections*/(w, h, self->w, self->h, thresh, map, relative, dets, letter);
            dets += count;
            if (prev_classes < 0)
               prev_classes = yl->getClasses();
            else if (prev_classes != yl->getClasses()) {
               printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
               prev_classes, yl->getClasses());
            }
         }
         if (l->type == LayerType.GAUSSIAN_YOLO) {
            // int count = get_gaussian_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
            //dets += count;
            a_error("不未实现 l.type == GAUSSIAN_YOLO");
         }
         if (l->type == LayerType.REGION) {
            // custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
            //dets += l.w*l.h*l.n;
            a_error("不未实现 l.type == REGION");

         }
         if (l->type == LayerType.DETECTION) {
            a_error("不未实现 l.type == DETECTION");

            //get_detection_detections(l, w, h, thresh, dets);
            // dets += l.w*l.h*l.n;
         }
      }
   }

   //原型 get_network_boxes darknet.h network.c
   Detection **getBoxes(int w, int h, float thresh, float hier,
         int *map, int relative, int *num, int letter){
      printf("NNetwork getBoxes w:%d h:%d thresh:%f hier:%f relative:%d letter:%d\n",w,h,thresh,hier,relative,letter);
      Detection **dets = makeBoxes(thresh, num);
      fillBoxes/*!fill_network_boxes*/(w, h, thresh, hier, map, relative, dets, letter);
      return dets;
   }

   // JSON format:
   //{
   // "frame_id":8990,
   // "objects":[
   //  {"class_id":4, "name":"aeroplane", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.793070},
   //  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.265497}
   // ]
   //},
   //原型 detection_to_json darknet.h network.c
   char *detectionToJson(Detection **dets, int nboxes, int classes, char **names, long long int frame_id, char *filename){
      const float thresh = 0.005; // function get_network_boxes() has already filtred dets by actual threshold

      char *send_buf = (char *)calloc(1024, sizeof(char));
      if (!send_buf)
         return 0;
      if (filename) {
         sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"filename\":\"%s\", \n \"objects\": [ \n", frame_id, filename);
      }else {
         sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"objects\": [ \n", frame_id);
      }

      int i, j;
      int class_id = -1;
      for (i = 0; i < nboxes; ++i) {
         for (j = 0; j < classes; ++j) {
            int show = strncmp(names[j], "dont_show", 9);
            if (dets[i]->prob[j] > thresh && show){
               if (class_id != -1)
                  strcat(send_buf, ", \n");
               class_id = j;
               char *buf = (char *)calloc(2048, sizeof(char));
               if (!buf)
                  return 0;
               //sprintf(buf, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f}",
               //    image_id, j, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]);

               sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, \"width\":%f, \"height\":%f}, \"confidence\":%f}",
               j, names[j], dets[i]->bbox.x, dets[i]->bbox.y, dets[i]->bbox.w, dets[i]->bbox.h, dets[i]->prob[j]);

               int send_buf_len = strlen(send_buf);
               int buf_len = strlen(buf);
               int total_len = send_buf_len + buf_len + 100;
               send_buf = (char *)realloc(send_buf, total_len * sizeof(char));
               if (!send_buf) {
                  if (buf)
                     free(buf);
                  return 0;
               }
               strcat(send_buf, buf);
               free(buf);
            }
         }
      }
      strcat(send_buf, "\n ] \n}");
      return send_buf;
   }

   //原型 get_current_iteration network.h network.c
   aint64 getCurrentIteration(){
      return cur_iteration;
   }

   //原型 get_network_cost network.h network.c
   float getNetworkCost(){
      int i;
      float sum = 0;
      int count = 0;
      for(i = 0; i < self->layerCount; ++i){
         NLayer *layer= self->layers[i];
         if(layer varof$ OutputLayer){
            sum += ((OutputLayer *)layer)->getCost();
            ++count;
         }
      }
      float cost = sum/count;
      return cost;
   }

   //原型 backward_network network.h network.cc
   void backward(NetworkState state){
      int i;
      InputData *original_input = state.input;
      DeltaData  *original_delta =state.delta;
      state.workspace = self->workspace;
      for(i = layerCount-1; i >= 0; --i){
         state.index = i;
         NLayer *prev=NULL;
         if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
         }else{
            prev = layers[i-1];
            IOData *ioData=(IOData*) prev->outputData;
            state.input =(InputData *)ioData;
            state.delta = prev->deltaData;
         }
         NLayer *l = layers[i];
         //printf("backward 开始反向传播----00 i:%d %s prev:%s\n",
              // i,NLayer.getLayerString(l->type),prev?NLayer.getLayerString(prev->type):"NULL");

         if (l->stopbackward)
            break;
         if (l->onlyforward)
            continue;
         l->backward(state);
      }
   }

   //原型 get_sequence_value network.h network.c
   int getSequenceValue(){
       int sequence = 1;
       if (sequential_subdivisions != 0)
          sequence = subdivisions / sequential_subdivisions;
       if (sequence < 1)
          sequence = 1;
       return sequence;
   }

   //原型 update_network network.h network.c
   void updateCPU(){
      int i;
      int update_batch = batch*subdivisions;
      float rate =getCurrentRate();
      for(i = 0; i < layerCount; ++i){
         NLayer *l = layers[i];
         if (l->train == 0)
            continue;
         //           if(l.update){
         //               l.update(l, update_batch, rate, net.momentum, net.decay);
         //           }
         l->update(update_batch, rate, momentum, decay);
      }
   }

   void updateMTCS(){
       const int iteration_num = (seen) / (batch * subdivisions);
       int i;
       int update_batch = batch*subdivisions * getSequenceValue()/*!get_sequence_value(net)*/;
       float rate = getCurrentRate/*!get_current_rate*/();
      // printf("update_network_gpu 00 iteration_num:%d update_batch:%d rate:%f\n",iteration_num,update_batch,rate);
       for(i = 0; i < layerCount; ++i){
           NLayer *l = layers[i];
           if (l->train == 0)
              continue;

           l->t = getCurrentBatch/*!get_current_batch*/();
//           printf("update_network_gpu 11 t:%d max_batches:%d burnin_update:%d burn_in:%d train_only_bn:%d\n",
//                 l->t,max_batches,l->burnin_update,burn_in,l->train_only_bn);
           if (iteration_num > (max_batches * 1 / 2))
              l->deform = 0;
           if (l->burnin_update && (l->burnin_update*burn_in > iteration_num))
              continue;
           if (l->train_only_bn)
              continue;

           if(l->dont_update < iteration_num){
             // printf("update_network_gpu 22 i:%d ub:%d rate:%f momentum:%f decay:%f lossscale:%f\n",
               //i,update_batch,rate,momentum,decay, loss_scale);
               l->update(update_batch, rate, momentum, decay, loss_scale);
           }
       }
   }

   void update(){
      if(useMtcs)
         updateMTCS();
      else
         updateCPU();
   }


   //原型 is_ema_initialized network.h network.c
   int isEmaInitialized(){
      int i;
      for (i = 0; i < layerCount; ++i) {
         NLayer *l =layers[i];
         if (l->type == LayerType.CONVOLUTIONAL) {
            int k;
            WeightData *weights=l->weightData;
            int size=weights->getSize();
            if (weights->ema/*!l.weights_ema*/) {
               for (k = 0; k < size/*!l.nweights*/; ++k) {
                  if (weights->ema/*!l.weights_ema*/[k] != 0)
                     return 1;
               }
            }
         }
      }
      return 0;
   }

   //原型 train_network_datum network.h network.c
   float train(InputData *input,TruthData *truthData) {
      NetworkState state={0};
      seen += batch;
      state.index = 0;
      state.input = input;
      state.delta = 0;
      state.truth = truthData->ref();
      state.train = 1;
      forward(state);
     // printf("train --- 进入反向---\n");
      backward(state);
      //printf("train --- 进入反向--sss-\n");

      float error =getNetworkCost();/*!get_network_cost(net);*/
      //if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
      //if(*(state.net.total_bbox) > 0)
      //  fprintf(stderr, " total_bbox = %d, rewritten_bbox = %f %% \n", *(state.net.total_bbox), 100 * (float)*(state.net.rewritten_bbox) / *(state.net.total_bbox));
      return error;
   }


   //原型 train_network_waitkey network.h network.c
   float train(ImageData *data,int waitKey){
      int i;
      float sum = 0;
      for(i = 0; i < subdivisions; ++i){
         //取图片数据存入inputData中
         data->getNextBatch/*!get_next_batch*/(batch, i*batch, inputData, truthData);
         current_subdivision = i;
         float err = train/*!train_network_datum*/(inputData, truthData);
         sum += err;
      }

      cur_iteration += 1;
      update();/*!update_network(net);*/

      int ema_start_point = max_batches / 2;

      if (ema_alpha && cur_iteration >= ema_start_point){
         int ema_period = (max_batches - ema_start_point - 1000) * (1.0 -ema_alpha);
         int ema_apply_point = max_batches - 1000;

         if (!isEmaInitialized()/*!is_ema_initialized(net)*/){
            // ema_update(net, 0); // init EMA
            printf(" EMA initialization \n");
         }

         if (cur_iteration== ema_apply_point){
            //ema_apply(net); // apply EMA (BN rolling mean/var recalculation is required)
            printf(" ema_apply() \n");
         }else
            if (cur_iteration < ema_apply_point){// && (*net.cur_iteration) % ema_period == 0)
               // ema_update(net, ema_alpha); // update EMA
               printf(" ema_update(), ema_alpha = %f \n",ema_alpha);
            }
      }


      int reject_stop_point = max_batches*3/4;
      if ((cur_iteration) < reject_stop_point &&
         weights_reject_freq && cur_iteration % weights_reject_freq == 0){
         float sim_threshold = 0.4;
         //reject_similar_weights(net, sim_threshold);
      }
      return (float)sum/(subdivisions*batch);
   }

   //原型 train_network network.h network.c
   float train(ImageData *data){
      int imgCount=data->getImagCount();
      a_assert(imgCount % batch == 0 && imgCount/batch==subdivisions);
      //printf("train start batch:%d subdivisions %d\n",batch,subdivisions);
      return train/*!train_network_waitkey*/(data, 0);
   }

   public$ aboolean tryFixNan(){
      return try_fix_nan;
   }

   //对立
   public$ aboolean  adversarial(){
      return adversarial;
   }

   public$ int getWorkspaceSize(){
      int i;
      size_t ws=0;
      for(i=0;i<layerCount;i++){
         if(layers[i]->workspace_size>ws)
            ws = layers[i]->workspace_size;
      }
      return ws;
   }


};
