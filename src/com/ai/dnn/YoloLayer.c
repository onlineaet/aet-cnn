#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include "YoloLayer.h"
#include "DnnUtils.h"
#include "NNetwork.h"
#include "NDetection.h"
#include "Box.h"

typedef struct train_yolo_args {
    NetworkState state;
    float tot_iou;
    float tot_giou_loss;
    float tot_iou_loss;
    int count;
    int class_count;
} train_yolo_args;




//https://blog.csdn.net/u013798145/article/details/115907594 yolo层forward_yolo_layer函数详解
//https://blog.51cto.com/u_15279692/3412056
//https://blog.csdn.net/Zivid_Liu/article/details/123256184 损失函数的说明
//目标检测中的损失函数与正负样本分配：YOLO v1, v2, v3，v3-spp-ultralytics
//https://blog.csdn.net/kill2013110/article/details/125545685?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-125545685-blog-123380975.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-125545685-blog-123380975.pc_relevant_aa&utm_relevant_index=5
//https://blog.csdn.net/weixin_39128022/article/details/105201250 YOLOv3损失函数个人见解
//https://zhuanlan.zhihu.com/p/114370969 反向传播推导与卷积公式
//https://blog.csdn.net/weixin_45209433/article/details/107974655 //yolov 计算误差δ的公式推导
//https://www.cnblogs.com/pinard/p/10750718.html  机器学习中的矩阵向量求导(一) 求导定义与求导布局
//https://www.cnblogs.com/pinard/p/6494810.html 已知池化层的δ^l，推导上一隐藏层的δ^l−1
//https://zhuanlan.zhihu.com/p/45190898?utm_source=wechat_timeline 神经网络，BP算法的理解与推导

impl$ YoloLayer{

   /**
   * self->c = number*(classes + 4 + 1)和上一个卷积输出的filter是一致的
   * 例如 number=3 classes=1 self->c就等于18
   * 上一层的filter一定也是18,这样输入出输的大小才能一样。
   * batch 一个batch中包含图片的张数
   * w 输入特征图的宽度
   * h 输入特征图的高度
   * anchorNumbers 一个cell预测多少个bbox
   * total total Anchor bbox的数目
   * mask 使用的是0,1,2 还是
   * classes 网络需要识别的物体类别数
   *
   * 给出下采样值，将图片划分为 M 行 N 列个小方框，以每个小方框的中心生成多个形状的锚框。
   * 将原始图片划分成m×n个区域，假设原始图片高度H=640, 宽度W=480，如果我们选择小块区域的尺寸为32×32，即下采样值为32，则m和n分别为：
   * m= 640/32 ​=20                  n= 480/32 =15
   */
   YoloLayer(int batch, int w, int h, int anchorNumbers, int total, int *mask, int classes){
      int i;
      self->type =LayerType.YOLO;
      self->anchorNumbers = anchorNumbers;//当前使用的先验框个数
      self->anchorsCount=total;//总的先验框个数
      //self->total = total;
      self->batch = batch;
      //printf("一个cell预测多少个bbox n:%d batch:%d total:%d classes:%d w:%d h:%d\n",anchorNumbers,batch,total,classes,w,h);
      // forward_yolo_layer  n:3 batch:4 total:9 classes:1 w:15 h:15
      int channels = anchorNumbers*(classes + 4 + 1);
      setInputDimen(w,h,channels);
      setOutputDimen(w,h,channels);

      self->classes = classes;
      self->cost = 0;
      //存储bbox的Anchor box的[w,h]
      //self->biases = calloc(total*2, sizeof(float));
      //存储bbox的Anchor box的[w,h]的更新值
      //self->bias_updates = calloc(anchorNumbers*2, sizeof(float));
      if(mask)
         self->mask = mask;
      else{
         self->mask = calloc(anchorNumbers, sizeof(int));
         for(i = 0; i < anchorNumbers; ++i){
            self->mask[i] = i;
         }
      }
      //一张训练图片经过yolo层后得到的输出元素个数（等于网格数*每个网格预测的矩形框数*每个矩形框的参数个数）
      self->outputs = h*w*anchorNumbers*(classes + 4 + 1);
      self->inputs = self->outputs;
      self->truths = 90*(4 + 1);
      setDefaultAnchors();
      fprintf(stderr, "yolo\n");
      srand(0);
      outputData=DataFactory.getInstance()->createOutputData(w,h,channels,batch);
      //yolo层误差项(包含整个batch的)
      deltaData=DataFactory.getInstance()->createDeltaData(w,h,channels,batch);
      self->maxBoxes=90;
      self->map=NULL;
   }

   YoloLayer(int batch, int w, int h, int anchorNumbers, int total, int *mask, int classes, int max_boxes){
      int i;
      self->type =LayerType.YOLO;
      lock=new$ AMutex();
      self->anchorNumbers = anchorNumbers;//当前使用的先验框个数
      self->anchorsCount=total;//总的先验框个数
      //self->total = total;
      self->batch = batch;
      int channels = anchorNumbers*(classes + 4 + 1);
      setInputDimen(w,h,channels);
      setOutputDimen(w,h,channels);

      self->classes = classes;
      self->cost = 0;//(float*)xcalloc(1, sizeof(float));
      //l.biases = (float*)xcalloc(total * 2, sizeof(float));
      self->biasData = DataFactory.getInstance()->createBiasData(total * 2);
      //l.nbiases = total * 2;
      if(mask)
         self->mask = mask;
      else{
         self->mask = (int*)xcalloc(anchorNumbers, sizeof(int));
         for(i = 0; i < anchorNumbers; ++i){
            self->mask[i] = i;
         }
      }
      //l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
      self->biasData->createUpdates();
      self->outputs = h*w*anchorNumbers*(classes + 4 + 1);
      self->inputs = self->outputs;
      self->max_boxes = max_boxes;
      self->truth_size = 4 + 2;
      self->truths = max_boxes*truth_size;    // 90*(4 + 1);
      int len=batch *w*h*anchorNumbers;
      self->labels = (int*)xcalloc(len, sizeof(int));
      for (i = 0; i <len; ++i)
         self->labels[i] = -1;
      self->class_ids = (int*)xcalloc(len, sizeof(int));
      for (i = 0; i < len; ++i)
         self->class_ids[i] = -1;

      //l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
      deltaData=DataFactory.getInstance()->createDeltaData(w,h,channels,batch);
      // l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
      outputData=DataFactory.getInstance()->createOutputData(w,h,channels,batch);

      /*!
      for(i = 0; i < total*2; ++i){
      l.biases[i] = .5;
      }
      */
      self->biasData->setBiasValue(.5);

      #ifdef GPU
      l.forward_gpu = forward_yolo_layer_gpu;
      l.backward_gpu = backward_yolo_layer_gpu;
      l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
      l.output_avg_gpu = cuda_make_array(l.output, batch*l.outputs);
      l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

      free(l.output);
      if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped))
         l.output_pinned = 1;
      else {
         cudaGetLastError(); // reset CUDA-error
         l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
      }

      free(l.delta);
      if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped))
         l.delta_pinned = 1;
      else {
         cudaGetLastError(); // reset CUDA-error
         l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
      }
      #endif

      fprintf(stderr, "yolo\n");
      srand(time(0));
   }

   /*
   * 获取预测框的位置和大小
   * 公式如下:
   * bx=σ(tx)+cx
   * by=σ(ty)+cy
   * bw=pw*e^tw
   * bw=ph*e^th
   * cx,cy为feature map中的坐标，feature map大小为32x32，则(cx,cy)为(0,0)到(31,31)1024个点，pw,ph为cfg文件中anchors尺寸
   * σ(tx)来自x,即该层的输出数据outputData,在forward方法中outputData已经用公式LOGISTIC激活了。
   * 公式中没有除以lw,lh,w,h。这里除以lw,lh,w,h是为了把数据变成[0,1]。
   * 参数如下:
   * outputData  yolo_layer的输出，包含所有batch预测得到的矩形框信息
   * anchors 表示Anchor框的w,h，从cfg文件yolo层配置取出的。anchors是一个数组
   * index 取outputData的索引号
   * n     取anchors的索引号
   * i 第几行（yolo_layer维度为l.out_w*l.out_c）
   * j 第几列
   * lw 特征图的宽度
   * lh 特征图的高度
   * w 输入图像的宽度
   * h 输入图像的高度
   * stride 不同的特征图具有不同的步长(即是两个grid cell之间跨的像素个数不同)
   * 详见:
   * https://blog.csdn.net/lxk2017/article/details/90606973?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-90606973-blog-84109828.pc_relevant_multi_platform_whitelistv6&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-90606973-blog-84109828.pc_relevant_multi_platform_whitelistv6&utm_relevant_index=1
   */

   Box getBox(float *outputData, float *anchors, int n, int index, int i, int j,
      int lw, int lh, int w, int h, int stride){
      Box b=new$ Box();
      b.x = (i + outputData[index + 0*stride]) / lw;
      b.y = (j + outputData[index + 1*stride]) / lh;
      b.w = exp(outputData[index + 2*stride]) * anchors[2*n]   / w;
      b.h = exp(outputData[index + 3*stride]) * anchors[2*n+1] / h;
      return b;
   }

   float delta_yolo_box(Box *truth, float *x, float *biases, int n, int index, int i, int j,
      int lw, int lh, int w, int h, float *delta, float scale, int stride){
      Box pred = getBox(x, biases, n, index, i, j, lw, lh, w, h, stride);
      float iou = pred.iou(truth);
      float tx = (truth->x*lw - i);
      float ty = (truth->y*lh - j);
      float tw = log(truth->w*w / biases[2*n]);
      float th = log(truth->h*h / biases[2*n + 1]);
      delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
      delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
      delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
      delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
      return iou;
   }


   //原型 delta_yolo_class yolo_layer.c
   void deltaClass(float *output, float *delta, int index, int class_id,
         int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps,
         float *classes_multipliers, float cls_normalizer){
      int n;
      if (delta[index + stride*class_id]){
         float y_true = 1;
         if(label_smooth_eps)
            y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
         float result_delta = y_true - output[index + stride*class_id];
         if(!isnan(result_delta) && !isinf(result_delta))
            delta[index + stride*class_id] = result_delta;
         //delta[index + stride*class_id] = 1 - output[index + stride*class_id];

         if (classes_multipliers)
            delta[index + stride*class_id] *= classes_multipliers[class_id];
         if(avg_cat) *avg_cat += output[index + stride*class_id];
            return;
      }
      // Focal loss
      if (focal_loss) {
         // Focal Loss
         float alpha = 0.5;    // 0.25 or 0.5
         //float gamma = 2;    // hardcoded in many places of the grad-formula

         int ti = index + stride*class_id;
         float pt = output[ti] + 0.000000000000001F;
         // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
         float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
         //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

         for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
         }
      }else {
         // default
         for (n = 0; n < classes; ++n) {
            float y_true = ((n == class_id) ? 1 : 0);
            if (label_smooth_eps)
               y_true = y_true *  (1 - label_smooth_eps) + 0.5*label_smooth_eps;
            float result_delta = y_true - output[index + stride*n];
            if (!isnan(result_delta) && !isinf(result_delta))
               delta[index + stride*n] = result_delta;

            if (classes_multipliers && n == class_id)
               delta[index + stride*class_id] *= classes_multipliers[class_id] * cls_normalizer;
            if (n == class_id && avg_cat)
               *avg_cat += output[index + stride*n];
         }
      }
   }


   int entryIndex(int b ,int  location, int entry){
       int w = inputDimen.w;
       int h = inputDimen.h;
       int n =   location / (w*h);
       int loc = location % (w*h);
       return b*outputs + n*w*h*(4+classes+1) + entry*w*h + loc;
   }

   // Converts output of the network to detection boxes
   // w,h: image width,height
   // netw,neth: network width,height
   // relative: 1 (all callers seems to pass TRUE)
   //原型 correct_yolo_boxes yolo_layer.h yolo_layer.c
   void correctBoxes(Detection **dets, int n, int w, int h, int netw, int neth, int relative, int letter){
      int i;
      // network height (or width)
      int new_w = 0;
      // network height (or width)
      int new_h = 0;
      // Compute scale given image w,h vs network w,h
      // I think this "rotates" the image to match network to input image w/h ratio
      // new_h and new_w are really just network width and height
      if (letter) {
         if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
         } else {
            new_h = neth;
            new_w = (w * neth) / h;
         }
      } else {
         new_w = netw;
         new_h = neth;
      }
      // difference between network width and "rotated" width
      float deltaw = netw - new_w;
      // difference between network height and "rotated" height
      float deltah = neth - new_h;
      // ratio between rotated network width and network width
      float ratiow = (float)new_w / netw;
      // ratio between rotated network width and network width
      float ratioh = (float)new_h / neth;
      for (i = 0; i < n; ++i) {

         Box b = dets[i]->bbox;
         // x = ( x - (deltaw/2)/netw ) / ratiow;
         //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
         b.x = (b.x - deltaw / 2. / netw) / ratiow;
         b.y = (b.y - deltah / 2. / neth) / ratioh;
         // scale to match rotation of incoming image
         b.w *= 1 / ratiow;
         b.h *= 1 / ratioh;

         // relative seems to always be == 1, I don't think we hit this condition, ever.
         if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
         }

         dets[i]->bbox = b;
      }
   }

   void correct_yolo_boxes(NDetection *dets, int n, int w, int h, int netw, int neth, int relative){
      int i;
      int new_w=0;
      int new_h=0;
      if (((float)netw/w) < ((float)neth/h)) {
         new_w = netw;
         new_h = (h * netw)/w;
      } else {
         new_h = neth;
         new_w = (w * neth)/h;
      }
      for (i = 0; i < n; ++i){
         Box b = dets[i].bbox;
         b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
         b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
         b.w *= (float)netw/new_w;
         b.h *= (float)neth/new_h;
         if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
         }
         dets[i].bbox = b;
      }
   }

   //原型 yolo_num_detections yolo_layer.h yolo_layer.c
   int numDetections(float thresh){
      int w=inputDimen.w;
      int h=inputDimen.h;
      int inputWH=w*h;
      int i, n;
      int count = 0;
      float *output=outputData->getDataArray();
      for(n = 0; n < anchorNumbers; ++n){
         for (i = 0; i < inputWH; ++i) {
            int obj_index  = entryIndex(0, n*inputWH + i, 4);
            if(output[obj_index] > thresh){
               ++count;
            }
         }
      }
      return count;
   }

   void avg_flipped_yolo(){
      int i,j,n,z;
      //float *flip = self->output + self->outputs;
      float *flip=outputData->getData(1);
      int w=inputDimen.w;
      int h=inputDimen.h;
      for (j = 0; j < h; ++j) {
         for (i = 0; i < w/2; ++i) {
            for (n = 0; n < anchorNumbers; ++n) {
               for(z = 0; z < self->classes + 4 + 1; ++z){
                  int i1 = z*w*h*anchorNumbers + n*w*h + j*w + i;
                  int i2 = z*w*h*anchorNumbers + n*w*h + j*w + (w - i - 1);
                  float swap = flip[i1];
                  flip[i1] = flip[i2];
                  flip[i2] = swap;
                  if(z == 0){
                     flip[i1] = -flip[i1];
                     flip[i2] = -flip[i2];
                  }
               }
            }
         }
      }
      float *output=outputData->getData(0);
      for(i = 0; i < self->outputs; ++i){
         output[i] = (output[i] + flip[i])/2.;
      }
   }


   //原型 get_yolo_box yolo_layer.c
   Box getYoloBox(float *x, float *biases, int n, int index, int i, int j,
            int lw, int lh, int w, int h, int stride, int new_coords){
      Box b=new$ Box();
      // ln - natural logarithm (base = e)
      // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
      // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
      // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
      // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
      if (new_coords) {
         b.x = (i + x[index + 0 * stride]) / lw;
         b.y = (j + x[index + 1 * stride]) / lh;
         b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
         b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
      } else {
         b.x = (i + x[index + 0 * stride]) / lw;
         b.y = (j + x[index + 1 * stride]) / lh;
         b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
         b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
      }
      return b;
   }

   //原型 get_yolo_detections yolo_layer.h yolo_layer.c
   int getDetections(int w, int h, int netw, int neth, float thresh,
                  int *map, int relative, Detection **dets, int letter){
      //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
      int i,j,n;
      float *predictions =outputData->getDataArray();// self->output;
      int lw=inputDimen.w;
      int lh=inputDimen.h;
      float *bias=biasData->getBias();
      // This snippet below is not necessary
      // Need to comment it in order to batch processing >= 2 images
      //if (l.batch == 2) avg_flipped_yolo(l);
      int count = 0;
      for (i = 0; i < lw*lh; ++i){
         int row = i / lw;
         int col = i % lw;
         for(n = 0; n < anchorNumbers; ++n){
            int obj_index  = entryIndex(0, n*lw*lh + i, 4);
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
               //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
               int box_index = entryIndex(0, n*lw*lh + i, 0);
               dets[count]->bbox = getYoloBox/*!get_yolo_box*/(predictions, bias, mask[n],
                        box_index, col, row, lw, lh, netw, neth, lw*lh, new_coords);
               dets[count]->objectness = objectness;
               dets[count]->classes = self->classes;
               if (embedding_output) {
                  DnnUtils.getEmbedding/*!get_embedding*/(embedding_output, lw, lh, anchorNumbers*embedding_size, embedding_size,
                        col, row, n, 0, dets[count]->embeddings);
               }

               for (j = 0; j < classes; ++j) {
                  int class_index = entryIndex(0, n*lw*lh + i, 4 + 1 + j);
                  float prob = objectness*predictions[class_index];
                  dets[count]->prob[j] = (prob > thresh) ? prob : 0;
               }
               ++count;
            }
         }
      }
      correctBoxes/*!correct_yolo_boxes*/(dets, count, w, h, netw, neth, relative, letter);
      return count;
   }

   void printOutput(){
      int b=0;
      int n=0;
      int size=inputDimen.w*inputDimen.h*anchorNumbers*(classes + 4 + 1);
      for (b = 0; b < self->batch; ++b){
         float *item=outputData->getData(b);//大小是w*h*number*(classes+4+1) item有w*h个网格
         for(n = 0; n < size; ++n){
            printf("output===batch:%d n:%d %f\n",batch,n,item[n]);
         }
      }
   }

   /**
   * 获取预测框与实际标注框的交集和并集的比值
   * 返回最大的IOU
   */
   static float getBestIOU(int b,Box *pred,int *index){
      NNetwork *net=(NNetwork *)network;
      int boxCount=net->truthData->getBoxCount(b);
      Box truth=new$ Box();
      int i;
      float best_iou = 0;
      int best_t = 0;
      for(i = 0; i < boxCount; ++i){
         float *boxData= net->truthData->getData(b,i);//boxData 见BoxLabel的encode方法。
         truth.refresh(boxData);
         //printf("truth box t:%d %f %f %f %f boxCount:%d\n",i,truth.x,truth.y,truth.w,truth.h,boxCount);
         float iou = pred->iou(&truth);
         if (iou > best_iou) {
            best_iou = iou;
            best_t = i;
         }
      }
      *index=best_t;
      return best_iou;
   }


   int getMaskIndex(int val){
      int i;
      for(i = 0; i < anchorNumbers; ++i){
         if(self->mask[i] == val)
            return i;
      }
      return -1;
   }

   /**
   * 误差公式: δ=∂C/∂z
   * 类别用的损失函数是交叉熵。
   * 所以:-δ=t-σ
   */
   void deltaClass(int b, int index, int class, float *avg_cat) {
      float *output=outputData->getData(b);
      float *delta=deltaData->getData(b);
      int classes=self->classes;
      int stride=inputDimen.w*inputDimen.h;
      int n;
      if (delta[index]){
         delta[index + stride*class] = 1 - output[index + stride*class];
         if(avg_cat)
            *avg_cat += output[index + stride*class];
         return;
      }
      //当n == class即label=1时,delta=1-output[index + stride*n]
      for(n = 0; n < classes; ++n){
         delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
         if(n == class && avg_cat)
            *avg_cat += output[index + stride*n];
      }
   }


   /**
   * 复制误差到上一层的误差
   */
   void backward(NetworkState state){
      //已知本层误差，计算上一层的误差
      int i,j;
      for(i=0;i<batch;++i){
         float *delta=deltaData->getData(i);
         float *stateDelta=state.delta->getData(i);
         for(j=0;j<inputs;++j){
            stateDelta[j]+=delta[j];
         }
      }
   }

   void resize(int w, int h){
      NNetwork *net=(NNetwork *)network;
      setInputDimen(w,h);
      self->outputs = h*w*anchorNumbers*(self->classes + 4 + 1);
      self->inputs = self->outputs;
      outputData->resize( w,h);
      deltaData->resize( w,h);
   }

   void setDefaultAnchors(){
      int i;
      for(i = 0; i < anchorsCount*2; ++i){
         self->anchors[i] = .5;
      }
   }

   /**
   * 设置先验框
   */
   void setAnchors(float *data,int len){
      int i;
      for(i = 0; i < len; ++i){
         self->anchors[i] =data[i];
      }
   }

   void setMaxBoxes(int max){
      maxBoxes=max;
   }

   void setIgnoreThresh(float thresh){
      ignoreThresh=thresh;
   }

   void setTruthThresh(float thresh){
      truthThresh=thresh;
   }

   void setMap(int *map){
      if(self->map){
         free(self->map);
         self->map=NULL;
      }
      self->map=map;
   }

   /**
   * 实现输出层的接口
   */
   float   getCost(){
      return cost;
   }

   /**
   * 实现接口ClassesIface的方法
   * getClasses
   */
   int  getClasses(){
      return self->classes;
   }

   int  getCoords(){
      return 0;
   }

   float *getEmbeddingOutput(){
      return embedding_output;
   }

   int  getEmbeddingSize(){
      return embedding_size;
   }

   /**
   * 用LOGISTIC函数激活输出数据x，y,Confidence(置信度),classes,但不激活w,h。
   * 输出数据初始值等于输入数据。
   * 输出数据的存储格式是:
   *  output是float指针数组，数组元素是每张图片的数据。数据组成如下
   *  因此batch只能为0；并假设l.out_w=l.out_h=2,l.classes=2）：
   *  如果 w=2 h=2 classes=2 就有w*h=4个grid 每个grid有3个锚框，
   *  每个锚框的数据是grid*x+grid*y+grid*w+grid*h+grid*(1个目标)+grid*(2分类)
   *  第一个锚框存储如下
   *  xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2 4个x 4个y 4个置信度 因为有两个classes所以有4*classes1+4*classes2=8个分类
   *  第二个锚框存储如下
   *  xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2，
   *  第三个锚框存储如下
   *  ......同上
   */


   //原型 scal_add_cpu blas.h blas.c
   void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX){
       int i;
       for (i = 0; i < N; ++i)
          X[i*INCX] = X[i*INCX] * ALPHA + BETA;
   }


   //    1.每一个目标都只有一个正样本，max-iou matching策略，匹配规则为IOU最大（没有阈值），选取出来的即为正样本；
   //    2.IOU<0.2（人为设定阈值）的作为负样本；
   //    3.除了正负样本，其余的全部为忽略样本
   //    比如drbox与gtbox的IOU最大为0.9，设置IOU小于0.2的为负样本。
   //    那么有一个IOU为0.8的box，那么这个box就是忽略样本，有一个box的IOU为0.1，那么就是负样本；同样的drbox与gtbox的IOU最大为0.4，那么它也是正样本。
   //    4.正anchor用于分类和回归的学习，正负anchor用于置信度confidence的学习，忽略样本不考虑。
   // https://blog.csdn.net/weixin_39128022/article/details/105201250 求误差的公式

   /**https://www.bilibili.com/video/av247479444/?ivk_sa=1024320u 《高等数学》（下册）第六讲 方向导数与梯度（数学一、数学二）
   * 把输入数据拷到输出数据 层的输入数据参见NNetwork.c中的forward方法。
   * 这里输入数据是卷积层的输出数据（图的特征数据 feature map）
   * 每张图上有number个(4+1+classes)
   * 1.
   */
   //原型 forward_yolo_layer yolo_layer.h yolo_layer.c
   void forward(NetworkState state){
      //int i, j, b, t, n;
      NNetwork *net=(NNetwork *)network;
      InputData *inputData = state.input;
      ((IOData *)inputData)->checkNan("检查inputData---",getOrderNumber());
      //把inputdata数据复制到outputdata中
      ((IOData *)inputData)->copy((IOData *)outputData);/*!memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));*/
      int b, n;
      int w=inputDimen.w;
      int h =inputDimen.h;
      float *output= outputData->getDataArray();
      for (b = 0; b < batch; ++b) {
         for (n = 0; n < anchorNumbers/*!l.n*/; ++n) {
            int bbox_index = entryIndex(b, n*w*h, 0);
            if (new_coords) {
               ;//activate_array(l.output + bbox_index, 4 * l.w*l.h, LOGISTIC);    // x,y,w,h
            }else {
               Activation.activate(output + bbox_index, 2 *w*h, ActivationType.LOGISTIC);        // x,y,
               int obj_index = entryIndex(b, n*w*h, 4);
               Activation.activate(output  + obj_index, (1 + classes)*w*h, ActivationType.LOGISTIC);
            }
            scal_add_cpu(2 * w*h, scale_x_y, -0.5*(scale_x_y - 1), output + bbox_index, 1);    // scale x,y
         }
      }

      // delta is zeroed
      /*!memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
      if (!state.train) return;
      */
      ((IOData*)outputData)->checkNan("检查激活后的output",getOrderNumber());
      //误差数据初始为0
      deltaData->setZero();
      if(!state.train)
         return;

      int i;
      int len=batch*inputDimen.w*inputDimen.h*anchorNumbers;
      for (i = 0; i < len; ++i)
         labels[i] = -1;
      for (i = 0; i < len; ++i)
         class_ids[i] = -1;
      //float avg_iou = 0;
      float tot_iou = 0;
      float tot_giou = 0;
      float tot_diou = 0;
      float tot_ciou = 0;
      float tot_iou_loss = 0;
      float tot_giou_loss = 0;
      float tot_diou_loss = 0;
      float tot_ciou_loss = 0;
      float recall = 0;
      float recall75 = 0;
      float avg_cat = 0;
      float avg_obj = 0;
      float avg_anyobj = 0;
      int count = 0;
      int class_count = 0;
      /*!*(l.cost) = 0;*/
      self->cost = 0;

      /*!
      int num_threads = l.batch;
      pthread_t* threads = (pthread_t*)calloc(num_threads, sizeof(pthread_t));

      struct train_yolo_args* yolo_args = (train_yolo_args*)xcalloc(batch, sizeof(struct train_yolo_args));

      for (b = 0; b < batch; b++){
      yolo_args[b].l = (NLayer *)self;
      yolo_args[b].state = state;
      yolo_args[b].b = b;

      yolo_args[b].tot_iou = 0;
      yolo_args[b].tot_iou_loss = 0;
      yolo_args[b].tot_giou_loss = 0;
      yolo_args[b].count = 0;
      yolo_args[b].class_count = 0;

      if (pthread_create(&threads[b], 0, process_batch, &(yolo_args[b])))
      a_error("Thread creation failed", );
      }

      for (b = 0; b < l.batch; b++)
      {
      pthread_join(threads[b], 0);

      tot_iou += yolo_args[b].tot_iou;
      tot_iou_loss += yolo_args[b].tot_iou_loss;
      tot_giou_loss += yolo_args[b].tot_giou_loss;
      count += yolo_args[b].count;
      class_count += yolo_args[b].class_count;
      }

      free(yolo_args);
      free(threads);
      */
      //fprintf(stderr,"yololayer forward 11\n");
      train_yolo_args totalArgs={state,0,0,0,0,0};
      int id=DstbCompute.getInstance()->addTask((ComputingUnit *)self,(apointer)&totalArgs);
      fprintf(stderr,"yololayer forward 22\n");
      DstbCompute.getInstance()->wait(id);
      tot_iou = totalArgs.tot_iou;
      tot_iou_loss = totalArgs.tot_iou_loss;
      tot_giou_loss = totalArgs.tot_giou_loss;
      count = totalArgs.count;
      class_count = totalArgs.class_count;

      // Search for an equidistant point from the distant boundaries of the local minimum
      int iteration_num =net->getCurrentIteration/*!get_current_iteration*/();
      const int start_point = net->max_batches * 3 / 4;
      //printf(" equidistant_point ep = %d, it = %d \n", state.net.equidistant_point, iteration_num);

      if ((net->badlabels_rejection_percentage && start_point < iteration_num) ||
      (net->num_sigmas_reject_badlabels && start_point < iteration_num) ||
      (net->equidistant_point && net->equidistant_point < iteration_num)){
         const float progress_it = iteration_num - net->equidistant_point;
         const float progress = progress_it / (net->max_batches - net->equidistant_point);
         float ep_loss_threshold = (*net->delta_rolling_avg) * progress * 1.4;

         float cur_max = 0;
         float cur_avg = 0;
         float counter = 0;
         for (i = 0; i < batch ; ++i) {
            int j;
            float *delta = deltaData->getData(i);
            for(j=0;j<outputs;j++){
               if (delta[i] != 0) {
                  counter++;
                  cur_avg += fabs(delta[j]);
                  if (cur_max < fabs(delta[j]))
                     cur_max = fabs(delta[j]);
               }
            }
         }

         cur_avg = cur_avg / counter;

         if (*net->delta_rolling_max == 0)
            *net->delta_rolling_max = cur_max;
         *net->delta_rolling_max = *net->delta_rolling_max * 0.99 + cur_max * 0.01;
         *net->delta_rolling_avg = *net->delta_rolling_avg * 0.99 + cur_avg * 0.01;

         // reject high loss to filter bad labels
         if (net->num_sigmas_reject_badlabels && start_point < iteration_num){
            const float rolling_std = (*net->delta_rolling_std);
            const float rolling_max = (*net->delta_rolling_max);
            const float rolling_avg = (*net->delta_rolling_avg);
            const float progress_badlabels = (float)(iteration_num - start_point) / (start_point);

            float cur_std = 0;
            float counter = 0;
            for (i = 0; i < batch ; ++i) {
               int j;
               float *delta = deltaData->getData(i);
               for(j=0;j<outputs;++j){
                  if (delta[j] != 0) {
                     counter++;
                     cur_std += pow(delta[j] - rolling_avg, 2);
                  }
               }
            }
            cur_std = sqrt(cur_std / counter);

            *net->delta_rolling_std = *net->delta_rolling_std * 0.99 + cur_std * 0.01;

            float final_badlebels_threshold = rolling_avg + rolling_std * net->num_sigmas_reject_badlabels;
            float badlabels_threshold = rolling_max - progress_badlabels * fabs(rolling_max - final_badlebels_threshold);
            badlabels_threshold = max_val_cmp(final_badlebels_threshold, badlabels_threshold);
            for (i = 0; i < batch ; ++i) {
               int j;
               float *delta = deltaData->getData(i);
               for(j=0;j<outputs;++j){
                  if (fabs(delta[j]) > badlabels_threshold)
                     delta[j] = 0;
               }
               printf(" rolling_std = %f, rolling_max = %f, rolling_avg = %f \n", rolling_std, rolling_max, rolling_avg);
               printf(" badlabels loss_threshold = %f, start_it = %d, progress = %f \n",badlabels_threshold, start_point, progress_badlabels *100);
               ep_loss_threshold = min_val_cmp(final_badlebels_threshold, rolling_avg) * progress;
            }
         }
         // reject some percent of the highest deltas to filter bad labels
         if (net->badlabels_rejection_percentage && start_point < iteration_num) {
            if (*net->badlabels_reject_threshold == 0)
               *net->badlabels_reject_threshold = *net->delta_rolling_max;

            printf(" badlabels_reject_threshold = %f \n", *net->badlabels_reject_threshold);

            const float num_deltas_per_anchor = (classes + 4 + 1);
            float counter_reject = 0;
            float counter_all = 0;
            for (i = 0; i < batch ; ++i) {
               int j;
               float *delta = deltaData->getData(i);
               for(j=0;j<outputs;++j){
                  if (delta[j] != 0) {
                     counter_all++;
                     if (fabs(delta[j]) > (*net->badlabels_reject_threshold)) {
                        counter_reject++;
                        delta[j] = 0;
                     }
                  }
               }
            }
            float cur_percent = 100 * (counter_reject*num_deltas_per_anchor / counter_all);
            if (cur_percent > net->badlabels_rejection_percentage) {
               *net->badlabels_reject_threshold += 0.01;
               printf(" increase!!! \n");
            }else if (*net->badlabels_reject_threshold > 0.01) {
               *net->badlabels_reject_threshold -= 0.01;
               printf(" decrease!!! \n");
            }

            printf(" badlabels_reject_threshold = %f, cur_percent = %f, badlabels_rejection_percentage = %f, delta_rolling_max = %f \n",
                     *net->badlabels_reject_threshold, cur_percent, net->badlabels_rejection_percentage, *net->delta_rolling_max);
         }//end  if (net->badlabels_rejection_percentage && start_point < iteration_num) {


         // reject low loss to find equidistant point
         if (net->equidistant_point && net->equidistant_point < iteration_num) {
            printf(" equidistant_point loss_threshold = %f, start_it = %d, progress = %3.1f %% \n",
                  ep_loss_threshold, net->equidistant_point, progress * 100);
            for (i = 0; i < batch ; ++i) {
               int j;
               float *delta = deltaData->getData(i);
               for(j=0;j<outputs;++j){
                  if (fabs(delta[j]) < ep_loss_threshold)
                     delta[j] = 0;
               }
            }
         }
      } //end if ((net->badlabels_rejection_percentage && start_point < iteration_num) ||

      if (count == 0)
         count = 1;
      if (class_count == 0)
         class_count = 1;

      if (show_details == 0) {
         float loss = pow(DnnUtils.magArray/*!mag_array(l.delta, l.outputs * l.batch)*/(deltaData->getDataArray(),outputs*batch), 2);
         self->cost = loss;
         loss /= batch;
         fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f),\
                count: %d, total_loss = %f \n",
               (iou_loss == MSE ? "mse" : (iou_loss == GIOU ? "giou" : "iou")),
               iou_normalizer,obj_normalizer, cls_normalizer, state.index, tot_iou / count, count, loss);
      }else {
         // show detailed output
         int lw=inputDimen.w;
         int lh=inputDimen.h;
         int stride = lw*lh;
         /*!
         float* no_iou_loss_delta = (float *)calloc(batch * outputs, sizeof(float));
         memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
         */
         DeltaData *no_iou_loss_delta=deltaData->clone();
         int j, n;
         for (b = 0; b < batch; ++b) {
            float *lossDelta=no_iou_loss_delta->getData(b);
            for (j = 0; j < lh; ++j) {
               for (i = 0; i < lw; ++i) {
                  for (n = 0; n < anchorNumbers; ++n) {
                     int index = entryIndex(b, n*lw*lh + j*lw + i, 0);
                     lossDelta[index + 0 * stride] = 0;
                     lossDelta[index + 1 * stride] = 0;
                     lossDelta[index + 2 * stride] = 0;
                     lossDelta[index + 3 * stride] = 0;
                  }
               }
            }
         }

         float classification_loss = obj_normalizer *
               pow(DnnUtils.magArray/*!mag_array(l.delta, l.outputs * l.batch)*/(no_iou_loss_delta->getDataArray(),outputs*batch), 2);
         no_iou_loss_delta->unref();
         float loss = pow(DnnUtils.magArray/*!mag_array(l.delta, l.outputs * l.batch)*/(deltaData->getDataArray(),outputs*batch), 2);
         float iou_loss = loss - classification_loss;

         float avg_iou_loss = 0;
         self->cost = loss;

         // gIOU loss + MSE (objectness) loss
         if (iou_loss == MSE) {
            self->cost = pow(DnnUtils.magArray/*!mag_array(l.delta, l.outputs * l.batch)*/(deltaData->getDataArray(),outputs*batch), 2);
         }else {
            // Always compute classification loss both for iou + cls loss and for logging with mse loss
            // TODO: remove IOU loss fields before computing MSE on class
            //   probably split into two arrays
            if (iou_loss == GIOU) {
               avg_iou_loss = count > 0 ? iou_normalizer * (tot_giou_loss / count) : 0;
            } else {
               avg_iou_loss = count > 0 ? iou_normalizer * (tot_iou_loss / count) : 0;
            }
            self->cost  = avg_iou_loss + classification_loss;
         }


         loss /= batch;
         classification_loss /= batch;
         iou_loss /= batch;

         fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f),\
                count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
         (iou_loss == MSE ? "mse" : (iou_loss == GIOU ? "giou" : "iou")), iou_normalizer, obj_normalizer,
         cls_normalizer, state.index, tot_iou / count, count, classification_loss, iou_loss, loss);

         //fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
         //    (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.obj_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
         //    classification_loss, iou_loss, loss);
      }
   }

   //原型 compare_yolo_class yolo_layer.c
   int compareClass(float *output, int classes, int class_index,
            int stride, float objectness, int class_id, float conf_thresh){
      int j;
      for (j = 0; j < classes; ++j) {
         //float prob = objectness * output[class_index + stride*j];
         float prob = output[class_index + stride*j];
         if (prob > conf_thresh)
            return 1;
      }
      return 0;
   }

   //原型 delta_yolo_box yolo_layer.c
   ious deltaBox(Box truth, float *x, float *biases, int n, int index,
            int i, int j, int lw, int lh, int w, int h, float *delta,
            float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss,
            int accumulate, float max_delta, int *rewritten_bbox, int new_coords){
      if (delta[index + 0 * stride] || delta[index + 1 * stride]
            || delta[index + 2 * stride] || delta[index + 3 * stride])
         (*rewritten_bbox)++;

      ious all_ious = { 0 };
      // i - step in layer width
      // j - step in layer height
      //  Returns a box in absolute coordinates
      Box pred = getYoloBox/*!get_yolo_box*/(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
      all_ious.iou = pred.iou/*!box_iou*/(&truth);
      all_ious.giou = Box.giou/*!box_giou*/(pred, truth);
      all_ious.diou = Box.diou/*!box_diou*/(pred, truth);
      all_ious.ciou =  Box.ciou/*box_ciou*/(pred, truth);
      // avoid nan in dx_box_iou
      if (pred.w == 0)
         pred.w = 1.0;
      if (pred.h == 0)
         pred.h = 1.0;
      if (iou_loss == MSE) {   // old loss
         float tx = (truth.x*lw - i);
         float ty = (truth.y*lh - j);
         float tw = log(truth.w*w / biases[2 * n]);
         float th = log(truth.h*h / biases[2 * n + 1]);

         if (new_coords) {
            //tx = (truth.x*lw - i + 0.5) / 2;
            //ty = (truth.y*lh - j + 0.5) / 2;
            tw = sqrt(truth.w*w / (4 * biases[2 * n]));
            th = sqrt(truth.h*h / (4 * biases[2 * n + 1]));
         }

         //printf(" tx = %f, ty = %f, tw = %f, th = %f \n", tx, ty, tw, th);
         //printf(" x = %f, y = %f, w = %f, h = %f \n", x[index + 0 * stride], x[index + 1 * stride], x[index + 2 * stride], x[index + 3 * stride]);

         // accumulate delta
         delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;
         delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
         delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
         delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
      }else {
         // https://github.com/generalized-iou/g-darknet
         // https://arxiv.org/abs/1902.09630v2
         // https://giou.stanford.edu/
         all_ious.dx_iou = Box.dxBoxIou/*!dx_box_iou*/(pred, truth, iou_loss);

         // jacobian^t (transpose)
         //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
         //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
         //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
         //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

         // jacobian^t (transpose)
         float dx = all_ious.dx_iou.dt;
         float dy = all_ious.dx_iou.db;
         float dw = all_ious.dx_iou.dl;
         float dh = all_ious.dx_iou.dr;


         // predict exponential, apply gradient of e^delta_t ONLY for w,h
         if (new_coords) {
            //dw *= 8 * x[index + 2 * stride];
            //dh *= 8 * x[index + 3 * stride];
            //dw *= 8 * x[index + 2 * stride] * biases[2 * n] / w;
            //dh *= 8 * x[index + 3 * stride] * biases[2 * n + 1] / h;

            //float grad_w = 8 * exp(-x[index + 2 * stride]) / pow(exp(-x[index + 2 * stride]) + 1, 3);
            //float grad_h = 8 * exp(-x[index + 3 * stride]) / pow(exp(-x[index + 3 * stride]) + 1, 3);
            //dw *= grad_w;
            //dh *= grad_h;
         } else {
            dw *= exp(x[index + 2 * stride]);
            dh *= exp(x[index + 3 * stride]);
         }
         //dw *= exp(x[index + 2 * stride]);
         //dh *= exp(x[index + 3 * stride]);

         // normalize iou weight
         dx *= iou_normalizer;
         dy *= iou_normalizer;
         dw *= iou_normalizer;
         dh *= iou_normalizer;

         dx = DnnUtils.fixNanInf/*!fix_nan_inf*/(dx);
         dy = DnnUtils.fixNanInf/*!fix_nan_inf*/(dy);
         dw = DnnUtils.fixNanInf/*!fix_nan_inf*/(dw);
         dh = DnnUtils.fixNanInf/*!fix_nan_inf*/(dh);

         if (max_delta != FLT_MAX) {
            dx = DnnUtils.clipValue/*!clip_value*/(dx, max_delta);
            dy = DnnUtils.clipValue/*!clip_value*/(dy, max_delta);
            dw = DnnUtils.clipValue/*!clip_value*/(dw, max_delta);
            dh = DnnUtils.clipValue/*!clip_value*/(dh, max_delta);
         }
         if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
         }
         // accumulate delta
         delta[index + 0 * stride] += dx;
         delta[index + 1 * stride] += dy;
         delta[index + 2 * stride] += dw;
         delta[index + 3 * stride] += dh;
      }

      return all_ious;
   }

   //原型 averages_yolo_deltas yolo_layer.c
   void averagesDeltas(int class_index, int box_index, int stride, int classes, float *delta){
      int classes_in_one_box = 0;
      int c;
      for (c = 0; c < classes; ++c) {
         if (delta[class_index + stride*c] > 0)
            classes_in_one_box++;
      }

      if (classes_in_one_box > 0) {
         delta[box_index + 0 * stride] /= classes_in_one_box;
         delta[box_index + 1 * stride] /= classes_in_one_box;
         delta[box_index + 2 * stride] /= classes_in_one_box;
         delta[box_index + 3 * stride] /= classes_in_one_box;
      }
   }

   train_yolo_args  processBatch(NetworkState state,int b){
      train_yolo_args args;
      memset(&args,0,sizeof(train_yolo_args));
      NNetwork *net=(NNetwork *)self->network;
      {

         int i, j, t, n;
         //printf(" b = %d \n", b, b);
         //float tot_iou = 0;
         float tot_giou = 0;
         float tot_diou = 0;
         float tot_ciou = 0;
         //float tot_iou_loss = 0;
         //float tot_giou_loss = 0;
         float tot_diou_loss = 0;
         float tot_ciou_loss = 0;
         float recall = 0;
         float recall75 = 0;
         float avg_cat = 0;
         float avg_obj = 0;
         float avg_anyobj = 0;
         //int count = 0;
         //int class_count = 0;
         int lh=inputDimen.h;
         int lw=inputDimen.w;
         int ln=anchorNumbers;
         float *output =outputData->getData(b);// self->output;
         float *delta = deltaData->getData(b);
         float *biases=biasData->getBias();

         for (j = 0; j < lh; ++j) {
            for (i = 0; i < lw; ++i) {
               for (n = 0; n < ln; ++n) {
                  const int class_index = entryIndex(b,n * lw * lh + j * lw + i, 4 + 1);
                  const int obj_index = entryIndex(b,n * lw * lh + j * lw + i, 4);
                  const int box_index = entryIndex(b,n * lw * lh + j * lw + i, 0);

                  const int stride = lw * lh;
                  Box pred = getYoloBox/*!get_yolo_box*/(output, biases,
                                 mask[n], box_index,i, j, lw, lh, net->w, net->h, lw * lh,new_coords);
                  float best_match_iou = 0;
                  int best_match_t = 0;
                  float best_iou = 0;
                  int best_t = 0;
                  for (t = 0; t < max_boxes; ++t) {
                     Box truth = Box.floatToBoxStride/*!float_to_box_stride*/(
                           (float*)state.truth->truths[b].values[t]/*!state.truth + t * truth_size + b * truths*/, 1);

                     if (!truth.x)
                        break;  // continue;
                     /*!int class_id = state.truth[t * truth_size + b * truths + 4];*/
                     int class_id = state.truth->truths[b].values[t][4];

                     if (class_id >= classes || class_id < 0) {
                        printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. \
                        In txt-labels class_id should be [from 0 to %d] \n", class_id, classes, classes - 1);
                        printf("\n truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n",
                        truth.x, truth.y, truth.w, truth.h, class_id);
                        // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value
                        continue;
                     }

                     float objectness = output[obj_index];
                     if (isnan(objectness) || isinf(objectness))
                        output[obj_index] = 0;
                     int class_id_match =compareClass/*!compare_yolo_class*/(output, classes,class_index, lw * lh, objectness, class_id, 0.25f);

                     float iou =pred.iou/*!box_iou(pred, truth);*/(&truth);
                     if (iou > best_match_iou && class_id_match == 1) {
                        best_match_iou = iou;
                        best_match_t = t;
                     }
                     if (iou > best_iou) {
                        best_iou = iou;
                        best_t = t;
                     }
                  }

                  avg_anyobj += output[obj_index];
                  delta[obj_index] = obj_normalizer * (0 - output[obj_index]);
                  if (best_match_iou > ignore_thresh) {
                     if (objectness_smooth) {
                        const float delta_obj = obj_normalizer * (best_match_iou - output[obj_index]);
                        if (delta_obj > delta[obj_index])
                           delta[obj_index] = delta_obj;
                     }else
                        delta[obj_index] = 0;
                  } else if (net->adversarial) {
                     int stride = lw * lh;
                     float scale = pred.w * pred.h;
                     if (scale > 0)
                        scale = sqrt(scale);
                     delta[obj_index] = scale * obj_normalizer * (0 - output[obj_index]);
                     int cl_id;
                     int found_object = 0;
                     for (cl_id = 0; cl_id < classes; ++cl_id) {
                        if (output[class_index + stride * cl_id] * output[obj_index] > 0.25) {
                           delta[class_index + stride * cl_id] = scale * (0 - output[class_index + stride * cl_id]);
                           found_object = 1;
                        }
                     }
                     if (found_object) {
                        // don't use this loop for adversarial attack drawing
                        for (cl_id = 0; cl_id < classes; ++cl_id)
                           if (output[class_index + stride * cl_id] * output[obj_index] < 0.25)
                              delta[class_index + stride * cl_id] = scale * (1 - output[class_index + stride * cl_id]);

                        delta[box_index + 0 * stride] += scale * (0 - output[box_index + 0 * stride]);
                        delta[box_index + 1 * stride] += scale * (0 - output[box_index + 1 * stride]);
                        delta[box_index + 2 * stride] += scale * (0 - output[box_index + 2 * stride]);
                        delta[box_index + 3 * stride] += scale * (0 - output[box_index + 3 * stride]);
                     }
                  }
                  if (best_iou > truth_thresh) {
                     const float iou_multiplier = best_iou * best_iou;// (best_iou - l.truth_thresh) / (1.0 - l.truth_thresh);
                     if (objectness_smooth)
                        delta[obj_index] = obj_normalizer * (iou_multiplier - output[obj_index]);
                     else
                        delta[obj_index] = obj_normalizer * (1 - output[obj_index]);
                     //l.delta[obj_index] = l.obj_normalizer * (1 - l.output[obj_index]);

                     /*!int class_id = state.truth[best_t * truth_size + b * truths + 4];*/
                     int class_id = state.truth->truths[b].values[best_t][4];

                     if (map)
                        class_id = map[class_id];
                     deltaClass/*!delta_yolo_class*/(output, delta, class_index, class_id,
                              classes, lw * lh, 0, focal_loss, label_smooth_eps, classes_multipliers, cls_normalizer);
                     const float class_multiplier = (classes_multipliers) ? classes_multipliers[class_id] : 1.0f;
                     if (objectness_smooth)
                        delta[class_index + stride * class_id] = class_multiplier *
                              (iou_multiplier - output[class_index + stride * class_id]);
                     Box truth = Box.floatToBoxStride/*!float_to_box_stride*/(
                                          (float*)state.truth->truths[b].values[best_t]/*!state.truth + best_t * truth_size + b * truths*/, 1);
                     deltaBox/*!delta_yolo_box*/(truth, output, biases, mask[n], box_index,
                                    i, j, lw, lh, net->w, net->h, delta,
                                    (2 - truth.w * truth.h), lw * lh, iou_normalizer * class_multiplier,
                                    iou_loss, 1, max_delta, net->rewritten_bbox, new_coords);
                                    (*net->total_bbox)++;
                  }
               }
            }
         }
         for (t = 0; t < max_boxes; ++t) {
            Box truth = Box.floatToBoxStride/*!float_to_box_stride*/(
                             (float*)state.truth->truths[b].values[t]/*!state.truth + t * truth_size + b * truths*/, 1);
            if (!truth.x)
               break;  // continue;
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
               char buff[256];
               printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
               sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list",
                           truth.x, truth.y, truth.w, truth.h);
               system(buff);
            }
            /*!int class_id = state.truth[t * truth_size + b * truths + 4];*/
            int class_id = state.truth->truths[b].values[t][4];

            if (class_id >= classes || class_id < 0)
               continue; // if label contains class_id more than number of classes in the cfg-file and class_id check garbage value

            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * lw);
            j = (truth.y * lh);
            Box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < anchorsCount; ++n) {
               Box pred = new$ Box();
               pred.w = biases[2 * n] / net->w;
               pred.h = biases[2 * n + 1] / net->h;
               float iou = pred.iou/*!box_iou*/(&truth_shift);
               if (iou > best_iou) {
                  best_iou = iou;
                  best_n = n;
               }
            }

            int mask_n = DnnUtils.intIndex/*!int_index*/(mask, best_n, anchorNumbers);
            if (mask_n >= 0) {
               /*!int class_id = state.truth[t * truth_size + b * truths + 4];*/
               int class_id = state.truth->truths[b].values[t][4];
               if (map)
                  class_id = map[class_id];

               int box_index = entryIndex(b,mask_n * lw * lh + j * lw + i, 0);
               const float class_multiplier = (classes_multipliers) ? classes_multipliers[class_id] : 1.0f;
               ious all_ious = deltaBox/*!delta_yolo_box*/(truth, output, biases, best_n, box_index,
                              i, j, lw, lh, net->w, net->h, delta, (2 - truth.w * truth.h),
                              lw * lh, iou_normalizer * class_multiplier, iou_loss, 1, max_delta, net->rewritten_bbox, new_coords);
               (*net->total_bbox)++;

               const int truth_in_index = t * truth_size + b * truths + 5;
               const int track_id = state.truth->truths[b].values[t][5];/*!state.truth[truth_in_index];*/
               const int truth_out_index = b * anchorNumbers* lw * lh + mask_n * lw * lh + j * lw + i;
               labels[truth_out_index] = track_id;
               class_ids[truth_out_index] = class_id;
               //printf(" track_id = %d, t = %d, b = %d, truth_in_index = %d, truth_out_index = %d \n", track_id, t, b, truth_in_index, truth_out_index);

               // range is 0 <= 1
               args.tot_iou += all_ious.iou;
               args.tot_iou_loss += 1 - all_ious.iou;
               // range is -1 <= giou <= 1
               tot_giou += all_ious.giou;
               args.tot_giou_loss += 1 - all_ious.giou;

               tot_diou += all_ious.diou;
               tot_diou_loss += 1 - all_ious.diou;

               tot_ciou += all_ious.ciou;
               tot_ciou_loss += 1 - all_ious.ciou;

               int obj_index = entryIndex(b,mask_n * lw * lh + j * lw + i, 4);
               avg_obj += output[obj_index];
               if (objectness_smooth) {
                  float delta_obj = class_multiplier * obj_normalizer * (1 - output[obj_index]);
                  if (delta[obj_index] == 0)
                     delta[obj_index] = delta_obj;
               } else
                  delta[obj_index] = class_multiplier * obj_normalizer * (1 - output[obj_index]);

               int class_index = entryIndex(b,mask_n * lw * lh + j * lw + i, 4 + 1);
               deltaClass/*!delta_yolo_class*/(output, delta, class_index, class_id, classes,
                           lw * lh, &avg_cat, focal_loss, label_smooth_eps, classes_multipliers, cls_normalizer);

               //printf(" label: class_id = %d, truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", class_id, truth.x, truth.y, truth.w, truth.h);
               //printf(" mask_n = %d, l.output[obj_index] = %f, l.output[class_index + class_id] = %f \n\n", mask_n, l.output[obj_index], l.output[class_index + class_id]);

               ++(args.count);
               ++(args.class_count);
               if (all_ious.iou > .5)
                  recall += 1;
               if (all_ious.iou > .75)
                  recall75 += 1;
            }

            // iou_thresh
            for (n = 0; n < anchorsCount; ++n) {
               int mask_n = DnnUtils.intIndex/*!int_index*/(mask, n, anchorNumbers);
               if (mask_n >= 0 && n != best_n && iou_thresh < 1.0f) {
                  Box pred = new$ Box();
                  pred.w = biases[2 * n] / net->w;
                  pred.h = biases[2 * n + 1] / net->h;
                  float iou = Box.iouKind/*!box_iou_kind*/(pred, truth_shift, iou_thresh_kind); // IOU, GIOU, MSE, DIOU, CIOU
                  // iou, n

                  if (iou > iou_thresh) {
                     /*!int class_id = state.truth[t * truth_size + b * truths + 4];*/
                     int class_id = state.truth->truths[b].values[t][4];
                     if (map)
                        class_id = map[class_id];

                     int box_index = entryIndex(b,mask_n * lw * lh + j * lw + i, 0);
                     const float class_multiplier = (classes_multipliers) ? classes_multipliers[class_id] : 1.0f;
                     ious all_ious = deltaBox/*!delta_yolo_box*/(truth, output, biases, n, box_index,
                                             i, j, lw, lh, net->w, net->h, delta, (2 - truth.w * truth.h), lw * lh,
                                             iou_normalizer * class_multiplier, iou_loss, 1, max_delta, net->rewritten_bbox, new_coords);
                     (*net->total_bbox)++;

                     // range is 0 <= 1
                     args.tot_iou += all_ious.iou;
                     args.tot_iou_loss += 1 - all_ious.iou;
                     // range is -1 <= giou <= 1
                     tot_giou += all_ious.giou;
                     args.tot_giou_loss += 1 - all_ious.giou;

                     tot_diou += all_ious.diou;
                     tot_diou_loss += 1 - all_ious.diou;

                     tot_ciou += all_ious.ciou;
                     tot_ciou_loss += 1 - all_ious.ciou;

                     int obj_index = entryIndex(b,mask_n * lw * lh + j * lw + i, 4);
                     avg_obj += output[obj_index];
                     if (objectness_smooth) {
                        float delta_obj = class_multiplier * obj_normalizer * (1 - output[obj_index]);
                        if (delta[obj_index] == 0)
                           delta[obj_index] = delta_obj;
                     } else
                        delta[obj_index] = class_multiplier * obj_normalizer * (1 - output[obj_index]);

                     int class_index = entryIndex(b,mask_n * lw * lh + j * lw + i, 4 + 1);
                     deltaClass/*!delta_yolo_class*/(output, delta, class_index, class_id, classes,
                              lw * lh, &avg_cat, focal_loss, label_smooth_eps, classes_multipliers, cls_normalizer);

                     ++(args.count);
                     ++(args.class_count);
                     if (all_ious.iou > .5)
                        recall += 1;
                     if (all_ious.iou > .75)
                        recall75 += 1;
                  }//end             if (iou > iou_thresh) {
               }//end  if (mask_n >= 0 && n != best_n && iou_thresh < 1.0f) {
            }//end       for (n = 0; n < total; ++n) {
         }

         if (iou_thresh < 1.0f) {
            // averages the deltas obtained by the function: delta_yolo_box()_accumulate
            for (j = 0; j < lh; ++j) {
               for (i = 0; i < lw; ++i) {
                  for (n = 0; n <anchorNumbers; ++n) {
                     int obj_index = entryIndex(b,n*lw*lh + j*lw + i, 4);
                     int box_index = entryIndex(b,n*lw*lh + j*lw + i, 0);
                     int class_index = entryIndex(b,n*lw*lh + j*lw + i, 4 + 1);
                     const int stride = lw*lh;
                     if (delta[obj_index] != 0)
                        averagesDeltas/*!averages_yolo_deltas*/(class_index, box_index, stride, classes, delta);
                  }
               }
            }
         }

      }//与最开始的{对应

      return args;
   }

   /*
   * 以下三个方法实现ComputingUnit接口
   */
   void excuse(int start,int end,apointer userData){
      a_assert(end-start==1);
      train_yolo_args *totalArgs=(train_yolo_args*)userData;
      train_yolo_args args=processBatch(totalArgs->state,start);
      lock.lock();
      totalArgs->tot_iou+=args.tot_iou;
      totalArgs->tot_giou_loss+=args.tot_giou_loss;
      totalArgs->tot_iou_loss+=args.tot_iou_loss;
      totalArgs->count+=args.count;
      totalArgs->class_count+=args.class_count;
      lock.unlock();
   }

   int  getNeedThreadCount(){
      return batch;
   }

   int  getCircleCount(){
      return 1;
   }

};

