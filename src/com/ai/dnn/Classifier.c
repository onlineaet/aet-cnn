#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>

#include <aet/util/ARandom.h>
#include <aet/mtcs/MtcsMem.h>

#include "Classifier.h"
#include "NetworkFactory.h"
#include "ImageData.h"
#include "OutputLayer.h"
#include "cnnmicro.h"
#include "DnnUtils.h"

/**
 * 分类。
 * 图像分类为主
 * 给一幅图，把内容归类。
 */
impl$ Classifier{

   Classifier(int ngpus,char *dataCfg,char *cfgFile,char *weightFile,int argc,char **argv){
      super$(ngpus,dataCfg,cfgFile,weightFile);
      function = getFunction(argc,argv);
      self->argc=argc;
      self->argv=argv;
   }

   /**
   * 获取业务对应的具体功能，比如 测试，训练，draw等
   * ./xxxx detector test 功能在第3元素
   */
   const char *getFunction(int argc, char **argv){
      int i;
      if(argc>=2)
         return argv[2];
      return NULL;
   }

   char **getLabels(const char *fileName){
      FILE *fp=fopen(fileName,"r");
      if(!fp)
         return NULL;
      struct stat sb;
      long rv=0;
      if (stat(fileName, &sb) == 0){
         rv = sb.st_size;
      }
      if(rv<100*1024){
         char buffer[100*1024];
         int rev=fread(buffer,1,100*1024,fp);
         buffer[rev]='\0';
         char **items=a_strsplit(buffer,"\n",-1);
         return items;
      }else{
         char * buffer=malloc(rv+1);
         int rev=fread(buffer,1,rv,fp);
         buffer[rev]='\0';
         char **items=a_strsplit(buffer,"\n",-1);
         a_free(buffer);
         return items;
      }
   }

   float sec(clock_t clocks){
      return (float)clocks/CLOCKS_PER_SEC;
   }

   //	    void train(char *datacfg, char *cfgfile, char *weightfile, int *gpus,
   //	          int ngpus, int dontuse_opencv, int dont_show, int mjpeg_port,
   //	          int calc_topk, int show_imgs, char* chart_path)
   void train(int dont_show, int mjpeg_port, int calc_topk, int show_imgs, char* chart_path){
      int i;
      int ngpus = 1;
      float avg_loss = -1;
      float avg_contrastive_acc = 0;
      char *base=getBaseFile();
      printf("%s\n", base);
      printf("%d\n", ngpus);
      printf("Classifier.c train 22 创建网络 %s\n",getCfgFile());
      NNetwork *net=NetworkFactory.getInstance()->createTrainNetwork(getCfgFile());
      char *weightFile = getWeightFile();
      if (weightFile) {
         net->loadWeights(weightFile);
      }

      srand(time(0));
      int seed = rand();
      srand(seed);
      int  clear=  findArg("-clear");
      if (clear) {
         net->seen = 0;
         net->cur_iteration = 0;
      }
      net->learning_rate *= ngpus;
      srand(time(0));
      int imgs = net->batch * net->subdivisions * ngpus;
      printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
      ConfigFile *dataCfg =new$ ConfigFile(getDataCfgFile());
      printf("Classfier.c runTrain 00 数据文件:%s train:%d\n",getDataCfgFile(),net->train);
      char *backup_directory=dataCfg->getStr(NULL,"backup","/backup/");
      char *label_list = dataCfg->getStr(NULL, "labels", "data/labels.list");
      char *train_list = dataCfg->getStr(NULL, "train", "data/train.list");
      int classes = dataCfg->optionFindInt(NULL, "classes", 2);
      int topk_data = dataCfg->optionFindInt(NULL, "top", 5);
      char topk_buff[10];
      sprintf(topk_buff, "top%d", topk_data);

      NLayer *l=net->getLastLayer();
      if (classes != l->outputs && (l->type == LayerType.SOFTMAX || l->type == LayerType.COST)) {
         printf("\n Error: num of filters = %d in the last conv-layer in cfg-file doesn't match to classes = %d in data-file \n",
                     l->outputs, classes);
         a_error("配置错误！");
      }
      net->setClasses(classes);
      net->setInitInputsAndOutputs(TRUE);

      int labelsCount = 0;
      char **labels = getLabels/*!get_labels_custom*/(label_list);
      if(labels){
         labelsCount=a_strv_length(labels);//a_strv_length返回的是11不是10
         if(strlen(labels[labelsCount-1])==0)
            labelsCount-=1;
      }
      int pathsCount = 0;
      char **paths = getLabels(train_list);
      if(paths){
         pathsCount=a_strv_length(paths);//a_strv_length返回的是11不是10
         if(strlen(paths[pathsCount-1])==0)
            pathsCount-=1;
      }
      printf("标签数：%d 训练图片数 %d\n",labelsCount,pathsCount);
      int train_images_num =pathsCount;

      ImageData *imageData=new$ ImageData(train_images_num,imgs,classes,net->w,net->h,net->c);
      imageData->threads = 12;
      if (net->contrastive && imageData->threads > net->batch/2)
         imageData->threads = net->batch / 2;
      imageData->hierarchy = net->hierarchy;

      imageData->contrastive = net->contrastive;
      //imageData->dontuse_opencv = dontuse_opencv;
      imageData->min = net->min_crop;
      imageData->max = net->max_crop;
      imageData->use_flip = net->flip;
      imageData->blur = net->blur;
      imageData->angle = net->angle;
      imageData->aspect = net->aspect;
      imageData->exposure = net->exposure;
      imageData->saturation = net->saturation;
      imageData->hue = net->hue;
      imageData->size = net->w > net->h ? net->w : net->h;

      imageData->label_smooth_eps = net->label_smooth_eps;
      imageData->mixup = net->mixup;
      if (dont_show && show_imgs)
         show_imgs = 2;
      imageData->show_imgs = show_imgs;
      imageData->paths = paths;
      imageData->labels = labels;
      imageData->type = CLASSIFICATION_DATA;

      int waitId = imageData->loadData/*!load_data*/();


      int iter_save = net->getCurrentBatch/*!get_current_batch*/();
      int iter_topk = iter_save;/*!get_current_batch(net);*/
      float topk = 0;

      //初始化分类训练数据
      int count = 0;
      double start, time_remaining, avg_time = -1, alpha_time = 0.01;
      start = what_time_is_it_now();
      clock_t time;
      clock_t sst=clock();
      while(net->getCurrentBatch() < net->max_batches || net->max_batches == 0){
         time=clock();
         //printf("装载数据\n");
         imageData->wait(waitId);
         waitId = imageData->loadData/*!load_data*/();
         //printf("Loaded: %lf seconds\n", sec(clock()-time));
         time=clock();
         float loss = net->train(imageData);/*!train_network(net, train);*/
         if(avg_loss == -1 || isnan(avg_loss) || isinf(avg_loss))
            avg_loss = loss;
         avg_loss = avg_loss*.9 + loss*.1;

         i = net->getCurrentBatch();/*!get_current_batch(net);*/
         // calculate TOPk for each 2 Epochs
         int calc_topk_for_each = iter_topk + 2 * train_images_num / (net->batch * net->subdivisions);
         calc_topk_for_each = fmax(calc_topk_for_each, net->burn_in);
         calc_topk_for_each = fmax(calc_topk_for_each, 100);
         if (i % 10 == 0) {
            if (calc_topk) {
               fprintf(stderr, "\n (next TOP%d calculation at %d iterations) ", topk_data, calc_topk_for_each);
               if (topk > 0)
                  fprintf(stderr, " Last accuracy TOP%d = %2.2f %% \n", topk_data, topk * 100);
            }

            if (net->cudnn_half) {
               if (i < net->burn_in * 3)
                  fprintf(stderr, " Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net->burn_in);
               else
                  fprintf(stderr, " Tensor Cores are used.\n");
            }
         }

         int draw_precision = 0;
         if (calc_topk && (i >= calc_topk_for_each || i == net->max_batches)) {
            iter_topk = i;
            if (net->contrastive && l->type != LayerType.SOFTMAX && l->type != LayerType.COST) {
               int k;
               for (k = 0; k < net->layerCount; ++k)
                  if (net->layers[k]->type == LayerType.CONTRASTIVE)
                     break;
               OutputLayer *outlayer=(OutputLayer *)net->layers[k];
               topk = outlayer->getCost()/100;/*!*(net->layers[k].loss) / 100;)*/
               sprintf(topk_buff, "Contr");
            }else {
               //topk = validate_classifier_single(datacfg, cfgfile, weightfile, &net, topk_data); // calc TOP-n
               //printf("\n accuracy %s = %f \n", topk_buff, topk);
               a_error("validate_classifier_single 还未实现\n");
            }
            draw_precision = 1;
         }

         time_remaining = ((net->max_batches - i) / ngpus) * (what_time_is_it_now() - start) / 60 / 60;
         // set initial value, even if resume training from 10000 iteration
         if (avg_time < 0)
            avg_time = time_remaining;
         else
            avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
         start = what_time_is_it_now();
         printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %llu images, %f hours left\n",
               net->getCurrentBatch/*!get_current_batch*/(), (float)(net->seen)/ train_images_num, loss, avg_loss,
               net->getCurrentRate/*!get_current_rate*/(), sec(clock()-time), net->seen, avg_time);

         if (i >= (iter_save + 1000)) {
            iter_save = i;
            char buff[256];
            printf("1000次所花时间:%f loss:%f avg_loss:%f\n",sec(clock()-sst),loss,avg_loss);
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            net->saveWeights/*!save_weights*/(buff);
         }

      }//end while
      printf("总时间:%f avg_loss:%f\n",sec(clock()-sst),avg_loss);

      char buff[256];
      sprintf(buff, "%s/%s_final.weights", backup_directory, base);
      net->saveWeights/*!save_weights*/(buff);
      imageData->unref();/*!free_data(train);*/

      if(labels)
         free(labels);
      a_strfreev(paths);
      free(base);
   }

   void change_leaves(tree *t, char *leaf_list){
       char **leaves = getLabels/*!get_labels_custom*/(leaf_list);
       int labelsCount = 0;
       if(leaves){
          labelsCount=a_strv_length(leaves);//a_strv_length返回的是11不是10
          if(strlen(leaves[labelsCount-1])==0)
             labelsCount-=1;
       }
       int n = labelsCount;
       int i,j;
       int found = 0;
       for(i = 0; i < t->n; ++i){
           t->leaf[i] = 0;
           for(j = 0; j < n; ++j){
               if (0==strcmp(t->name[i], leaves[j])){
                   t->leaf[i] = 1;
                   ++found;
                   break;
               }
           }
       }
       fprintf(stderr, "Found %d leaves.\n", found);
   }

   //测试
   //原型 validate_classifier_single classifier.c
   float validateSingle(int topk_custom){

      int i, j;
      int old_batch = -1;
      printf("validate_classifier_single 00 data cfg:%s\n",getDataCfgFile());
      printf("validate_classifier_single 11 cfg:%s\n",getCfgFile());
      printf("validate_classifier_single 22 weightfile:%s\n",getWeightFile());
      printf("validate_classifier_single 33 topk_custom:%d\n",topk_custom);

      NNetwork *net=NetworkFactory.getInstance()->createNetwork(getCfgFile(),1,0);
      char *weightFile = getWeightFile();
      if (weightFile) {
         net->loadWeights(weightFile);
         net->fuseConvBatchnorm/*!fuse_conv_batchnorm*/();
         net->calculateBinaryWeights/*!calculate_binary_weights*/();
      }
      srand(time(0));
      ConfigFile *dataCfg =new$ ConfigFile(getDataCfgFile());
      printf("Classfier.c valid 00 数据文件:%s\n",getDataCfgFile());
      char *backup_directory=dataCfg->getStr(NULL,"backup","/backup/");
      char *label_list = dataCfg->getStr(NULL, "labels", "data/labels.list");
      char *leaf_list = dataCfg->getStr(NULL, "leaves", NULL);
      if(leaf_list)
         change_leaves(net->hierarchy, leaf_list);
      char *valid_list =  dataCfg->getStr(NULL, "valid", "data/train.list");

      int classes = dataCfg->optionFindInt(NULL, "classes", 2);
      int topk = dataCfg->optionFindInt(NULL, "top", 1);
      if (topk_custom > 0)
         topk = topk_custom;    // for validation during training
      if (topk > classes)
         topk = classes;

      int labelsCount = 0;
      char **labels = getLabels/*!get_labels_custom*/(label_list);
      if(labels){
         labelsCount=a_strv_length(labels);//a_strv_length返回的是11不是10
         if(strlen(labels[labelsCount-1])==0)
            labelsCount-=1;
      }

      int pathCount = 0;
      char **paths = getLabels/*!get_labels_custom*/(valid_list);
      if(paths){
         pathCount=a_strv_length(paths);//a_strv_length返回的是11不是10
         if(strlen(paths[pathCount-1])==0)
            pathCount-=1;
      }

      float avg_acc = 0;
      float avg_topk = 0;
      int* indexes = (int*)xcalloc(topk, sizeof(int));
      printf("classifier valid  pathCount:%d train:%d batch:%d topk:%d classes:%d\n",pathCount,net->train,net->batch,topk,classes);
      net->setInitInputsAndOutputs(FALSE);
      float *pred=malloc(classes*sizeof(float));
      for(i = 0; i < pathCount; ++i){
         int class_id = -1;
         char *path = paths[i];
         for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
               class_id = j;
               break;
            }
         }
         NImage *im=NImage.createImage(paths[i],0,0);
         NImage *resized = im->resizeMin(net->w);
         NImage  *crop = resized->crop((resized->width - net->w)/2, (resized->height - net->h)/2, net->w, net->h);
         OutputData *outputData = net->predict/*!network_predict*/(crop);
         float *out = outputData->getDataArray();
         if(net->useMtcs){
            MtcsMem.memcpy(pred,out,outputData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
         }else{
            pred=out;
         }
         if(net->hierarchy)
            hierarchy_predictions(pred, net->outputs, net->hierarchy, 1);
         //int x;
         //for(x=0;x<classes;x++)
           // printf("预测值:%d %f class_id:%d %s\n",x,pred[x],class_id,paths[i]);

         if(resized!= im)
            resized->unref();
         im->unref();
         crop->unref();
         DnnUtils.topK/*!top_k*/(pred, classes, topk, indexes);

         if(indexes[0] == class_id)
            avg_acc += 1;
         for(j = 0; j < topk; ++j){
            if(indexes[j] == class_id)
               avg_topk += 1;
         }
         printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
      }
      free(indexes);
      float topk_result = avg_topk / i;
      return topk_result;
   }

   void run(){
      if(!strcmp(function,"train")){
         int mjpeg_port = findIntArg("-mjpeg_port", -1);
         int dont_show = findArg("-dont_show");
         int benchmark = findArg("-benchmark");
         int benchmark_layers = findArg("-benchmark_layers");
         if (benchmark_layers)
            benchmark = 1;
         int dontuse_opencv = findArg("-dontuse_opencv");
         int show_imgs = findArg("-show_imgs");
         int calc_topk = findArg("-topk");
         int cam_index = findIntArg("-c", 0);
         int top = findIntArg("-t", 0);
         // int clear = findArg("-clear");
         char* chart_path = findCharArg("-chart", 0);
         train(dont_show, mjpeg_port, calc_topk, show_imgs, chart_path);
      }else if(!strcmp(function,"valid")){
         validateSingle(-1);
      }
   }

};

