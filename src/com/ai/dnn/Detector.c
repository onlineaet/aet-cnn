#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <aet/io/AFile.h>
#include "Detector.h"
#include "NetworkFactory.h"
#include "ClassesIface.h"
#include "YoloLayer.h"
#include "DnnUtils.h"

/**
* 实现检测
* 一幅图像里有几个目标。每个目标属于那一类，并且目标的大小和位置，属于某类的置信度是多少。
*/
impl$ Detector{

   Detector(int ngpus,char *dataCfg,char *cfgFile,char *weightFile,int argc,char **argv){
      super$(ngpus,dataCfg,cfgFile,weightFile);
      function = getFunction(argc,argv);
      self->argc=argc;
      self->argv=argv;
   }

   int findArg(int argc, char* argv[], char *arg){
      int i;
      for(i = 0; i < argc; ++i) {
         if(!argv[i])
            continue;
         if(0==strcmp(argv[i], arg))
            return 1;
      }
      return 0;
   }

   int findIntArg(int argc, char **argv, char *arg, int def){
      int i;
      for(i = 0; i < argc-1; ++i){
         if(!argv[i])
            continue;
         if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            break;
         }
      }
      return def;
   }

   float findFloatArg(int argc, char **argv, char *arg, float def){
      int i;
      for(i = 0; i < argc-1; ++i){
         if(!argv[i])
            continue;
         if(0==strcmp(argv[i], arg)){
            def = atof(argv[i+1]);
            break;
         }
      }
      return def;
   }


   char *findCharArg(int argc, char **argv, char *arg, char *def){
      int i;
      for(i = 0; i < argc-1; ++i){
         if(!argv[i])
            continue;
         if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            break;
         }
      }
      return def;
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

   const char *getFunction(){
      return function;
   }

   char **getLabels(const char *fileName){
      FILE *fp=fopen(fileName,"r");
      if(!fp)
         return NULL;
      char buffer[10*1024];
      int rev=fread(buffer,1,10*1024,fp);
      buffer[rev]='\0';
      char **items=a_strsplit(buffer,"\n",-1);
      return items;
   }


   NImage **load_alphabet(){
      int i, j;
      const int nsize = 8;
      NImage **alphabets = (NImage**)xcalloc(nsize, sizeof(NImage*));
      for(j = 0; j < nsize; ++j){
         NImage **items = (NImage**)xcalloc(128, sizeof(NImage*));
         for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "data/labels/%d_%d.png", i, j);
            items[i] =NImage.createImage(buff, 0, 0);
         }
         alphabets[j]=items;
      }
      return alphabets;
   }

   /*
   * 检测配置数据如下:
   * classes= 80
   * train  = /home/pjreddie/data/coco/trainvalno5k.txt
   * valid  = coco_testdev
   * #valid = data/coco_val_5k.list
   * names = data/coco.names
   * backup = /home/pjreddie/backup/
   * eval=coco
   */
   //检测测式 batch 等于1
   void runTest(){
      float thresh = findFloatArg(argc, argv, "-thresh", .25);    // 0.24
      float hier_thresh = findFloatArg(argc, argv, "-hier", .5);
      int dont_show = findArg(argc, argv, "-dont_show");
      int ext_output = findArg(argc, argv, "-ext_output");
      int save_labels = findArg(argc, argv, "-save_labels");
      char *outfile = findCharArg(argc, argv, "-out", 0);
      int letter_box = findArg(argc, argv, "-letter_box");
      int benchmark_layers = findArg(argc, argv, "-benchmark_layers");
      char *testfile = findCharArg(argc, argv, "-testfile", 0);

      ConfigFile *dataCfg =new$ ConfigFile(getDataCfgFile());
      printf("Detector.c runTest 00 数据文件:%s\n",getDataCfgFile());
      char *nameList=dataCfg->getStr(NULL,"names",TRUE);
      printf("Detector.c runTest 11 nameList:%s\n",nameList);

      int namesCount = 0;
      char **names = getLabels/*!get_labels_custom*/(nameList);
      if(names){
         namesCount=a_strv_length(names);//a_strv_length返回的是81不是80
         if(strlen(names[namesCount-1])==0)
            namesCount-=1;
      }
      NImage **alphabet=load_alphabet();
      printf("Detector.c runTest 22 创建网络 %s\n",getCfgFile());
      NNetwork *net=NULL;//NetworkFactory.getInstance()->createNetwork(getCfgFile());
      printf("Detector.c runTest 33 加载权重 net:%p\n",net);
      char *weightFile = getWeightFile();
      if (weightFile) {
         net->loadWeights(weightFile);
      }

      if (net->letter_box)
         letter_box = 1;
      net->benchmark_layers = benchmark_layers;
      net->fuseConvBatchnorm/*!fuse_conv_batchnorm(net)*/();
      net->calculateBinaryWeights/*!calculate_binary_weights(net);*/();
      NLayer  *lastLayer=net->getLastLayer();
      if(lastLayer->type==LayerType.YOLO
      ||lastLayer->type==LayerType.SOFTMAX
      ||lastLayer->type==LayerType.REGION
      ||lastLayer->type==LayerType.DETECTION
      ||lastLayer->type==LayerType.GAUSSIAN_YOLO){
         int classes=((ClassesIface*)lastLayer)->getClasses();
         if(classes!=namesCount){
            printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
                        nameList, namesCount, classes, getDataCfgFile());
         }
      }else{
         a_error("最后一层是:%s\n",NLayer.getLayerString(lastLayer->type));
      }

      srand(2222222);
      char buff[256];
      char *input = buff;
      char *json_buf = NULL;
      int json_image_id = 0;
      FILE* json_file = NULL;
      if (outfile) {
         json_file = fopen(outfile, "wb");
         if(!json_file) {
            a_error("fopen failed %s",outfile);
         }
         char *tmp = "[\n";
         fwrite(tmp, sizeof(char), strlen(tmp), json_file);
      }
      int j;
      float nms = .45;    // 0.4F
      while (1) {
         if(testfile) {
            strncpy(input, testfile, 256);
            if (strlen(input) > 0)
               if (input[strlen(input) - 1] == 0x0d)
                  input[strlen(input) - 1] = 0;
         }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input)
               break;
            strtok(input, "\n");
         }

         NImage *im = NImage.createImage/*!load_image*/(input, 0, 0, net->c);
         NImage *sized=NULL;
         printf("Detector.c runTest 检测的图片 input:%s letter_box:%d net w:%d h:%d c:%d\n",input,letter_box,net->w,net->h,net->c);

         if(letter_box)
            sized = im->letterboxImage/*!letterbox_image*/(net->w, net->h);
         else
            sized = im->resize/*!resize_image*/(net->w, net->h);

         NLayer *l =lastLayer;// net.layers[net.n - 1];
         int k;
         for (k = 0; k < net->getLayerCount(); ++k) {
            NLayer *lk=net->getLayer(k);
            if (lk->type ==LayerType.YOLO || lk->type == LayerType.GAUSSIAN_YOLO || lk->type == LayerType.REGION) {
               l = lk;
               printf(" Detection layer: %d - type = %d %s \n", k, lk->type,NLayer.getLayerString(lk->type));
            }
         }

         InputData *inputData=DataFactory.getInstance()->createInputData(sized->getSize(),1);
        // inputData->setImageData(sized->getData(),0);
         aint64 time= Time.currentTime();
         //进入核心
         printf("开始检测---\n");
         net->predict/*!network_predict(net, X)*/(inputData);
         //network_predict_image(&net, im); letterbox = 1;
         printf("%s: Predicted in %lli 毫秒.\n", input, (Time.currentTime() - time) / 1000);
         //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));
         if(!(l varof$ YoloLayer)){
            a_error("应该是yolo层。");
         }
         YoloLayer *yl=(YoloLayer*)l;
         int nboxes = 0;
         Detection **dets =net->getBoxes/*!get_network_boxes*/(im->width, im->height,
               thresh, hier_thresh, 0, 1, &nboxes, letter_box);
         printf("detector yolo is number:%d nboxes:%d\n",yl->getOrderNumber(),nboxes);
         if (nms) {
            printf("detector test_detector %f %d\n",nms,(yl->nms_kind == DEFAULT_NMS));
            if (yl->nms_kind == DEFAULT_NMS)
               Detection.doNmsSort/*!do_nms_sort*/(dets, nboxes, yl->classes, nms);
            else
               Detection.diounmsSort/*!diounms_sort*/(dets, nboxes, yl->classes, nms, yl->nms_kind, yl->beta_nms);
         }

         im->drawDetectionsV3/*!draw_detections_v3*/(dets, nboxes, thresh, names, alphabet, yl->classes, ext_output);
         im->save/*!save_image*/("predictions");
         if (!dont_show) {
            im->show/*!show_image*/("predictions");
         }

         if (json_file) {
            if (json_buf) {
               char *tmp = ", \n";
               fwrite(tmp, sizeof(char), strlen(tmp), json_file);
            }
            ++json_image_id;
            json_buf = net->detectionToJson/*!detection_to_json*/(dets, nboxes, yl->classes, names, json_image_id, input);

            fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
            free(json_buf);
         }

         // pseudo labeling concept - fast.ai
         if (save_labels){
            char labelpath[4096];
            DnnUtils.replaceImageToLabel/*!replace_image_to_label*/(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
               char buff[1024];
               int class_id = -1;
               float prob = 0;
               for (j = 0; j < yl->classes; ++j) {
                  if (dets[i]->prob[j] > thresh && dets[i]->prob[j] > prob) {
                     prob = dets[i]->prob[j];
                     class_id = j;
                  }
               }
               if (class_id >= 0) {
                  sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n",
                        class_id, dets[i]->bbox.x, dets[i]->bbox.y, dets[i]->bbox.w, dets[i]->bbox.h);
                  fwrite(buff, sizeof(char), strlen(buff), fw);
               }
            }
            fclose(fw);
         }

         int i;
         for(i=0;i<nboxes;i++)
            dets[i]->unref()/*!free_detections(dets, nboxes);*/;
         free(dets);
         im->unref();
         sized->unref();
         /*
         if (!dont_show) {
         wait_until_press_key_cv();
         destroy_all_windows_cv();
         }
         */

         if (testfile)
            break;
      }//end while

      if (json_file) {
         char *tmp = "\n]";
         fwrite(tmp, sizeof(char), strlen(tmp), json_file);
         fclose(json_file);
      }
      // free memory
      //          free_ptrs((void**)names, net.layers[net.n - 1].classes);
      //          free_list_contents_kvp(options);
      //          free_list(options);
      //          free_alphabet(alphabet);
      //          free_network(net);
   }

   void run(){
      if(!strcmp(function,"test")){
         runTest();
      }
   }

};

