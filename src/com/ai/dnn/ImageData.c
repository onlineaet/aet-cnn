#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <aet/lang/AAssert.h>

#include <aet/util/ARandom.h>
#include "Classifier.h"
#include "ImageData.h"
#include "NetworkFactory.h"
#include "DnnUtils.h"
#include "DstbCompute.h"
#include "cnnmicro.h"


impl$ ImageData{

   ImageData(int totalImageCount,int batchImgCount,int classes,int w,int h,int c){
      self->totalImageCount=totalImageCount;
      self->batchImageCount=batchImgCount;
      self->classes=classes;
      self->w=w;
      self->h=h;
      self->c=c;
      //两倍大小的图像数据
      batchImageData =malloc(w*h*c*batchImgCount*2*sizeof(float));
      truthsData =malloc(classes*batchImgCount*2*sizeof(float));
      imoffset=0;
      int i;
      for(i=0;i<batchImageCount;i++){
         srcImgs[i]=new$ NImage(w,h,c);
      }
   }

   //从总图片数中随机取imgCount张图片
   void getRandomPaths(int imgCount,int contrastive,char **randomPaths){
      int i;
      int old_index = 0;
      for(i = 0; i < imgCount; ++i){
         do {
            int index = DnnUtils.randomGen/*!random_gen*/() % totalImageCount;
            if (contrastive && (i % 2 == 1))
               index = old_index;
            else
               old_index = index;
            randomPaths[i] = paths[index];
            if (strlen(randomPaths[i]) <= 4)
               printf(" Very small path to the image: %s \n", randomPaths[i]);
         } while (strlen(randomPaths[i]) == 0);
      }
   }

   //减少图像操作，测试gpu的频率是否能提升，如果能，证明Gpu频率下降是在等cpu
   void loadImageAugmentPaths(int start,int imgCount,char **paths){
      int i;
      for(i = 0; i < imgCount; ++i){
         int size = w > h ? w : h;
         int imgIndex = (contrastive) ? (i / 2) : i;
         NImage *im=srcImgs[start+i];
         im->reload(paths[imgIndex]);
         NImage *crop = im->randomAugment/*!random_augment_image*/(angle, aspect, min, max, size);
         int flip = use_flip ? DnnUtils.randomGen/*!random_gen*/() % 2 : 0;
         if (flip)
            crop->flip/*!flip_image*/();

         crop->randomDistort/*!random_distort_image*/(hue, saturation, exposure);
         float *data=batchImageData+(imoffset+start+i)*w*h*c;
         crop->resize/*!resize_image*/(w, h,c,data);
         crop->unref();
      }
   }

   //原型 fill_truth_smooth data.h data.c
   //如果有10个类别
   //truth可能是这样 {0,0,0,1,0,0,0,0,0,0}或{1,0,0,0,0,0,0,0,0,0}
   //1表示path代表的是labels中的类别
   //labels的内容是{dog,cat,airplane,car,...}; 上例中的truth中的1在第4个元素，代表的是car,说明path这张图标注为car
   void fillTruthSmooth(char *path, float *truth){
      int k=classes;//类别数
      int i;
      memset(truth, 0, k * sizeof(float));
      int count = 0;
      for (i = 0; i < k; ++i) {
         if(strstr(path, labels[i])){
            // if(strstr(path, re)){
            truth[i] = (1 - label_smooth_eps);
            ++count;
         } else {
            truth[i] = label_smooth_eps / (k - 1);
         }
      }
      if (count != 1) {
         int j = 0;
         for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
               printf("\t label %d: %s  \n", j, labels[i]);
               j++;
            }
         }
         a_error("Too many or too few labels: %d, %s\n", count, path);
      }
   }

   //原型 fill_hierarchy data.c
   void fillHierarchy(float *truth, int k, tree *hierarchy){
      int j;
      for(j = 0; j < k; ++j){
         if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
               truth[parent] = 1;
               parent = hierarchy->parent[parent];
            }
         }
      }
      int i;
      int count = 0;
      for(j = 0; j < hierarchy->groups; ++j){
         //printf("%d\n", count);
         int mask = 1;
         for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
               mask = 0;
               break;
            }
         }
         if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
               truth[count + i] = SECRET_NUM;
            }
         }
         count += hierarchy->group_size[j];
      }
   }

   void loadLabelPaths(int start,int imgCount,char **randomPath){
      int i;
      if (labels) {
         // supervised learning
         for (i = 0; i < imgCount; ++i) {
            const int img_index = (contrastive) ? (i / 2) : i;
            float *truth =truthsData+(imoffset+start+i)*classes;
            fillTruthSmooth/*!fill_truth_smooth*/(randomPath[img_index], truth);
            //printf(" n = %d, i = %d, img_index = %d, class_id = %d \n", n, i, img_index, find_max(y.vals[i], k));
            if (hierarchy) {
               fillHierarchy/*!fill_hierarchy*/(truth, classes, hierarchy);
            }
         }
      } else {
         // unsupervised learning
         for (i = 0; i < imgCount; ++i) {
            const int img_index = (contrastive) ? (i / 2) : i;
            const uintptr_t path_p = (uintptr_t)randomPath[img_index];// abs(random_gen());
            const int class_id = path_p % classes;
            int l;
            float *truth =truthsData+(imoffset+start+i)*classes;
            for (l = 0; l < classes; ++l)
               truth[l] = 0;
            truth[class_id] = 1;
         }
      }
   }

   /**
   * 生成输入层图片数据
   */
   int loadData(){
      int id=DstbCompute.getInstance()->addTask((ComputingUnit*)self,NULL);
     // fprintf(stderr,"ImageData 装载图片数据与真实数据 11\n");
    //  DstbCompute.getInstance()->wait(id);
     // fprintf(stderr,"ImageData 装载图片数据与真实数据 22 time:%lli 数量:%d\n",(Time.monotonic()-time),batchImageCount);
      return id;
   }

   /**
    * 调用loadData后，必须调用wait等待线程池中的任务完成。
    */
   void wait(int id){
      DstbCompute.getInstance()->wait(id);
      if(imoffset==0)
         imoffset=batchImageCount;
      else
         imoffset = 0;
   }

   //返回一次生成的图片数
   int getImagCount(){
      return batchImageCount;
   }

   //原型 get_next_batch data.h data.c
   //n是一个batch的大小
   //如果net->subdivisions = 1 net->batch=batchImageCount
   //网络开始需要的输入数据
   void getNextBatch(int n, int offset, InputData *X,TruthData *y){
      float *im=batchImageData+(imoffset==0?w*h*c*batchImageCount:0)+offset*w*h*c;
      X->setImageData(im);
      float *tr=truthsData+(imoffset==0?classes*batchImageCount:0)+offset*classes;
      y->setData(tr);
   }

   /**
   * 实现 ComputingUnit接口的三个方法
   * 该方法运行在线程中
   */
   void excuse(int start,int end,apointer userData){
      //printf("在线程中创建新图片 load_data_detection 00 angle:%f w:%d h:%d start:%d end:%d %p\n",
                  //angle,w,h,start,end,AThread.current());
      int imgCount=end-start;
      char *randomPath[imgCount];
      getRandomPaths(imgCount,0,randomPath);
      loadImageAugmentPaths(start,imgCount,randomPath);
      loadLabelPaths/*!load_labels_paths*/(start,imgCount,randomPath);
   }

   int  getNeedThreadCount(){
      return threads;
   }

   int  getCircleCount(){
      return batchImageCount;
   }
};

