#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/System.h>
#include <aet/io/AFile.h>
#include "WorkFactory.h"
#include "DnnUtils.h"
#include "cnnmicro.h"
#include "Detector.h"
#include "Classifier.h"

//声明在cnnmacro.h
int use_mtcs = 0;


/**
 * 参数列表
 * -train      是否训练
 * -test       是否测试
 * -data   -d  数据文件
 * -clear      是否清除
 * -cfg    -c  配置文件
 * -weight -w  权重文件
 * -backup -b  备份文件所在目录
 */
impl$ WorkFactory{
   static WorkFactory *getInstance(){
      static WorkFactory *singleton = NULL;
      if (!singleton){
         singleton =new$ WorkFactory();
      }
      return singleton;
   }

   /**
   * 是训练吗？
   */
   aboolean isTrain(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-train")==0){
            return TRUE;
         }
      }
      return FALSE;
   }

   /**
   * 是测试吗？
   */
   aboolean isTest(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-test")==0){
            return TRUE;
         }
      }
      return FALSE;
   }

   /**
   * 需要清除吗？
   */
   aboolean needClear(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-clear")==0){
            return TRUE;
         }
      }
      return FALSE;
   }

   /**
   * 数据所在目录
   */
   char *getDataDirectory(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-data")==0 || a_ascii_strcasecmp(arg,"-d")==0){
            return argv[i+1];
         }
      }
      return NULL;
   }

   /**
   * 返回配置文件
   */
   char *getCfgFile(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-cfg")==0 || a_ascii_strcasecmp(arg,"-c")==0){
            return argv[i+1];
         }
      }
      return NULL;
   }

   /**
   * 返回权重所在文件
   */
   char *getWeightFile(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-weight")==0 || a_ascii_strcasecmp(arg,"-w")==0){
            return argv[i+1];
         }
      }
      return NULL;
   }

   /**
   * 创建检测业务
   */
   Business *buildDetector(int argc, char **argv){
      aint64 time=System.currentTime();
      int ngpus = 1;
      aboolean clear = needClear(argc,argv);
      char *datacfg = getDataDirectory(argc,argv);//数据所在目录
      char *cfg = getCfgFile(argc,argv);    //网络和层的配置文件
      char *weights = getWeightFile(argc,argv);//权重文件
      Detector *det=new$ Detector(ngpus,datacfg,cfg,weights,argc,argv);
      det->setClear(clear);
      printf("这是检测训练---ngpus:%d dataCfg:%s cfg:%s weights:%s \n",ngpus,datacfg,cfg,weights);
      return det;
   }

   /**
   * 返回权重所在文件
   */
   aboolean useMtcs(int argc, char **argv){
      int i;
      for (i = 1; i < argc; ++i) {
         char *arg = argv[i];
         if(a_ascii_strcasecmp(arg,"-usemtcs")==0){
            return TRUE;
         }
      }
      return FALSE;
   }
   /**
   * 创建分类业务
   */
   Business *buildClassifier(int argc, char **argv){
      int ngpus = 1;
      aboolean clear = needClear(argc,argv);
      char *datacfg = getDataDirectory(argc,argv);//数据所在目录
      char *cfg = getCfgFile(argc,argv);          //网络和层的配置文件
      char *weights = getWeightFile(argc,argv);   //权重文件
      Classifier *classifier=new$ Classifier(ngpus,datacfg,cfg,weights,argc,argv);
      classifier->setClear(clear);
      return classifier;
   }

   /**
   * 两种AI业务 1.检测(detector) 2.分类(classifier)
   */
   Business *build(int argc, char **argv){
      aboolean isMtcs=useMtcs(argc,argv);
      if(isMtcs)
         use_mtcs = 1;
      if(a_ascii_strcasecmp(argv[1],"detector")==0){
         //检测业务
         return buildDetector(argc,argv);
      }else if(a_ascii_strcasecmp(argv[1],"classifier")==0){
         return buildClassifier(argc,argv);
      }
      return NULL;
   }

};



