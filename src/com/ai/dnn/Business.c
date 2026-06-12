#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <aet/time/Time.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/AString.h>
#include <aet/io/AFile.h>
#include <aet/util/ARandom.h>
#include "NetworkFactory.h"
#include "Business.h"

impl$ Business{

   Business(int ngpus,char *dataCfg,char *cfgFile,char *weightFile){
      nets=a_new(NNetwork *,ngpus);
      netCount=ngpus;
      self->dataCfg=a_strdup(dataCfg);
      self->cfgFile=a_strdup(cfgFile);
      self->configFile=new$ ConfigFile(cfgFile);
      self->weightFile=a_strdup(weightFile);
      self->clear=FALSE;
      backupDirectory=createBackupDirectory();
      self->baseFile=getCfgName(cfgFile);
   }

   public$ char  *getDataCfgFile(){
      return dataCfg;
   }

   public$ char  *getWeightFile(){
      return weightFile;
   }

   /**
   *   从 cfgFile="/home/aet/workspace/train1/cfg/yolov3-tiny.cfg"
   *   取出 yolov3-tiny
   */
   char *getCfgName(char *cfgFile){
      AFile *file=new$ AFile(cfgFile);
      char *fileName=file->getName();
      AString *re=new$ AString(fileName);
      int index=re->lastIndexOf(".");
      char *last=NULL;
      if(index>=0){
         AString *ret=re->substring(0,index);
         last=ret->unrefStr();
      }else{
         last=re->unrefStr();
      }
      file->unref();
      return last;
   }

   public$ char *getCfgFile(){
      return cfgFile;
   }

   private$ char *createBackupDirectory(){
      char *backup=configFile->getBackup();
      char *ret=NULL;
      if(backup==NULL){
         AFile *file=new$ AFile("backup/");
         file->makeDirs();
         char *dir=file->getAbsolutePath();
         ret=a_strdup(dir);
         file->unref();
      }else{
         ret=a_strdup(backup);
      }
      return ret;
   }

   void setClear(aboolean clear){
      self->clear=clear;
   }

   NNetwork *getNetwork(){
      return nets[0];
   }

   int getNetCount(){
      return netCount;
   }

   ConfigFile *getConfigFile(){
      return configFile;
   }

   public$ char *getBackupDirectory(){
      return backupDirectory;
   }

   public$ char *getBaseFile(){
      return baseFile;
   }

   int findArg(char *arg){
      int i;
      for(i = 0; i < argc; ++i) {
         if(!argv[i])
            continue;
         if(0==strcmp(argv[i], arg))
            return 1;
      }
      return 0;
   }

   int findIntArg(char *arg, int def){
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

   float findFloatArg(char *arg, float def){
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


   char *findCharArg(char *arg, char *def){
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



};


