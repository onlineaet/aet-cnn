#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <aet/lang/AString.h>
#include "ConfigFile.h"


impl$ ConfigFile{

   ConfigFile(char *fileName){
      AError *error=NULL;
      cfgFile=createKeyFile(fileName,&error);
      if(error){
         a_error_free(error);
         //没有组名的key,换其它
         return NULL;
      }
      self->fileName=a_strdup(fileName);
   }

   AKeyFile *createKeyFile(char *fileName,AError **error){
      AError *curErr=NULL;
      AKeyFile *keyFile=new$ AKeyFile(fileName,&curErr);
      if(curErr){
         printf("读KeyFile 有一个信息:%s %d %d\n",curErr->message,curErr->code, KeyFileError.GROUP_NOT_FOUND);
         if(curErr->code== KeyFileError.GROUP_NOT_FOUND){
            //说明文件中没有组 加一个组
            FILE *fp=fopen(fileName,"r");
            if(!fp){
               a_error_transfer(error,curErr);
               a_error_free(curErr);
               return NULL;
            }
            a_error_free(curErr);
            curErr=NULL;
            char *group="[default]\n";
            int size=10*1024+strlen(group);
            char *buffer=malloc(size);
            memcpy(buffer,group,strlen(group));
            int rev=fread(buffer+strlen(group),1,size-strlen(group),fp);
            fclose(fp);
            buffer[rev+strlen(group)]='\0';
            keyFile=new$ AKeyFile(buffer,rev+strlen(group),0,&curErr);
            free(buffer);
            if(curErr){
               a_error_transfer(error,curErr);
               return NULL;
            }
         }else{
            a_error_transfer(error,curErr);
            return NULL;
         }
      }
      if(keyFile->isEmpty()){
         a_error("配置文件%s是空的。",fileName);
         return NULL;
      }
      return keyFile;
   }

   char *getStr(char *group, char *key,aboolean quiet){
      AError *error=NULL;
      if(group==NULL &&  cfgFile->getGroupCount()==1){
         group = cfgFile->getGroupName(0);
      }
      char *v = cfgFile->getValue(group, key,&error);
      if(error){
         //if(!quiet)
         // fprintf(stderr, "group:%s:key:%s 没有找到关键字对应的值 error:%s\n", group,key, error->message);
         a_error_free(error);
         return NULL;
      }
      return v;
   }

   char    *getStr(char *group, char *key,char *def){
      if(group==NULL &&  cfgFile->getGroupCount()==1){
         group = cfgFile->getGroupName(0);
      }
      char *v = cfgFile->getValue(group, key,NULL);
      if(!v)
         return def;
      return v;
   }


   int optionFindInt(char *group, char *key, int def,aboolean quiet){
      char *value=getStr(group,key,quiet);
      int ret=def;
      if(!value){
         //if(!quiet)
         //fprintf(stderr, "group:%s:key:%s Using default '%d'\n", group,key, def);
         return ret;
      }
      if(value) {
         ret=atoi(value);
      }
      return ret;
   }

   float optionFindFloat(char *group, char *key, float def,aboolean quiet){
      char *value=getStr(group,key,quiet);
      float ret=def;
      if(!value){
         // if(!quiet)
         // fprintf(stderr, "group:%s:key:%s Using default '%lf'\n", group,key, def);
         return ret;
      }
      if(value) {
         ret=atof(value);
      }
      return ret;
   }

   int optionFindInt(char *group, char *key, int def){
      return optionFindInt(group,key,def,FALSE);
   }

   float optionFindFloat(char *group, char *key, float def){
      return optionFindFloat(group,key,def,FALSE);
   }

   int optionFindIntQuiet(char *group, char *key, int def){
      return optionFindInt(group,key,def,TRUE);
   }

   float optionFindFloatQuiet(char *group, char *key, float def){
      return optionFindFloat(group,key,def,TRUE);
   }

   /**
   * 权重存放的路径
   */
   char *getBackup(){
      char *value=getStr("net", "backup",FALSE);
      if(value==NULL)
         value=getStr("network", "backup",FALSE);
      return value;
   }


   LearningRatePolicy getLearningRatePolicy(char *group,char *key,char *def){
      char *value=getStr(group, key,FALSE);
      char s[64];
      sprintf(s,"%s",value?value:def);
      if (strcmp(s, "random")==0) return LearningRatePolicy.RANDOM;
      if (strcmp(s, "poly")==0) return LearningRatePolicy.POLY;
      if (strcmp(s, "constant")==0) return LearningRatePolicy.CONSTANT;
      if (strcmp(s, "step")==0) return LearningRatePolicy.STEP;
      if (strcmp(s, "exp")==0) return LearningRatePolicy.EXP;
      if (strcmp(s, "sigmoid")==0) return LearningRatePolicy.SIG;
      if (strcmp(s, "steps")==0) return LearningRatePolicy.STEPS;
      printf("Couldn't find policy %s, going with constant\n", s);
      return LearningRatePolicy.CONSTANT;
   }

   LearningRatePolicy getPolicy(char *s){
      if (strcmp(s, "random")==0) return LearningRatePolicy.RANDOM;
      if (strcmp(s, "poly")==0) return LearningRatePolicy.POLY;
      if (strcmp(s, "constant")==0) return LearningRatePolicy.CONSTANT;
      if (strcmp(s, "step")==0) return LearningRatePolicy.STEP;
      if (strcmp(s, "exp")==0) return LearningRatePolicy.EXP;
      if (strcmp(s, "sigmoid")==0) return LearningRatePolicy.SIG;
      if (strcmp(s, "steps")==0) return LearningRatePolicy.STEPS;
      printf("Couldn't find policy %s, going with constant\n", s);
      return LearningRatePolicy.CONSTANT;
   }

   ActivationType getActivation(char *group,char *key,char *def){
      char *value=getStr(group, key,FALSE);
      return Activation.getType(value?value:def);
   }

   aboolean isNetWork(){
      return (cfgFile->isFirstGroup("net") ||  cfgFile->isFirstGroup("network"));
   }

   /**
   * 第一个组的组名是不是指定的groupName
   */
   aboolean isFirstGroup(const char *groupName){
      return cfgFile->isFirstGroup(groupName);
   }

   char   **getGroups(asize *length){
      return cfgFile->getGroups(length);
   }

   aboolean removeGroup(const char *groupName,AError  **error){
      return cfgFile->removeGroup(groupName,error);
   }

   int getGroupCount(){
      return cfgFile->getGroupCount();
   }

   const char *getFileName(){
      return fileName;
   }

   //获得指定组索的key对应的值。
   char    *getValue(int groupIndex,char *key){
      char *ret=cfgFile->getValue(groupIndex,key,NULL);
      return ret;
   }

};

