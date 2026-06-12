#ifndef __COM_AI_DNN_BUSINESS_H__
#define __COM_AI_DNN_BUSINESS_H__

#include <aet.h>

#include "NNetwork.h"
#include "ConfigFile.h"

package$ com.ai.dnn;
/**
 *深度学习的业务
 *有分类，检测以及....
 */
public$ abstract$ class$ Business{

   private$ NNetwork **nets;
   private$ int netCount;
   private$ ConfigFile *configFile;
   private$ char *dataCfg;
   private$ char *cfgFile;
   private$ char *weightFile;
   private$ char *backupDirectory;
   private$ char *baseFile;
   private$ aboolean clear;
   int argc;
   char **argv;
   protected$             Business(int ngpus,char *dataCfg,char *cfgFile,char *weightFile);
   public$ void           setClear(aboolean clear);
   public$ NNetwork      *getNetwork();
   public$ int            getNetCount();
   public$ ConfigFile    *getConfigFile();
   public$ abstract$ void run();
   public$ char          *getBackupDirectory();
   public$ char          *getBaseFile();
   public$ char          *getCfgFile();
   public$ char          *getDataCfgFile();
   public$ char          *getWeightFile();
   int                    findArg(char *arg);
   int                    findIntArg(char *arg, int def);
   float                  findFloatArg(char *arg, float def);
   char                  *findCharArg(char *arg, char *def);
};

#endif /* __N_MEM_H__ */

