

#ifndef __COM_AI_CONFIG_FILE_H__
#define __COM_AI_CONFIG_FILE_H__

#include <aet.h>

#include <aet/util/AKeyFile.h>
#include "Activation.h"
#include "NNetwork.h"


package$ com.ai.dnn;


public$ class$ ConfigFile{

    private$ AKeyFile *cfgFile;
    private$ char *fileName;
    ConfigFile(char *fileName);
    char    *getStr(char *group, char *key,aboolean quiet);
    char    *getStr(char *group, char *key,char *def);
    int      optionFindInt(char *group, char *key, int def,aboolean quiet);
    float    optionFindFloat(char *group, char *key, float def,aboolean quiet);
    int      optionFindInt(char *group, char *key, int def);
    float    optionFindFloat(char *group, char *key, float def);
    int      optionFindIntQuiet(char *group, char *key, int def);
    float    optionFindFloatQuiet(char *group, char *key, float def);
    LearningRatePolicy getLearningRatePolicy(char *group,char *key,char *def);
    LearningRatePolicy getPolicy(char *value);
    ActivationType getActivation(char *group,char *key,char *def);
    char    *getBackup();//权重存放的路径
    aboolean isNetWork();
    //代理AKeyFile的功能
    aboolean isFirstGroup(const char *groupName);
    char   **getGroups(asize *length);
    aboolean removeGroup(const char *groupName,AError  **error);
    int      getGroupCount();
    const char *getFileName();
    //获得指定组索的key对应的值。
    char    *getValue(int groupIndex,char *key);

};




#endif /* __N_MEM_H__ */

