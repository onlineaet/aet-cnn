
#include <locale.h>
#include <string.h>
#include <stdlib.h>
#include <aet/util/AKeyFile.h>
#include <aet/lang/AAssert.h>
#include <aet/lang/AAssert.h>
#include <com/ai/dnn/WorkFactory.h>
#include <com/ai/dnn/Detector.h>
#include <com/ai/dnn/Business.h>
#include <com/ai/dnn/DnnUtils.h>
#include <aet/mtcs/MtcsSystem.h>

int main (int argc, char *argv[])
{
    char *re=setlocale (LC_ALL, "");
    a_log_set_level(1);
    /**
     * 参数列表
     * -train      是否训练
     * -test       是否测试
     * -data   -d  数据文件
     * -clear      是否清除
     * -cfg    -c  配置文件
     * -weight -w  权重文件
     */
//    int argc_dnn=8;
//
//    char *parms[]={"aitest","classifier","train","-data",
//          "/home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar.data",
//          "-cfg","/home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar_small.cfg",
//          "-usemtcs"};


//    argc_dnn=10;
//    char *parms[]={"aitest","classifier","valid","-data",
//          "/home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar.data",
//          "-cfg","/home/xxx/aet-cnn-data/cfg/classifier/cifar/cifar_small.cfg",
//          "-weight","/home/xxx/aet-cnn-data/backup/classfier/cifar/darknet.weights",
//          "-usemtcs"};

   // Business *business=WorkFactory.getInstance()->build(argc_dnn,parms);
   // business->run();

    Business *business=WorkFactory.getInstance()->build(argc,argv);
    business->run();
    return 0;
}



