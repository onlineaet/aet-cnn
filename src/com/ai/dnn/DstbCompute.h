

#ifndef __COM_AI_DSTB_COMPUTE_H__
#define __COM_AI_DSTB_COMPUTE_H__

#include <aet.h>
#include "Detector.h"
#include  <aet/util/AThreadPool.h>
#include  <aet/util/AHashTable.h>
#include  <aet/util/AMutex.h>
#include  <aet/util/ACond.h>
#include "ComputingUnit.h"


package$ com.ai.dnn;


/**
 * 异构分布计算，加入GPU
 */
public$ class$ DstbCompute{
    public$ static int MAX_TASK=256;
    public$ static DstbCompute *getInstance();
    private$ AThreadPool *pool;
    private$ AMutex lock;
    private$ ACond  cond;
    private$ AMutex taskLock;
    private$ AHashTable *taskHash;
    private$ void *tasks[MAX_TASK];//任务组数，保存有预先创建的tassk
    private$ int taskId;
    private$ void complete(int id,int start,int end);
    public$  void wait(int id);
    public$  int addTask(ComputingUnit *unit,apointer userData);
};




#endif /* __N_MEM_H__ */

