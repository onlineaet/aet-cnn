#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/System.h>
#include <aet/lang/AThread.h>

#include "DnnUtils.h"
#include "DstbCompute.h"

typedef struct _Future{
    int id;
    int count;
    int complete;
}Future;

class$ Task{
    int id;
    int count;
    int start;
    int end;
    aboolean pop;//是否弹出使用
    int orderNumber;//序号
    ComputingUnit *unit;
    apointer userData;
    Task(int orderNumber);
};

impl$ Task{
   Task(int orderNumber){
      self->orderNumber=orderNumber;
   }

   ~Task(){
       unit=NULL;
   }
};

/**
 * 被AThreadPool回调的运行块
 */
static void pool_cb (apointer data, apointer userData)
{
    aint64 time1=Time.monotonic();
    DstbCompute *self=(DstbCompute *)userData;
    Task *task=(Task *)data;
    ComputingUnit *unit=(ComputingUnit *)task->unit;
    AThread *thread=AThread.current();
    //a_debug("pool_cb is ---- %p task:%p ref:%d current:%p\n",unit,task,task->getRefCount(),AThread.current());
    unit->excuse(task->start,task->end,task->userData);
    self->complete(task->id,task->start,task->end);
    task->pop=FALSE;
    //task->unref();
    //printf("pool_cb is ----current:%p bindcpu:%d time:%lli\n",AThread.current(),thread->getBindCpu(),Time.monotonic()-time1);
}


impl$ DstbCompute{

   static DstbCompute *getInstance(){
      static DstbCompute *singleton = NULL;
      if (!singleton){
         singleton =new$ DstbCompute();
      }
      return singleton;
   }

   DstbCompute(){
      int cpuThreads=System.getCpuThreads();
      if(cpuThreads<=0)
         cpuThreads=12;
      AError *err=NULL;
      taskId=0;
      pool=new$ AThreadPool("ai_dstcompute",pool_cb,self,cpuThreads,FALSE,&err);
      if(err){
         a_error_free(err);
         a_warning("创建线程池失败。%s",err->message);
         return NULL;
      }
      int i;
      for(i=0;i<DstbCompute.MAX_TASK;i++){
         Task *task=new$ Task(i);
         tasks[i]=(void*)task;
      }

      lock=new$ AMutex();
      cond=new$ ACond();
      taskLock=new$ AMutex();
      taskHash =new$ AHashTable( AHashTable.directHash,AHashTable.directEqual,NULL,NULL);
   }

   Task *createTask(int id,int count,int start,int end,ComputingUnit *unit,apointer userData){
      int i;
      for(i=0;i<DstbCompute.MAX_TASK;i++){
         Task *task=(Task*)tasks[i];
         if(!task->pop){
             task->start=start;
             task->end=end;
             task->userData=userData;
             task->id=id;
             task->count=count;
             task->unit=unit;
             task->pop=TRUE;
             a_debug("加入unit 了:%p id:%d count:%d start:%d end:%d objectSize:%d\n",unit,id,count,start,end,task->objectSize);
             return task;
         }
      }
      a_error("找不到task可用！");
   }

   //顺序分配任务
   int addTask0(ComputingUnit *unit,apointer userData){
      int id=0;
      lock.lock();
      id=self->taskId++;
      int needThreadCount=unit->getNeedThreadCount();
      int circleCount=unit->getCircleCount();
      Future *future=a_slice_new(Future);
      future->id=id;
      future->count=circleCount;
      future->complete=0;
      taskHash->put(AINT_TO_POINTER(id),future);
      if(needThreadCount<=0)
         needThreadCount=pool->getMaxThreads();
      //printf("addTask 00 total:%d needThreadCount:%d id:%d\n",circleCount,needThreadCount,id);
      int avg=circleCount/needThreadCount;
      if(avg*needThreadCount<circleCount)
         avg+=1;
      int i;
      int start=0;
      int end=avg;
      AError *err=NULL;
      for(i=0;i<needThreadCount;i++){
         Task *task= createTask(id,circleCount,start,end,unit,userData);
         pool->push(task,&err);
         if(err)
            a_error("出错了--- %s\n",err->message);
         a_debug("addtask pool_cb is :task:%p taskId:%d userData:%p total:%d start:%d end:%d\n",task,id,userData,circleCount,start,end);
         start=end;
         if(start>=circleCount)
            break;
         if(end+avg>circleCount)
            end=circleCount;
         else
            end+=avg;
      }
      lock.unlock();
      return id;
   }

   //均匀分布的任务
   int addTask(ComputingUnit *unit,apointer userData){
      int id=0;
      lock.lock();
      id=self->taskId++;
      int needThreadCount=unit->getNeedThreadCount();
      int circleCount=unit->getCircleCount();
      Future *future=a_slice_new(Future);
      future->id=id;
      future->count=circleCount;
      future->complete=0;
      taskHash->put(AINT_TO_POINTER(id),future);
      if(needThreadCount<=0)
         needThreadCount=pool->getMaxThreads();
      AError *err=NULL;
      int start=0;
      int i;
      for (i = 0; i <needThreadCount; ++i) {
         int n = (i + 1) * circleCount /needThreadCount - i * circleCount / needThreadCount;
         Task *task= createTask(id,circleCount,start,start+n,unit,userData);
         pool->push(task,&err);
         if(err)
            a_error("出错了--- %s\n",err->message);
         a_debug("addtask pool_cb is :task:%p taskId:%d userData:%p total:%d start:%d end:%d\n",
               task,id,userData,circleCount,start,start+n);
         start+=n;
      }

      lock.unlock();
      return id;
   }

   void complete(int id,int start,int end){
      aint64 time1=Time.monotonic();
      lock.lock();
      Future *future= taskHash->get(AINT_TO_POINTER(id));
      future->complete+=(end-start);
      if(future->complete==future->count){
         taskHash->remove(AINT_TO_POINTER(id));
         a_slice_free(Future,future);
         cond.broadcast();
      }
      lock.unlock();
      // printf("complete id:%d start:%d end:%d future->complete:%d count:%d spendtime:%lli\n",
      //    id,start,end,future->complete,future->count,Time.monotonic()-time1);

   }

   void wait(int id){
      //aint64 time1=Time.monotonic();
      taskLock.lock();
      if(!taskHash->contains(AINT_TO_POINTER(id))){
         taskLock.unlock();
         //a_error("不会进这里，设计问题。id:%d",id);
         // printf("已执行完:%d\n",id);
         return;
      }
      // aint64 time2=Time.monotonic();
      cond.wait(&taskLock);
      // printf("wait ---- %lli %lli\n",Time.monotonic()-time1,time2-time1);
      taskLock.unlock();
   }

   void cancel(int id){

   }
};

