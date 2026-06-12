#ifndef __COM_AI_DNN_MTCS_GEMM_H__
#define __COM_AI_DNN_MTCS_GEMM_H__

#include <aet.h>
#include <aet/util/AArray.h>
#include "../Gemm.h"


package$ com.ai.dnn.mtcs;

public$ class$ MtcsGemm extends$ Gemm{

   static  void */*!cublasHandle_t*/blasHandle;
   private$ static void *getBlasHandle();

};

#endif /* __COM_AI_DNN_MTCS_GEMM_H__ */

