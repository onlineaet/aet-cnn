

#ifndef __COM_AI_DNN_COST_LAYER_H__
#define __COM_AI_DNN_COST_LAYER_H__

#include <aet.h>
#include  "NLayer.h"
#include  "Activation.h"
#include  "OutputLayer.h"

package$ com.ai.dnn;

public$ class$ CostLayer extends$ NLayer implements$ OutputLayer{

   public$ enum$ CostType{
      SSE, MASKED, L1, SEG, SMOOTH,WGAN
   };

   public$ static CostType getCostType(char *s){
      if (strcmp(s, "sse")==0)
         return SSE;
      if (strcmp(s, "masked")==0)
         return MASKED;
      if (strcmp(s, "smooth")==0)
         return SMOOTH;
      fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
      return SSE;
   }

   public$ static char *getCostString(CostType a){
      switch(a){
         case SSE:
            return "sse";
         case MASKED:
            return "masked";
         case SMOOTH:
            return "smooth";
         default:
            return "sse";
      }
   }

   CostType costType;
   float scale;
   float cost;
   float ratio;

   CostLayer(int batch, int inputs, CostType costType, float scale,float ratio);


};




#endif /* __N_MEM_H__ */

