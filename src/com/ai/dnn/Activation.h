#ifndef __COM_AI_DNN_ACTIVATION_H__
#define __COM_AI_DNN_ACTIVATION_H__

#include <aet.h>

#include <math.h>
#include "NData.h"


package$ com.ai.dnn;
/**
 * 激活函数的类型
 * 激活函数把数据变成非线性的类型
 */
public$ enum$ ActivationType{
   LOGISTIC, RELU, RELU6, RELIE, LINEAR, RAMP, TANH, PLSE, REVLEAKY, LEAKY, ELU,
   LOGGY, STAIR, HARDTAN, LHTAN, SELU, GELU, SWISH, MISH, HARD_MISH,
   NORM_CHAN, NORM_CHAN_SOFTMAX, NORM_CHAN_SOFTMAX_MAXVAL
} ;


public$ class$ Activation{

   public$ static ActivationType getType(char *str);
   public$ static char *getString(ActivationType type);
   public$ static float activate(float x, ActivationType type);
   public$ static void  activate(float *x,int n,const ActivationType type);
   public$ static float gradient(float x, ActivationType type);
   public$ static void  gradientArray(const float *x, const int n, const ActivationType type, float *delta);
   public$ static void  gradientArraySwish(const float *x, const int n, const float * sigmoid, float * delta);
   //原型 gradient_array_mish activations.h activations.c
   public$ static void  gradientArrayMish(const int n, const float * activation_input, float * delta);
   //原型 gradient_array_hard_mish activation.h activation.c
   public$ static void  gradientArrayHardMish(const int n, const float * activation_input, float * delta);
   public$ static void  gradientArrayNormalizeChannels(float *x, const int n, int batch, int channels, int wh_step, float *delta);
   public$ static void  gradientArrayNormalizeChannelsSoftmax(float *x, const int n, int batch, int channels, int wh_step, float *delta);

};


static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

static inline float softplus_activate(float x, float threshold)
{
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1);
}

#endif

