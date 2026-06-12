/*
 * cnnmicro.h
 *
 *  Created on: 2026年2月22日
 *      Author: sns
 */

#ifndef COM_AI_CNN_CNNMICRO_H_
#define COM_AI_CNN_CNNMICRO_H_

#include <sys/time.h>

#define SECRET_NUM -1234

#define  MTCS_BLOCK 512
#define  WARP_SIZE 32

extern int use_mtcs;

static inline double what_time_is_it_now()
{
   struct timeval time;
   if (gettimeofday(&time, NULL)) {
      return 0;
   }
   return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


#endif /* COM_AI_CNN_CNNMICRO_H_ */
