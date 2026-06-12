#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include  "Box.h"


impl$ Box{

   Box(float x,float y,float w,float h){
      self->x=x;
      self->y=y;
      self->w=w;
      self->h=h;
   }

   Box(float *data){
      self->x=data[0];
      self->y=data[1];
      self->w=data[2];
      self->h=data[3];
   }

   void  refresh(float *data){
      self->x=data[0];
      self->y=data[1];
      self->w=data[2];
      self->h=data[3];
   }

   void  copy(Box *dest){
      dest->x=x;
      dest->y=y;
      dest->w=w;
      dest->h=h;
   }

   float overlap(float x1, float w1, float x2, float w2){
      float l1 = x1 - w1/2;
      float l2 = x2 - w2/2;
      float left = l1 > l2 ? l1 : l2;
      float r1 = x1 + w1/2;
      float r2 = x2 + w2/2;
      float right = r1 < r2 ? r1 : r2;
      return right - left;
   }

   float intersection(Box *b){
      float w = overlap(self->x, self->w, b->x, b->w);
      float h = overlap(self->y, self->h, b->y, b->h);
      if(w < 0 || h < 0)
         return 0;
      float area = w*h;
      return area;
   }

   float boxUnion(Box *b){
      float i = intersection(b);
      float u = self->w*self->h + b->w*b->h - i;
      return u;
   }

   /**
   *与b的交集/与b的并集
   */
   float iou(Box *b){
      return intersection(b)/boxUnion(b);
   }

   // https://github.com/Zzh-tju/DIoU-darknet
   // https://arxiv.org/abs/1911.08287
   //原型 box_diou box.h box.c
   static float diou(Box a, Box b){
      boxabs ba = boxC(a, b);
      float w = ba.right - ba.left;
      float h = ba.bot - ba.top;
      float c = w * w + h * h;
      float iou = a.iou/*!box_iou*/(&b);
      if (c == 0)
         return iou;
      float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
      float u = pow(d / c, 0.6);
      float diou_term = u;
      #ifdef DEBUG_PRINTS
      printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
      #endif
      return iou - diou_term;
   }

   //原型 box_giou box.h box.c
   static float giou(Box a, Box b){
      boxabs ba = boxC(a, b);
      float w = ba.right - ba.left;
      float h = ba.bot - ba.top;
      float c = w*h;
      float iou = a.iou/*!box_iou*/(&b);
      if (c == 0)
         return iou;
      float u = a.boxUnion/*!box_union*/(&b);
      float giou_term = (c - u) / c;
   #ifdef DEBUG_PRINTS
      printf("  c: %f, u: %f, giou_term: %f\n", c, u, giou_term);
   #endif
      return iou - giou_term;
   }

   // https://github.com/Zzh-tju/DIoU-darknet
   // https://arxiv.org/abs/1911.08287
   //原型 box_ciou box.h box.c
   static float ciou(Box a, Box b){
      boxabs ba = boxC(a, b);
      float w = ba.right - ba.left;
      float h = ba.bot - ba.top;
      float c = w * w + h * h;
      float iou =a.iou/*!box_iou*/(&b);
      if (c == 0)
         return iou;
      float u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
      float d = u / c;
      float ar_gt = b.w / b.h;
      float ar_pred = a.w / a.h;
      float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
      float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
      float ciou_term = d + alpha * ar_loss;                   //ciou
   #ifdef DEBUG_PRINTS
      printf("  c: %f, u: %f, riou_term: %f\n", c, u, ciou_term);
   #endif
      return iou - ciou_term;
   }


   //原型 box_diounms box.c
   static float diounms(Box a, Box b, float beta1){
      boxabs ba = boxC(a, b);
      float w = ba.right - ba.left;
      float h = ba.bot - ba.top;
      float c = w * w + h * h;
      float iou = a.iou/*!box_iou*/(&b);
      if (c == 0) {
         return iou;
      }
      float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
      float u = pow(d / c, beta1);
      float diou_term = u;
   #ifdef DEBUG_PRINTS
      printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
   #endif
      return iou - diou_term;
   }

   // where c is the smallest box that fully encompases a and b
   //原型 box_c box.c
   static boxabs boxC(Box a, Box b) {
      boxabs ba = { 0 };
      ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
      ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
      ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
      ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
      return ba;
   }

   //原型 float_to_box_stride box.h box.c
   public$ static Box floatToBoxStride(float *f, int stride){
      Box b = new$ Box();
      b.x = f[0];
      b.y = f[1 * stride];
      b.w = f[2 * stride];
      b.h = f[3 * stride];
      return b;
   }

   // representation from x, y, w, h to top, left, bottom, right
   //原型 to_tblr box.h box.c
   boxabs toTblr() {
      boxabs tblr = { 0 };
      float t = y - (h / 2);
      float b = y + (h / 2);
      float l = x - (w / 2);
      float r = x + (w / 2);
      tblr.top = t;
      tblr.bot = b;
      tblr.left = l;
      tblr.right = r;
      return tblr;
   }

   //原型 dx_box_iou box.h box.c
   static dxrep dxBoxIou(Box pred, Box truth, IOU_LOSS iou_loss) {
      boxabs pred_tblr = pred.toTblr/*!to_tblr*/();
      float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
      float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
      float pred_l = fmin(pred_tblr.left, pred_tblr.right);
      float pred_r = fmax(pred_tblr.left, pred_tblr.right);
      //dbox dover = derivative(pred,truth);
      //dbox diouu = diou(pred, truth);
      boxabs truth_tblr =truth.toTblr/*!to_tblr*/();
   #ifdef DEBUG_PRINTS
      printf("\niou: %f, giou: %f\n", pred.iou(&truth), Box.giou(pred, truth));
      printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n",
            pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
      printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n",
            truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
   #endif
      //printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l, pred_r);
      //printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
      dxrep ddx = {0};
      float X = (pred_b - pred_t) * (pred_r - pred_l);
      float Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
      float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
      float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
      float I = Iw * Ih;
      float U = X + Xhat - I;
      float S = (pred.x-truth.x)*(pred.x-truth.x)+(pred.y-truth.y)*(pred.y-truth.y);
      float giou_Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
      float giou_Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
      float giou_C = giou_Cw * giou_Ch;
      //float IoU = I / U;
      //#ifdef DEBUG_PRINTS
      //printf("X: %f", X);
      //printf(", Xhat: %f", Xhat);
      //printf(", Ih: %f", Ih);
      //printf(", Iw: %f", Iw);
      //printf(", I: %f", I);
      //printf(", U: %f", U);
      //printf(", IoU: %f\n", I / U);
      //#endif

      //Partial Derivatives, derivatives
      float dX_wrt_t = -1 * (pred_r - pred_l);
      float dX_wrt_b = pred_r - pred_l;
      float dX_wrt_l = -1 * (pred_b - pred_t);
      float dX_wrt_r = pred_b - pred_t;
      // UNUSED
      //// Ground truth
      //float dXhat_wrt_t = -1 * (truth_tblr.right - truth_tblr.left);
      //float dXhat_wrt_b = truth_tblr.right - truth_tblr.left;
      //float dXhat_wrt_l = -1 * (truth_tblr.bot - truth_tblr.top);
      //float dXhat_wrt_r = truth_tblr.bot - truth_tblr.top;

      // gradient of I min/max in IoU calc (prediction)
      float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
      float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
      float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
      float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
      // derivative of U with regard to x
      float dU_wrt_t = dX_wrt_t - dI_wrt_t;
      float dU_wrt_b = dX_wrt_b - dI_wrt_b;
      float dU_wrt_l = dX_wrt_l - dI_wrt_l;
      float dU_wrt_r = dX_wrt_r - dI_wrt_r;
      // gradient of C min/max in IoU calc (prediction)
      float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * giou_Cw) : 0;
      float dC_wrt_b = pred_b > truth_tblr.bot ? giou_Cw : 0;
      float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * giou_Ch) : 0;
      float dC_wrt_r = pred_r > truth_tblr.right ? giou_Ch : 0;

      float p_dt = 0;
      float p_db = 0;
      float p_dl = 0;
      float p_dr = 0;
      if (U > 0 ) {
         p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
         p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
         p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
         p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
      }
      // apply grad from prediction min/max for correct corner selection
      p_dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
      p_db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
      p_dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
      p_dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

      if (iou_loss == GIOU) {
         if (giou_C > 0) {
            // apply "C" term from gIOU
            p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
            p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
            p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
            p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
         }
         if (Iw<=0||Ih<=0) {
            p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
            p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
            p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
            p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
         }
      }

      float Ct = fmin(pred.y - pred.h / 2,truth.y - truth.h / 2);
      float Cb = fmax(pred.y + pred.h / 2,truth.y + truth.h / 2);
      float Cl = fmin(pred.x - pred.w / 2,truth.x - truth.w / 2);
      float Cr = fmax(pred.x + pred.w / 2,truth.x + truth.w / 2);
      float Cw = Cr - Cl;
      float Ch = Cb - Ct;
      float C = Cw * Cw + Ch * Ch;

      float dCt_dx = 0;
      float dCt_dy = pred_t < truth_tblr.top ? 1 : 0;
      float dCt_dw = 0;
      float dCt_dh = pred_t < truth_tblr.top ? -0.5 : 0;

      float dCb_dx = 0;
      float dCb_dy = pred_b > truth_tblr.bot ? 1 : 0;
      float dCb_dw = 0;
      float dCb_dh = pred_b > truth_tblr.bot ? 0.5: 0;

      float dCl_dx = pred_l < truth_tblr.left ? 1 : 0;
      float dCl_dy = 0;
      float dCl_dw = pred_l < truth_tblr.left ? -0.5 : 0;
      float dCl_dh = 0;

      float dCr_dx = pred_r > truth_tblr.right ? 1 : 0;
      float dCr_dy = 0;
      float dCr_dw = pred_r > truth_tblr.right ? 0.5 : 0;
      float dCr_dh = 0;

      float dCw_dx = dCr_dx - dCl_dx;
      float dCw_dy = dCr_dy - dCl_dy;
      float dCw_dw = dCr_dw - dCl_dw;
      float dCw_dh = dCr_dh - dCl_dh;

      float dCh_dx = dCb_dx - dCt_dx;
      float dCh_dy = dCb_dy - dCt_dy;
      float dCh_dw = dCb_dw - dCt_dw;
      float dCh_dh = dCb_dh - dCt_dh;

      // UNUSED
      //// ground truth
      //float dI_wrt_xhat_t = pred_t < truth_tblr.top ? (-1 * Iw) : 0;
      //float dI_wrt_xhat_b = pred_b > truth_tblr.bot ? Iw : 0;
      //float dI_wrt_xhat_l = pred_l < truth_tblr.left ? (-1 * Ih) : 0;
      //float dI_wrt_xhat_r = pred_r > truth_tblr.right ? Ih : 0;

      // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
      float p_dx = 0;
      float p_dy = 0;
      float p_dw = 0;
      float p_dh = 0;

      p_dx = p_dl + p_dr;           //p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
      p_dy = p_dt + p_db;
      p_dw = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
      p_dh = (p_db - p_dt);

      // https://github.com/Zzh-tju/DIoU-darknet
      // https://arxiv.org/abs/1911.08287
      if (iou_loss == DIOU) {
         if (C > 0) {
            p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
            p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
         }
         if (Iw<=0||Ih<=0){
            p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
            p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
         }
      }
      //The following codes are calculating the gradient of ciou.

      if (iou_loss == CIOU) {
         float ar_gt = truth.w / truth.h;
         float ar_pred = pred.w / pred.h;
         float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
         float alpha = ar_loss / (1 - I/U + ar_loss + 0.000001);
         float ar_dw=8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.h;
         float ar_dh=-8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.w;
         if (C > 0) {
            // dar*
            p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
            p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
         }
         if (Iw<=0||Ih<=0){
            p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
            p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
         }
      }

      ddx.dt = p_dx;      //We follow the original code released from GDarknet. So in yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
      ddx.db = p_dy;
      ddx.dl = p_dw;
      ddx.dr = p_dh;
      return ddx;
   }

   //原型 box_iou_kind box.h box.c
   static float iouKind(Box a, Box b, IOU_LOSS iou_kind){
      //IOU, GIOU, MSE, DIOU, CIOU
      switch(iou_kind) {
         case IOU:
            return a.iou/*!box_iou*/(&b);
         case GIOU:
            return Box.giou/*!box_giou*/(a, b);
         case DIOU:
            return  Box.diou/*!box_diou*/(a, b);
         case CIOU:
            return  Box.ciou/*!box_ciou*/(a, b);
      }
      return a.iou/*!box_iou*/(&b);
   }

};

//原型 nms_comparator_v3 box.c
//pa *是 Detection **中的元素地址，地址中保存的值才是Detection *的指针地址。
static int nmsComparatorV3(const void *pa, const void *pb){
//   Detection *a = (Detection *)pa;
//   Detection *b = (Detection *)pb;
   unsigned long aadd;
   memcpy(&aadd,pa,sizeof(void*));
   unsigned long badd;
   memcpy(&badd,pb,sizeof(void*));

   Detection *a = (Detection *)aadd;
   Detection *b = (Detection *)badd;

   float diff = 0;
   if (b->sort_class >= 0) {
      diff = a->prob[b->sort_class] - b->prob[b->sort_class]; // there is already: prob = objectness*prob
   }else {
      diff = a->objectness - b->objectness;
   }
   if (diff < 0)
      return 1;
   else if (diff > 0)
      return -1;
   return 0;
}


impl$ Detection{

   //原型 do_nms_sort darknet.h box.c
   static void doNmsSort(Detection **dets, int total, int classes, float thresh){
      int i, j, k;
      k = total - 1;
      for (i = 0; i <= k; ++i) {
         if (dets[i]->objectness == 0) {
            Detection *swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
         }
      }
      total = k + 1;

      for (k = 0; k < classes; ++k) {
         for (i = 0; i < total; ++i) {
            dets[i]->sort_class = k;
         }
         qsort(dets, total, sizeof(Detection*), nmsComparatorV3);
         for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i]->prob[k] == 0)
               continue;
            Box a = dets[i]->bbox;
            for (j = i + 1; j < total; ++j) {
               Box b = dets[j]->bbox;
               if (a.iou(&b)/*!box_iou(a, b)*/ > thresh) {
                  dets[j]->prob[k] = 0;
               }
            }
         }
      }
   }


   // https://github.com/Zzh-tju/DIoU-darknet
   // https://arxiv.org/abs/1911.08287
   //原型 diounms_sort darknet.h darknet.c
   void diounmsSort(Detection **dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1){
       int i, j, k;
       k = total - 1;
       for (i = 0; i <= k; ++i) {
           if (dets[i]->objectness == 0) {
               Detection *swap = dets[i];
               dets[i] = dets[k];
               dets[k] = swap;
               --k;
               --i;
           }
       }
       total = k + 1;

       for (k = 0; k < classes; ++k) {
           for (i = 0; i < total; ++i) {
               dets[i]->sort_class = k;
           }
           qsort(dets, total, sizeof(Detection *), nmsComparatorV3);
           for (i = 0; i < total; ++i){
               if (dets[i]->prob[k] == 0)
                  continue;
               Box a = dets[i]->bbox;
               for (j = i + 1; j < total; ++j) {
                   Box b = dets[j]->bbox;
                   if (a.iou(&b)/*!box_iou(a, b)*/ > thresh && nms_kind == CORNERS_NMS){
                       float sum_prob = pow(dets[i]->prob[k], 2) + pow(dets[j]->prob[k], 2);
                       float alpha_prob = pow(dets[i]->prob[k], 2) / sum_prob;
                       float beta_prob = pow(dets[j]->prob[k], 2) / sum_prob;
                       //dets[i].bbox.x = (dets[i].bbox.x*alpha_prob + dets[j].bbox.x*beta_prob);
                       //dets[i].bbox.y = (dets[i].bbox.y*alpha_prob + dets[j].bbox.y*beta_prob);
                       //dets[i].bbox.w = (dets[i].bbox.w*alpha_prob + dets[j].bbox.w*beta_prob);
                       //dets[i].bbox.h = (dets[i].bbox.h*alpha_prob + dets[j].bbox.h*beta_prob);
                       /*
                       if (dets[j].points == YOLO_CENTER && (dets[i].points & dets[j].points) == 0) {
                           dets[i].bbox.x = (dets[i].bbox.x*alpha_prob + dets[j].bbox.x*beta_prob);
                           dets[i].bbox.y = (dets[i].bbox.y*alpha_prob + dets[j].bbox.y*beta_prob);
                       }
                       else if ((dets[i].points & dets[j].points) == 0) {
                           dets[i].bbox.w = (dets[i].bbox.w*alpha_prob + dets[j].bbox.w*beta_prob);
                           dets[i].bbox.h = (dets[i].bbox.h*alpha_prob + dets[j].bbox.h*beta_prob);
                       }
                       dets[i].points |= dets[j].points;
                       */
                       dets[j]->prob[k] = 0;
                   }
                   else if (Box.diou/*!box_diou*/(a, b) > thresh && nms_kind == GREEDY_NMS) {
                       dets[j]->prob[k] = 0;
                   }
                   else {
                       if (Box./*!box_diounms*/diounms(a, b, beta1) > thresh && nms_kind == DIOU_NMS) {
                           dets[j]->prob[k] = 0;
                       }
                   }
               }

               //if ((nms_kind == CORNERS_NMS) && (dets[i].points != (YOLO_CENTER | YOLO_LEFT_TOP | YOLO_RIGHT_BOTTOM)))
               //    dets[i].prob[k] = 0;
           }
       }
   }


   // Creates array of detections with prob > thresh and fills best_class for them
   //原型 get_actual_detections box.h box.c
   static detection_with_class* getActualDetections(Detection **dets, int dets_num,
      float thresh, int* selected_detections_num, char **names){
      int selected_num = 0;
      detection_with_class* result_arr = (detection_with_class*)xcalloc(dets_num, sizeof(detection_with_class));
      int i;
      for (i = 0; i < dets_num; ++i) {
         int best_class = -1;
         float best_class_prob = thresh;
         int j;
         for (j = 0; j < dets[i]->classes; ++j) {
            int show = strncmp(names[j], "dont_show", 9);
            if (dets[i]->prob[j] > best_class_prob && show) {
               best_class = j;
               best_class_prob = dets[i]->prob[j];
            }
         }
         if (best_class >= 0) {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
         }
      }
      if (selected_detections_num)
         *selected_detections_num = selected_num;
      return result_arr;
   }
};

