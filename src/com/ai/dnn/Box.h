

#ifndef __COM_AI_DNN_BOX_H__
#define __COM_AI_DNN_BOX_H__

#include <aet.h>

package$ com.ai.dnn;

typedef enum {
    DEFAULT_NMS, GREEDY_NMS, DIOU_NMS, CORNERS_NMS
} NMS_KIND;

// parser.h
typedef enum {
    IOU, GIOU, MSE, DIOU, CIOU
} IOU_LOSS;

// box.h
typedef struct boxabs {
    float left, right, top, bot;
}boxabs;

typedef struct dxrep {
    float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
    float iou, giou, diou, ciou;
    dxrep dx_iou;
    dxrep dx_giou;
} ious;

public$ class$ Box{
    protected$ float  x,y,w,h;
    public$ Box(float x,float y,float w,float h);
    public$ Box(float *data);
    public$ float iou(Box *b);
    public$ float intersection(Box *b);
    public$ float boxUnion(Box *b);
    public$ void  refresh(float *data);
    public$ void  copy(Box *dest);
    //原型 to_tblr box.h box.c
    public$ boxabs toTblr();
    //原型 box_c box.c
    public$ static boxabs boxC(Box a, Box b);
    //原型 box_diounms box.c
    public$ static float diounms(Box a, Box b, float beta1);
    //原型 box_diou box.h box.c
    public$ static float diou(Box a, Box b);
    //原型 float_to_box_stride box.h box.c
    public$ static Box floatToBoxStride(float *f, int stride);
    //原型 box_giou box.h box.c
    public$  static float giou(Box a, Box b);
    //原型 box_ciou box.h box.c
    public$ static float ciou(Box a, Box b);
    //原型 dx_box_iou box.h box.c
    public$ static dxrep dxBoxIou(Box pred, Box truth, IOU_LOSS iou_loss);
    //原型 box_iou_kind box.h box.c
    public$ static float iouKind(Box a, Box b, IOU_LOSS iou_kind);
};

typedef struct _detection_with_class detection_with_class;


public$ class$ Detection{
   Box bbox;
   int classes;
   int best_class_idx;
   float *prob;
   float *mask;
   float objectness;
   int sort_class;
   float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
   int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
   float *embeddings;  // embeddings for tracking
   int embedding_size;
   float sim;
   int track_id;
   //原型 darknet.h box.c
   public$ static void doNmsSort(Detection **dets, int total, int classes, float thresh);
   public$ static void diounmsSort(Detection **dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);
   //原型 get_actual_detections box.h box.c
   public$ static detection_with_class* getActualDetections(Detection **dets, int dets_num,
      float thresh, int* selected_detections_num, char **names);
};

struct _detection_with_class {
    Detection *det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
};



#endif /* __N_MEM_H__ */

