

#ifndef __COM_AI_DNN_YOLO_LAYER_H__
#define __COM_AI_DNN_YOLO_LAYER_H__

#include <aet.h>
#include <aet/util/AMutex.h>
#include  "NLayer.h"
#include  "DeltaData.h"
#include  "OutputLayer.h"
#include  "ClassesIface.h"
#include  "Box.h"
#include "DstbCompute.h"

package$ com.ai.dnn;

/**
 */
public$ class$ YoloLayer extends$ NLayer  implements$ OutputLayer,ClassesIface,ComputingUnit{

    AMutex lock;
    int anchorNumbers; //当前使用的先验框个数
    private$ float anchors[100];
    private$ int anchorsCount;//总的先验框个数
    private$ int maxBoxes;
    float ignoreThresh;
    float truthThresh;
    int *map;
    private$   int *mask;
    private$   int truths;
    protected$ int classes ;//分类的数量
    private$ float cost;
    private$  int truth_size;
    private$ int *labels;
    private$ int *class_ids;
    float show_details;
    float max_delta;
    float *classes_multipliers;
    float label_smooth_eps;
    float scale_x_y;
    int objectness_smooth;
    int new_coords;
    float iou_normalizer;
    float obj_normalizer;
    float cls_normalizer;
    float delta_normalizer;
    IOU_LOSS iou_loss;
    IOU_LOSS iou_thresh_kind;
    NMS_KIND nms_kind;
    float beta_nms;
    int resize;
    int focal_loss;
    float ignore_thresh;
    float truth_thresh;
    float iou_thresh;
    float random;
    int track_history_size;
    float sim_thresh;
    int dets_for_track;
    int dets_for_show;
    float track_ciou_norm;
    int embedding_layer_id;
    float *embedding_output;
    int embedding_size;
    int max_boxes;

    public$ YoloLayer(int batch, int w, int h, int n, int total, int *mask, int classes);
    public$ YoloLayer(int batch, int w, int h, int anchorNumbers, int total, int *mask, int classes, int max_boxes);

    public$ void setAnchors(float *value,int len);
    public$ void setMaxBoxes(int max);
    public$ void setIgnoreThresh(float thresh);
    public$ void setTruthThresh(float thresh);
    public$ void setMap(int *map);
    //原型 yolo_num_detections yolo_layer.h yolo_layer.c
    public$ int numDetections(float thresh);
    //原型 correct_yolo_boxes yolo_layer.h yolo_layer.c
    void correctBoxes(Detection **dets, int n, int w, int h, int netw, int neth, int relative, int letter);
    //原型 get_yolo_detections yolo_layer.h yolo_layer.c
    int getDetections(int w, int h, int netw, int neth, float thresh,
          int *map, int relative, Detection **dets, int letter);
};




#endif /* __N_MEM_H__ */

