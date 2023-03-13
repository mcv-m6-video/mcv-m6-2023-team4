import numpy as np


def bbox_iou(bboxA, bboxB):
    # Code provided by teacher in M1
    # compute the intersection over union of two bboxes
    
    # Format of the bboxes is [xtl, ytl, xbr, ybr, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[0], bboxB[0])
    
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[3] - bboxA[1] + 1) * (bboxA[2] - bboxA[0] + 1)
    bboxBArea = (bboxB[3] - bboxB[1] + 1) * (bboxB[2] - bboxB[0] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou


def calculate_mean_iou(frame):
    bboxes_pred = []
    bboxes_gt = []
    # iterate over each dictionary in the array
    for bbox in frame:
        if bbox['predicted']:
            # if predicted is True, append to the bboxes_pred list
            bbox['iou'] = 0
            bbox['detected'] = False
            bboxes_pred.append(bbox)
        else:
            # if predicted is False, append to the bboxes_gt list
            bboxes_gt.append(bbox)
    
    #generate mean iou
    mean_iou = 0
    used_indexes_pred = []
    iou_bboxes_gt = []
    
    for bbox_gt in bboxes_gt:
        bbox_coords_gt = [bbox_gt['xtl'], bbox_gt['ytl'], bbox_gt['xbr'], bbox_gt['ybr']]
        max_iou = 0
        max_index = -1
        for index, bbox_pred in enumerate(bboxes_pred):
            if index not in used_indexes_pred:
                bbox_coords_pred = [bbox_pred['xtl'], bbox_pred['ytl'], bbox_pred['xbr'], bbox_pred['ybr']]
                iou = bbox_iou(bbox_coords_gt, bbox_coords_pred)
                if iou > max_iou:
                    max_iou = iou
                    max_index = index
        
        # Each GT box can only be assigned to one predicted box
        if max_index != -1:
            iou_bboxes_gt.append(max_iou)
            used_indexes_pred.append(max_index)
            bboxes_pred[max_index]['detected'] = True
            bboxes_pred[max_index]['iou'] = max_iou
            
    for bbox in bboxes_pred:
        print('Bbox iou:', bbox['iou'])
        
    print(len(iou_bboxes_gt))
        
    mean_iou = np.array(iou_bboxes_gt).mean()

    return mean_iou, bboxes_gt, bboxes_pred


def generate_confidence(frame):
    for bbox in frame:
        bbox['confidence'] = np.random.uniform(0,1)
    return frame

def order_frame_by_confidence(frame):
    return sorted(frame, key=lambda x: x['confidence'], reverse=True)

def calculate_ap(frame_groundtruth, frame_preds, without_confidences=False):
    if without_confidences:
        N = 10
    else:
        N = 1
        
    iou_thresh = 0.5
    
    ap = 0
    
    for _ in range(N):
        # Generate confidence scores and sort
        if without_confidences:
            frame_preds = generate_confidence(frame_preds)
        sorted_frame_preds = order_frame_by_confidence(frame_preds)
        
        # Extract the confidence values in a thresholds list
        thresholds = [bbox['confidence'] for bbox in frame_preds]
        
        precisions = []
        recalls = []
        
        detected_bboxes = 0
        # Iterate for each threshold to find the precision and the recall
        for threshold in thresholds:
            true_positives = 0
            false_positives = 0

            for bbox in sorted_frame_preds:
                if bbox['detected']:
                    detected_bboxes += 1
                if bbox['confidence'] > threshold:
                    iou = bbox['iou']
                    if iou > iou_thresh:
                        true_positives += 1
                    else:
                        false_positives += 1
            
            #Check division by 0
            if true_positives + false_positives > 0:
                precisions.append(true_positives / (true_positives + false_positives))
            else:
                precisions.append(0)
            
            recalls.append(true_positives / len(frame_groundtruth))
        
        
        # If there are ground truth objects that were not detected, append a precision of 0 and a recall of 1
        if detected_bboxes < len(frame_groundtruth):
            precisions.append(0)
            recalls.append(1)
            
        ap += ap_pascal_VOC(recalls, precisions)

    return ap / N
        
    
def ap_pascal_VOC(recalls, precisions):
    index_recalls = len(recalls) - 2
    index_precisions = len(recalls) - 1
    average_precision = 0 
    for i in np.arange(1, -0.1, -0.1):
        if i < recalls[index_recalls]:
            if index_recalls != 0:
                index_recalls -= 1
                index_precisions -= 1
            elif index_recalls == 0 and index_precisions != 0:
                index_precisions -= 1

        average_precision += precisions[index_precisions]

    return average_precision / 11




