import cv2, time
import numpy as np

class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0[:, :,]
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2; dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (left, top))

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

def Non_Max_Suppression(boxes, scores, iou_threshold):
    boxes, scores = np.array(boxes), np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    selected_boxes = []

    while len(order) > 0:
        i = order[0]
        selected_boxes.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        to_keep = np.where(iou <= iou_threshold)[0]
        order = order[to_keep + 1]
    return selected_boxes
    
def xywh2xyxy(x):
    y = np.empty_like(x, dtype = np.float32)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y

def Preprocess(images):
    letterbox = LetterBox([640, 640], auto=False, stride=32)
    im = np.array([np.ascontiguousarray(np.stack([letterbox(image=image)])[..., ::-1])[0] for image in images])
    im = im.transpose((0, 3, 1, 2)).astype(np.float32)/ 255.0
    return im.transpose((0, 2, 3, 1))

def Foward_propagation(im, interpreter):
    y = []
    start_time = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], im)
    interpreter.invoke()
    for output in interpreter.get_output_details():
        x = interpreter.get_tensor(output['index'])
        print('Inference:', round((time.time() - start_time) * 1000,1), 'ms')
        x[:, [0, 2]] *= 640; x[:, [1, 3]] *= 640
        y.append(x)
    if len(y) == 2:
        if len(y[1].shape) != 4:
            y = list(reversed(y))
        y[1] = np.transpose(y[1], (0, 3, 1, 2))
    return np.array([x if isinstance(x, np.ndarray) else x.numpy() for x in y])

def Postprocess(prediction, conf_thres=0.25, iou_thres=0.7, agnostic=False, labels=(), max_det=300, nc=80, max_time_img=0.05, max_nms=30000, max_wh=7680):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres
    time_limit = 0.5 + max_time_img * bs
    prediction = np.transpose(prediction, (0, 2, 1))
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = np.concatenate((x, v), axis=0)
        if not x.shape[0]:
            continue
        split_indices = (4, 4 + nc, 4 + nc + nm)
        box, cls, mask = x[:, :split_indices[0]], x[:, split_indices[0]:split_indices[1]], x[:, split_indices[1]:]
        conf = np.max(cls, axis=1, keepdims=True)
        j = np.argmax(cls, axis=1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(np.float32), mask), axis=1)[conf.flatten() > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = Non_Max_Suppression(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
    return output