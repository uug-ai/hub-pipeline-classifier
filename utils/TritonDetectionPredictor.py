import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class TritonDetectionPredictor(DetectionPredictor):
    """Post-process raw YOLO outputs and end-to-end Triton detections."""

    def postprocess(self, preds, img, orig_imgs):
        prediction = preds[0] if isinstance(preds, (list, tuple)) else preds
        if prediction.ndim != 3 or prediction.shape[-1] != 6:
            return super().postprocess(preds, img, orig_imgs)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        class_filter = None
        if self.args.classes is not None:
            class_filter = torch.tensor(self.args.classes, device=prediction.device)

        results = []
        for index, detections in enumerate(prediction):
            detections = detections[detections[:, 4] >= self.args.conf]
            if class_filter is not None:
                detections = detections[(detections[:, 5:6] == class_filter).any(1)]
            detections = detections[:self.args.max_det].clone()

            orig_img = orig_imgs[index]
            detections[:, :4] = ops.scale_boxes(img.shape[2:], detections[:, :4], orig_img.shape)
            img_path = self.batch[0][index]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=detections))

        return results