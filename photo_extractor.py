import cv2, torch
import numpy as np
from PIL import Image
from typing import List, Dict
from ultralytics import YOLOE
from logger_config import get_logger


logger = get_logger("photo_extractor")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
yolo_model = YOLOE("yoloe-11l-seg.pt")
yolo_model.to(device=device)


_YOLO_CLASSES = ["person"]
yolo_model.set_classes(_YOLO_CLASSES, yolo_model.get_text_pe(_YOLO_CLASSES))


def cv2_to_jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    try:
        try:
            ok, enc = cv2.imencode(
                ".jpg",
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
        except Exception as enc_err:
            raise RuntimeError(f"OpenCV JPEG encoding error: {enc_err}")

        if not ok:
            raise RuntimeError("JPEG encoding failed (cv2.imencode returned False)")

        return enc.tobytes()

    except Exception as err:
        raise RuntimeError(f"Error in cv2_to_jpeg_bytes: {err}")


def extract_passport_photos_from_pages(
    pages: List[Image.Image]
) -> Dict[int, List[bytes]]:

    results: Dict[int, List[bytes]] = {}

    for page_index, page in enumerate(pages):
        logger.info(f"[PassportExtractor] Processing page {page_index}")

        try:
            page_rgb = page.convert("RGB")
            page_width, page_height = page_rgb.size
            page_np = np.array(page_rgb)

        except Exception as e:
            logger.exception(f"[PassportExtractor] Failed to convert page {page_index} to RGB/NumPy: {e}")
            results[page_index] = []
            continue

        try:
            preds = yolo_model.predict(page_np, verbose=False)
        except Exception as e:
            logger.exception(f"[PassportExtractor] YOLO inference failed for page {page_index}: {e}")
            results[page_index] = []
            continue

        if not preds:
            logger.info(f"[PassportExtractor] No predictions for page {page_index}")
            results[page_index] = []
            continue

        pred = preds[0]

        if not hasattr(pred, "boxes") or pred.boxes is None or len(pred.boxes) == 0:
            logger.info(f"[PassportExtractor] No bounding boxes found on page {page_index}")
            results[page_index] = []
            continue

        try:
            boxes_xyxy = pred.boxes.xyxy
            classes = pred.boxes.cls
            confs = getattr(pred.boxes, "conf", None)

            boxes_xyxy = boxes_xyxy.cpu().numpy()
            classes = classes.cpu().numpy().astype(int)
            confs_np = confs.cpu().numpy() if confs is not None else None

        except Exception as e:
            logger.exception(f"[PassportExtractor] Failed to parse prediction tensors on page {page_index}: {e}")
            results[page_index] = []
            continue

        extracted_bytes: List[bytes] = []

        for det_idx, ((x_min, y_min, x_max, y_max), cls_id) in enumerate(zip(boxes_xyxy, classes)):
            try:
                class_name = pred.names[int(cls_id)] if hasattr(pred, "names") else None
            except Exception:
                class_name = None

            if class_name != "person":
                logger.debug(f"[PassportExtractor] Ignored detection {det_idx} (class={class_name})")
                continue

            if confs_np is not None and confs_np[det_idx] < 0.5:
                logger.debug(
                    f"[PassportExtractor] Low confidence ({confs_np[det_idx]}) for detection {det_idx} on page {page_index}"
                )
                continue

            left = max(0, min(page_width, x_min))
            top = max(0, min(page_height, y_min))
            right = max(0, min(page_width, x_max))
            bottom = max(0, min(page_height, y_max))

            if right - left < 2 or bottom - top < 2:
                logger.warning(
                    f"[PassportExtractor] Very small crop skipped for det {det_idx} on page {page_index}"
                )
                continue

            try:
                cropped = page_rgb.crop((left, top, right, bottom))

                arr = np.array(cropped)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                jpeg_bytes = cv2_to_jpeg_bytes(bgr, quality=90)

                extracted_bytes.append(jpeg_bytes)

            except Exception as e_crop:
                logger.exception(
                    f"[PassportExtractor] Crop/encode failed for det {det_idx} on page {page_index}: {e_crop}"
                )
                continue

        logger.info(
            f"[PassportExtractor] Extracted {len(extracted_bytes)} images from page {page_index}"
        )

        results[page_index] = extracted_bytes

    return results
