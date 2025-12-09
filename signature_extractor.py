import cv2, os, time, fitz, io
import onnxruntime as ort
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
from huggingface_hub import hf_hub_download
from logger_config import get_logger


logger = get_logger("signature_extractor")


REPO_ID = "tech4humans/yolov8s-signature-detector"
FILENAME = "yolov8s.onnx"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")


def file_bytes_to_pil_pages(content: bytes, filename: str, dpi: int = 200) -> List[Image.Image]:
    
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            try:
                doc = fitz.open(stream=content, filetype="pdf")

                pages: List[Image.Image] = []
                scale = dpi / 72.0
                mat = fitz.Matrix(scale, scale)

                for page in doc:
                    pix = page.get_pixmap(matrix=mat, alpha=False)

                    mode = "RGB"
                    try:
                        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                    except Exception:
                        img = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")

                    pages.append(img)

                doc.close()
                return pages

            except Exception as pdf_err:
                raise ValueError(f"Failed to convert PDF to images: {pdf_err}")

        else:
            try:
                pil = Image.open(io.BytesIO(content)).convert("RGB")
                return [pil]
            except Exception as img_err:
                raise ValueError(f"Failed to open image file: {img_err}")

    except Exception as err:
        raise RuntimeError(f"Error processing file '{filename}': {err}")
    

def download_model() -> str:
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        logger.info(f"Downloading model from {REPO_ID} ...")
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODEL_DIR,
            cache_dir=None,
            token=os.getenv("HF_TOKEN"),
        )

        # Move to canonical MODEL_PATH if necessary
        if os.path.exists(model_path) and os.path.abspath(model_path) != os.path.abspath(MODEL_PATH):
            os.replace(model_path, MODEL_PATH)
            # cleanup potential empty directories created by hub
            empty_dir = os.path.join(MODEL_DIR, "tune")
            if os.path.exists(empty_dir):
                import shutil
                shutil.rmtree(empty_dir)
        logger.info("Model downloaded successfully")
        return MODEL_PATH
    except Exception:
        logger.exception("Error downloading model")
        raise


class SignatureDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.classes = ["signature"]
        self.input_width = 640
        self.input_height = 640

        # Simple in-memory metrics (no DB)
        self.total_inferences: int = 0
        self.last_inference_time_ms: float = 0.0

        # Initialize ONNX Runtime session
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Create session; if model_path doesn't exist, raise early
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run download_model() first.")
        self.session = ort.InferenceSession(self.model_path, options)
        # Attempt to set best provider; fallback to default providers if not available
        try:
            self.session.set_providers(["OpenVINOExecutionProvider"], [{"device_type": "CPU"}])
        except Exception:
            logger.warning("OpenVINO provider not available, using default ONNX Runtime providers")

    def update_metrics(self, inference_time_ms: float) -> None:
        """Update in-memory inference metrics (no DB)."""
        self.total_inferences += 1
        self.last_inference_time_ms = float(inference_time_ms)

    def get_metrics(self) -> dict:
        """Return simple in-memory metrics."""
        avg_time = self.last_inference_time_ms if self.total_inferences == 1 else None
        # keep simple: return last and total; user can extend if needed
        return {
            "total_inferences": self.total_inferences,
            "last_inference_time_ms": self.last_inference_time_ms,
            "avg_time_ms": avg_time,
        }

    def preprocess(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        # Convert PIL Image to cv2 BGR for drawing and keep original dims
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.img_height, self.img_width = img_cv2.shape[:2]

        # Convert to RGB for resize/prep
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        image_data = np.array(img_resized) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data, img_cv2

    def postprocess_raw(
        self,
        output: np.ndarray,
        conf_thres: float,
        iou_thres: float,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw ONNX output to list of detection dicts:
         {xmin,ymin,xmax,ymax,score,class_id,label}
        Coordinates are in original image space (not resized-space).
        """
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= conf_thres:
                class_id = int(np.argmax(classes_scores))
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)

        # Apply NMS (OpenCV expects lists)
        detections: List[Dict[str, Any]] = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
            # Normalize indices to a flat list of ints
            if isinstance(indices, (list, tuple)):
                idxs = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]
            else:
                try:
                    idxs = indices.flatten().astype(int).tolist()
                except Exception:
                    idxs = [int(indices)]
            for idx in idxs:
                left, top, width, height = boxes[idx]
                xmin = float(max(0, left))
                ymin = float(max(0, top))
                xmax = float(min(self.img_width, left + width))
                ymax = float(min(self.img_height, top + height))
                detections.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "score": float(scores[idx]),
                    "class_id": int(class_ids[idx]),
                    "label": self.classes[int(class_ids[idx])] if int(class_ids[idx]) < len(self.classes) else str(class_ids[idx])
                })
        return detections

    def predict_raw(self, image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.5) -> List[Dict[str, Any]]:
        
        img_data, original_image = self.preprocess(image)
        start_time = time.time()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        inference_time_ms = (time.time() - start_time) * 1000.0
        self.update_metrics(inference_time_ms)
        detections = self.postprocess_raw(outputs, conf_thres, iou_thres)
        return detections

    def postprocess(self, input_image: np.ndarray, output: np.ndarray, conf_thres: float, iou_thres: float) -> np.ndarray:
        """Keep drawing method in case you need it; no DB/plot dependencies."""
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= conf_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            # draw on image
            x1, y1, w, h = box
            color = (0, 255, 0)
            cv2.rectangle(input_image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    def detect(self, image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.5) -> Tuple[Image.Image, dict]:
        """Backwards-compatible detect that returns an image with drawn boxes and simple metrics."""
        img_data, original_image = self.preprocess(image)
        start_time = time.time()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        inference_time = (time.time() - start_time) * 1000
        output_image = self.postprocess(original_image, outputs, conf_thres, iou_thres)
        self.update_metrics(inference_time)
        return output_image, self.get_metrics()

    def detect_example(self, image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.5) -> Image.Image:
        output_image, _ = self.detect(image, conf_thres, iou_thres)
        return output_image
    

# ---------- Utility: JPEG encode helper used in your original code ----------

def cv2_to_jpeg_bytes(bgr: np.ndarray, quality: int = 90) -> bytes:
    success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()


# ---------- Extraction function (uses SignatureDetector.predict_raw) ----------

def extract_signatures_photos_from_pages(
    pages: List[Image.Image],
    detector: Optional[SignatureDetector] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    padding: int = 0,
) -> Dict[int, List[bytes]]:
    
    if detector is None:
        # ensure model exists (download if necessary)
        if not os.path.exists(MODEL_PATH):
            download_model()
        detector = SignatureDetector(MODEL_PATH)

    results: Dict[int, List[bytes]] = {}

    for page_index, page in enumerate(pages):
        logger.info(f"Processing page {page_index}")
        page_rgb = page.convert("RGB")
        page_width, page_height = page_rgb.size

        try:
            detections = detector.predict_raw(page_rgb, conf_thres=conf_threshold, iou_thres=iou_threshold)
        except Exception:
            logger.exception(f"Model inference failed on page {page_index}")
            results[page_index] = []
            continue

        if not detections:
            logger.info(f"No signature found on page {page_index}")
            results[page_index] = []
            continue

        extracted_bytes: List[bytes] = []

        for det_idx, det in enumerate(detections):
            try:
                score = float(det.get("score", 0.0))
                label = (det.get("label", "") or "").lower()

                if "sign" not in label and "signature" not in label:
                    logger.debug(f"Ignored detection {det_idx} (label={label})")
                    continue

                if score < conf_threshold:
                    logger.debug(f"Low score ({score}) for detection {det_idx}, threshold {conf_threshold}")
                    continue

                xmin = float(det.get("xmin", 0.0))
                ymin = float(det.get("ymin", 0.0))
                xmax = float(det.get("xmax", 0.0))
                ymax = float(det.get("ymax", 0.0))

                # apply padding and clamp coordinates
                left = int(max(0, xmin - padding))
                top = int(max(0, ymin - padding))
                right = int(min(page_width, xmax + padding))
                bottom = int(min(page_height, ymax + padding))

                if right - left < 2 or bottom - top < 2:
                    logger.warning(f"Very small crop skipped on page {page_index}, det {det_idx}")
                    continue

                try:
                    cropped = page_rgb.crop((left, top, right, bottom))
                    arr = np.array(cropped)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    jpeg_bytes = cv2_to_jpeg_bytes(bgr, quality=90)
                    extracted_bytes.append(jpeg_bytes)
                except Exception:
                    logger.exception(f"Crop/encode failed on page {page_index}, det {det_idx}")
                    continue

            except Exception:
                logger.exception(f"Error parsing detection on page {page_index}, det {det_idx}")
                continue

        logger.info(f"Extracted {len(extracted_bytes)} signatures from page {page_index}")
        results[page_index] = extracted_bytes

    return results