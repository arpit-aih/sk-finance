import cv2, io
import numpy as np
from PIL import Image
from typing import List, Dict
from transformers import pipeline
from logger_config import get_logger
from pdf2image import convert_from_bytes


logger = get_logger("signature_extractor")


signature_pipe = pipeline(
    "object-detection",
    model="tech4humans/conditional-detr-50-signature-detector",
    framework="pt"
)


def file_bytes_to_pil_pages(content: bytes, filename: str) -> List[Image.Image]:

    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            try:
                pages = convert_from_bytes(content, dpi=200)
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


def extract_signatures_photos_from_pages(
    pages: List[Image.Image]
) -> Dict[int, List[bytes]]:

    results: Dict[int, List[bytes]] = {}

    for page_index, page in enumerate(pages):
        logger.info(f"[SignatureExtractor] Processing page {page_index}")

        page_rgb = page.convert("RGB")
        page_width, page_height = page_rgb.size

        try:
            detections = signature_pipe(page_rgb)
        except Exception as e:
            logger.exception(f"[SignatureExtractor] Model inference failed on page {page_index}: {e}")
            results[page_index] = []
            continue

        if not detections:
            logger.info(f"[SignatureExtractor] No detections found on page {page_index}")
            results[page_index] = []
            continue

        extracted_bytes: List[bytes] = []

        for det_idx, det in enumerate(detections):
            try:
                box = det.get("box", {})
                score = float(det.get("score", 0.0))
                label = det.get("label", "").lower()

                if "sign" not in label:
                    logger.debug(f"[SignatureExtractor] Ignored detection {det_idx} (label={label})")
                    continue

                if score < 0.5:
                    logger.debug(f"[SignatureExtractor] Low score ({score}) for detection {det_idx}")
                    continue

                xmin = float(box.get("xmin", 0.0))
                ymin = float(box.get("ymin", 0.0))
                xmax = float(box.get("xmax", 0.0))
                ymax = float(box.get("ymax", 0.0))

                left = max(0, min(page_width, xmin))
                top = max(0, min(page_height, ymin))
                right = max(0, min(page_width, xmax))
                bottom = max(0, min(page_height, ymax))

                if right - left < 2 or bottom - top < 2:
                    logger.warning(
                        f"[SignatureExtractor] Very small crop skipped on page {page_index}, det {det_idx}"
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
                        f"[SignatureExtractor] Crop/encode failed on page {page_index}, det {det_idx}: {e_crop}"
                    )
                    continue

            except Exception as e_det:
                logger.exception(
                    f"[SignatureExtractor] Error parsing detection on page {page_index}, det {det_idx}: {e_det}"
                )
                continue

        logger.info(
            f"[SignatureExtractor] Extracted {len(extracted_bytes)} signatures from page {page_index}"
        )

        results[page_index] = extracted_bytes

    return results
