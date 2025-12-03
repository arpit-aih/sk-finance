import io, base64, uvicorn, cv2, os
import numpy as np
from logger_config import get_logger
from PIL import Image
from typing import List
from datetime import datetime
from pdf2image import convert_from_bytes
from fastapi.responses import JSONResponse
from azure_openai import get_model_response
from photo_extractor import cv2_to_jpeg_bytes
from prompt import signature_prompt, photo_prompt
from azure_face import azure_detect_face, azure_verify
from fastapi import FastAPI, UploadFile, File, HTTPException
from photo_extractor import extract_passport_photos_from_pages
from signature_extractor import extract_signatures_photos_from_pages


logger = get_logger("app")

UPLOAD_PHOTOS_DIR = "uploads/photos"
UPLOAD_SIGNATURES_DIR = "uploads/signatures"
PHOTOS_CROPS_DIR = "extracted/photos"
SIGNATURES_CROPS_DIR = "extracted/signatures"

os.makedirs(UPLOAD_PHOTOS_DIR, exist_ok=True)
os.makedirs(UPLOAD_SIGNATURES_DIR, exist_ok=True)
os.makedirs(SIGNATURES_CROPS_DIR, exist_ok=True)
os.makedirs(PHOTOS_CROPS_DIR, exist_ok=True)


app = FastAPI()


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
    

def bytes_to_cv2_img(b: bytes) -> np.ndarray:
    try:
        arr = np.frombuffer(b, np.uint8)
        try:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            return img

        except Exception as decode_err:
            raise ValueError(f"Failed to decode image bytes with OpenCV: {decode_err}")

    except Exception as err:
        raise RuntimeError(f"Error converting bytes to cv2 image: {err}")


def ensure_jpeg_bytes(item) -> bytes:
        
        if isinstance(item, (bytes, bytearray)):
            arr = np.frombuffer(item, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode returned image bytes")
            return cv2_to_jpeg_bytes(img, quality=90)
        if isinstance(item, Image.Image):
            arr = np.array(item.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return cv2_to_jpeg_bytes(bgr, quality=90)
        if isinstance(item, np.ndarray):
            return cv2_to_jpeg_bytes(item, quality=90)
        raise ValueError(f"Unsupported extracted photo type: {type(item)}")


# @app.post("/verify-signature")
# async def compare_signatures(
#     file: UploadFile = File(..., description="PDF or image containing signature(s)"),
#     reference_signature: UploadFile = File(..., description="Reference signature image to compare against"),
# ):
#     content = await file.read()
#     ref_bytes = await reference_signature.read()

#     if not content:
#         raise HTTPException(status_code=400, detail="Uploaded file is empty")

#     if not ref_bytes:
#         raise HTTPException(status_code=400, detail="Empty reference signature upload")

#     try:
#         pages = file_bytes_to_pil_pages(content, file.filename or "upload")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to parse upload: {e}")

#     try:
#         ref_cv2 = bytes_to_cv2_img(ref_bytes)
#         ref_jpeg = cv2_to_jpeg_bytes(ref_cv2, quality=90)
#         ref_base64 = base64.b64encode(ref_jpeg).decode("ascii")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to decode reference image: {e}")

#     try:
#         extraction_results = extract_signatures_photos_from_pages(
#             pages
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Signature extractor error: {e}")

#     def ensure_jpeg_bytes(item) -> bytes:
#         if isinstance(item, (bytes, bytearray)):
#             arr = np.frombuffer(item, np.uint8)
#             img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#             if img is None:
#                 raise ValueError("Could not decode returned image bytes")
#             return cv2_to_jpeg_bytes(img, quality=90)

#         if isinstance(item, Image.Image):
#             arr = np.array(item.convert("RGB"))
#             bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#             return cv2_to_jpeg_bytes(bgr, quality=90)

#         if isinstance(item, np.ndarray):
#             return cv2_to_jpeg_bytes(item, quality=90)

#         raise ValueError(f"Unsupported extracted crop type: {type(item)}")

#     response = {"images": []}

#     for page_index in sorted(extraction_results.keys()):
#         crops = extraction_results[page_index]
#         page_output = {"image_index": page_index, "signatures": []}

#         for ci, crop_item in enumerate(crops):
#             crop_b64 = None
#             model_result = None

#             try:
#                 try:
#                     crop_jpeg = ensure_jpeg_bytes(crop_item)
#                     crop_b64 = base64.b64encode(crop_jpeg).decode("ascii")
#                 except Exception:
#                     if isinstance(crop_item, (bytes, bytearray)):
#                         try:
#                             crop_b64 = base64.b64encode(crop_item).decode("ascii")
#                         except Exception:
#                             crop_b64 = None

#                 if crop_b64:
#                     try:
#                         model_result = await get_model_response(
#                             signature_prompt,
#                             ref_base64,
#                             crop_b64,
#                         )
#                     except Exception as e:
#                         print("Confidence score error:", e)
#                         model_result = None

#                     try:
#                         confidence_score = await get_confidence_score(
#                             ref_base64,
#                             crop_b64
#                         )
#                     except Exception as e:
#                         print("Confidence score error:", e)
#                         confidence_score = None

#                 page_output["signatures"].append({
#                     "image_index": ci,
#                     "report": model_result,
#                     "confidence_score": confidence_score,
#                     "image": crop_b64,
#                 })

#             except Exception as e:
#                 # Hard failure for this crop
#                 page_output["signatures"].append({
#                     "image_index": ci,
#                     "report": None,
#                     "similarity_score": None,
#                     "image": None,
#                 })

#         response["images"].append(page_output)

#     return JSONResponse(response)



@app.post("/verify-signature")
async def compare_signatures(
    file: UploadFile = File(..., description="PDF or image containing signature(s)"),
    reference_signature: UploadFile = File(..., description="Reference signature image to compare against"),
):
    content = await file.read()
    ref_bytes = await reference_signature.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if not ref_bytes:
        raise HTTPException(status_code=400, detail="Empty reference signature upload")

    # Save original uploaded files
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original_filepath = os.path.join(UPLOAD_SIGNATURES_DIR, f"{timestamp}_{file.filename}")
    with open(original_filepath, "wb") as f:
        f.write(content)

    ref_filepath = os.path.join(UPLOAD_SIGNATURES_DIR, f"{timestamp}_{reference_signature.filename}")
    with open(ref_filepath, "wb") as f:
        f.write(ref_bytes)

    try:
        pages = file_bytes_to_pil_pages(content, file.filename or "upload")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse upload: {e}")

    try:
        extraction_results = extract_signatures_photos_from_pages(pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signature extractor error: {e}")

    def ensure_jpeg_bytes(item) -> bytes:
        if isinstance(item, (bytes, bytearray)):
            arr = np.frombuffer(item, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode returned image bytes")
            return cv2_to_jpeg_bytes(img, quality=90)

        if isinstance(item, Image.Image):
            arr = np.array(item.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return cv2_to_jpeg_bytes(bgr, quality=90)

        if isinstance(item, np.ndarray):
            return cv2_to_jpeg_bytes(item, quality=90)

        raise ValueError(f"Unsupported extracted crop type: {type(item)}")

    response = {"images": []}

    for page_index in sorted(extraction_results.keys()):
        crops = extraction_results[page_index]
        page_output = {"image_index": page_index, "signatures": []}

        for ci, crop_item in enumerate(crops):
            crop_b64 = None
            model_result = None

            try:
                try:
                    crop_jpeg = ensure_jpeg_bytes(crop_item)
                    crop_b64 = base64.b64encode(crop_jpeg).decode("ascii")
                except Exception:
                    if isinstance(crop_item, (bytes, bytearray)):
                        try:
                            crop_b64 = base64.b64encode(crop_item).decode("ascii")
                        except Exception:
                            crop_b64 = None

                if crop_b64:
                    crop_filename = f"{timestamp}_page{page_index}_sig{ci}.jpg"
                    crop_filepath = os.path.join(SIGNATURES_CROPS_DIR, crop_filename)
                    with open(crop_filepath, "wb") as f:
                        f.write(base64.b64decode(crop_b64))

                    try:
                        model_result = await get_model_response(
                            signature_prompt,
                            base64.b64encode(ref_bytes).decode("ascii"),
                            crop_b64,
                        )
                    except Exception as e:
                        logger.exception(f"Model response error: {e}")
                        model_result = None

                page_output["signatures"].append({
                    "image_index": ci,
                    "report": model_result,
                    "image": crop_b64,
                })

            except Exception as e:
                page_output["signatures"].append({
                    "image_index": ci,
                    "report": None,
                    "image": None,
                })

        response["images"].append(page_output)

    return JSONResponse(response)


@app.post("/verify-photo")
async def compare_photos(
    file: UploadFile = File(..., description="PDF or image containing passport/ID photos"),
    reference_photo: UploadFile = File(..., description="Reference face/photo image to compare against"),
):

    content = await file.read()
    ref_bytes = await reference_photo.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty")
    if not ref_bytes:
        raise HTTPException(status_code=400, detail="Reference photo is empty")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original_filepath = os.path.join(UPLOAD_PHOTOS_DIR, f"{timestamp}_{file.filename}")
    with open(original_filepath, "wb") as f:
        f.write(content)

    ref_filepath = os.path.join(UPLOAD_PHOTOS_DIR, f"{timestamp}_{reference_photo.filename}")
    with open(ref_filepath, "wb") as f:
        f.write(ref_bytes)

    try:
        pages = file_bytes_to_pil_pages(content, file.filename or "upload")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse document: {e}")
    
    ref_face = await azure_detect_face(ref_bytes)
    if "error" in ref_face:
        raise HTTPException(status_code=500, detail=f"Azure detect error (reference): {ref_face['error']}")
    ref_faceId = ref_face.get("faceId")
    if not ref_faceId:
        raise HTTPException(status_code=400, detail="No face detected")

    try:
        extracted = extract_passport_photos_from_pages(pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Photo extraction failed: {e}")

    def ensure_jpeg_bytes(item) -> bytes:
        if isinstance(item, (bytes, bytearray)):
            arr = np.frombuffer(item, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode returned image bytes")
            return cv2_to_jpeg_bytes(img, quality=90)

        if isinstance(item, Image.Image):
            arr = np.array(item.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return cv2_to_jpeg_bytes(bgr, quality=90)

        if isinstance(item, np.ndarray):
            return cv2_to_jpeg_bytes(item, quality=90)

        raise ValueError(f"Unsupported extracted photo type: {type(item)}")

    response = {"photos": []}

    for page_index in sorted(extracted.keys()):
        photos = extracted[page_index]
        page_entry = {"photo_index": page_index, "photos": []}

        for pi, photo_item in enumerate(photos):
            model_result = None
            photo_b64 = None
            confidence_score = None

            try:
                try:
                    photo_jpeg = ensure_jpeg_bytes(photo_item)
                    photo_b64 = base64.b64encode(photo_jpeg).decode("ascii")
                except Exception:
                    if isinstance(photo_item, (bytes, bytearray)):
                        try:
                            photo_b64 = base64.b64encode(photo_item).decode("ascii")
                        except Exception:
                            photo_b64 = None

                if photo_b64:
                    crop_filename = f"{timestamp}_page{page_index}_photo{pi}.jpg"
                    crop_filepath = os.path.join(PHOTOS_CROPS_DIR, crop_filename)
                    with open(crop_filepath, "wb") as f:
                        f.write(base64.b64decode(photo_b64))

                    try:
                        model_result = await get_model_response(photo_prompt, 
                                                          base64.b64encode(ref_bytes).decode("ascii"), 
                                                          photo_b64)
                    except Exception as e:
                        logger.exception(f"Model response error: {e}")
                        model_result = None

                    try:
                        crop_bytes = base64.b64decode(photo_b64)
                        crop_face = await azure_detect_face(crop_bytes)

                        if "error" in crop_face:
                            confidence_score = None
                        elif not crop_face.get("faceId"):
                            confidence_score = None
                        else:
                            verify_result = await azure_verify(ref_faceId, crop_face["faceId"])
                            if "error" in verify_result:
                                confidence_score = None
                            else:
                                confidence_score = verify_result.get("confidence")
                                confidence_score = round(confidence_score * 100, 2) if confidence_score is not None else None 

                    except Exception as e:
                        logger.exception(f"Azure verification error: {e}")
                        confidence_score = None

                    page_entry["photos"].append({
                        "photo_index": pi,
                        "report": model_result,
                        "confidence_score": confidence_score,
                        "image": photo_b64
                    })
                    
                page_entry["photos"].append({
                    "photo_index": pi,
                    "report": model_result,
                    "confidence_score": confidence_score,
                    "image": photo_b64
                })

            except Exception:
                page_entry["photos"].append({
                    "photo_index": pi,
                    "report": None,
                    "confidence_score": None,
                    "image": None,
                })

        response["photos"].append(page_entry)

    return JSONResponse(response)
