import io, base64, uvicorn, cv2, os, aiohttp, json
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from datetime import datetime
from typing import List, Tuple
from logger_config import get_logger
from fastapi.responses import JSONResponse
from azure_openai import get_model_response, get_signature_model_response
from photo_extractor import cv2_to_jpeg_bytes
from prompt import signature_prompt, photo_prompt
from azure_face import azure_detect_face, azure_verify
from fastapi import FastAPI, UploadFile, File, HTTPException
from photo_extractor import extract_passport_photos_from_pages
from signature_extractor import (
                  extract_signatures_photos_from_pages,
                  file_bytes_to_pil_pages
                  )
from cost_analysis import (
                           calculate_faceapi_cost, 
                           calculate_input_cost,
                           calculate_output_cost,
                           count_tokens,
                           calculate_gemini_cost
                           ) 


logger = get_logger("app")


app = FastAPI()


UPLOAD_PHOTOS_DIR = "uploads/photos"
UPLOAD_SIGNATURES_DIR = "uploads/signatures"
PHOTOS_CROPS_DIR = "extracted/photos"
SIGNATURES_CROPS_DIR = "extracted/signatures"

os.makedirs(UPLOAD_PHOTOS_DIR, exist_ok=True)
os.makedirs(UPLOAD_SIGNATURES_DIR, exist_ok=True)
os.makedirs(SIGNATURES_CROPS_DIR, exist_ok=True)
os.makedirs(PHOTOS_CROPS_DIR, exist_ok=True)


client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1beta'}
)


async def compare_signatures_gemini(sig1_bytes: bytes, sig2_bytes: bytes):
   
    try:
        # Load images from bytes
        img1 = Image.open(io.BytesIO(sig1_bytes))
        img2 = Image.open(io.BytesIO(sig2_bytes))
    except Exception as e:
        logger.error(f"Failed to load images for Gemini: {e}")
        return None
    
    # Define the prompt for structured output
    prompt = """
    You are a forensic document analysis assistant. 
    Analyze these two signature images. Compare their stroke flow, pen pressure indications, 
    slant, spacing, and letter formation.
    
    Determine if they are likely from the same person.
    
    Return a JSON object with exactly these fields:
    - "similarity_score": A float between 0 (completely different) and 100 (exact match).
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                prompt,
                img1,
                img2
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                temperature=0.1 
            )
        )
        
        # Parse the result
        result_json = json.loads(response.text)
        return result_json.get("similarity_score")
    except Exception as e:
        logger.error(f"Gemini API request failed: {e}")
        return None


def signature_evaluate_confidence(confidence_score) -> str:
    if confidence_score is None:
        return "rejected"

    return "accepted" if confidence_score >= 70 else "rejected"


def photo_evaluate_confidence(confidence_score) -> str:
    if confidence_score is None:
        return "rejected"

    return "accepted" if confidence_score >= 75 else "rejected"


@app.post("/verify-signature")
async def compare_signatures(
    file: List[UploadFile] = File(..., description="One or more PDFs or images containing signature(s)"),
    reference_signature: UploadFile = File(..., description="Reference signature image to compare against"),
):
    if not file or len(file) == 0:
        raise HTTPException(status_code=400, detail="No document uploaded")

    ref_bytes_raw = await reference_signature.read()
    if not ref_bytes_raw:
        raise HTTPException(status_code=400, detail="Empty reference signature")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    pages: List[Image.Image] = []
    pages_meta: List[Tuple[int, str, int]] = [] 

    for f_idx, upload in enumerate(file):
        try:
            content = await upload.read()
            if not content:
                continue

            # Save original file
            try:
                original_filepath = os.path.join(
                    UPLOAD_SIGNATURES_DIR, f"{timestamp}_{f_idx}_{upload.filename}"
                )
                with open(original_filepath, "wb") as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Failed saving uploaded file: {e}")

            # Convert to PIL pages
            try:
                upload_pages = file_bytes_to_pil_pages(content, upload.filename or f"upload_{f_idx}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse file '{upload.filename}': {e}")

            # Store pages + metadata
            for p_idx, p in enumerate(upload_pages):
                pages.append(p.convert("RGB"))
                pages_meta.append((f_idx, upload.filename or f"upload_{f_idx}", p_idx))

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error reading upload {upload.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    if not pages:
        raise HTTPException(status_code=400, detail="No pages or images detected in upload")

    try:
        ref_pages = file_bytes_to_pil_pages(ref_bytes_raw, reference_signature.filename or "reference")
        ref_page = ref_pages[0].convert("RGB")
        ref_arr = np.array(ref_page)
        ref_bgr = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2BGR)
        ref_jpeg = cv2_to_jpeg_bytes(ref_bgr, quality=95)

        # Save normalized reference signature
        ref_filepath = os.path.join(
            UPLOAD_SIGNATURES_DIR, f"{timestamp}_ref_{reference_signature.filename}.jpg"
        )
        with open(ref_filepath, "wb") as f:
            f.write(ref_jpeg)

    except Exception as e:
        logger.exception(f"Reference conversion failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to convert reference signature")

    try:
        extracted = extract_signatures_photos_from_pages(pages)
    except Exception as e:
        logger.exception(f"Signature extractor failed: {e}")
        raise HTTPException(status_code=500, detail="Signature extraction failed")

    def ensure_jpeg_bytes(item):
        if isinstance(item, (bytes, bytearray)):
            arr = np.frombuffer(item, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                pil = Image.open(io.BytesIO(item)).convert("RGB")
                arr = np.array(pil)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return cv2_to_jpeg_bytes(img, quality=95)

        if isinstance(item, Image.Image):
            arr = np.array(item.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return cv2_to_jpeg_bytes(bgr, quality=95)

        if isinstance(item, np.ndarray):
            return cv2_to_jpeg_bytes(item, quality=95)

        raise ValueError(f"Unsupported type {type(item)}")
    
    total_model_cost = 0
    total_gemini_calls = 0
    files_map = {}

    # Initialize maps
    for f_idx, fname, _ in pages_meta:
        files_map[f_idx] = {"file_index": f_idx, "filename": fname, "signatures": []}

    for page_idx in sorted(extracted.keys()):
        crops = extracted[page_idx]

        try:
            orig_file_idx, orig_filename, page_within_file = pages_meta[page_idx]
        except Exception:
            orig_file_idx, orig_filename, page_within_file = -1, "unknown", -1

        for sig_idx, crop_item in enumerate(crops):
            crop_b64 = None
            model_result = None
            similarity_score = None
            confidence_status = None

            try:
                # JPEG + Base64
                crop_jpeg = ensure_jpeg_bytes(crop_item)
                crop_b64 = base64.b64encode(crop_jpeg).decode("ascii")

                # Save crop
                crop_filename = f"{timestamp}_f{orig_file_idx}_sig{sig_idx}.jpg"
                crop_filepath = os.path.join(SIGNATURES_CROPS_DIR, crop_filename)
                with open(crop_filepath, "wb") as f:
                    f.write(base64.b64decode(crop_b64))

                try:
                    similarity_score = await compare_signatures_gemini(crop_jpeg, ref_jpeg)
                    total_gemini_calls += 1
                    if similarity_score is not None:
                        confidence_status = signature_evaluate_confidence(similarity_score)
                except Exception as e:
                    logger.error(f"Gemini comparison error: {e}")
                    similarity_score = None

                input_tokens = count_tokens(signature_prompt)
                total_model_cost += calculate_input_cost(input_tokens)

                # Model report
                try:
                    raw_result = await get_signature_model_response(
                        signature_prompt,
                        base64.b64encode(ref_jpeg).decode("ascii"),
                        crop_b64,
                    )
                    
                    output_tokens = count_tokens(str(raw_result))
                    total_model_cost += calculate_output_cost(output_tokens)

                    # try:
                    #     model_result = json.loads(raw_result)
                    # except Exception:
                    #     try:
                    #         cleaned = (
                    #             raw_result.strip()
                    #             .replace("```json", "")
                    #             .replace("```", "")
                    #             .strip()
                    #         )
                    #         model_result = json.loads(cleaned)
                    #     except Exception as parse_err:
                    #         logger.error(f"Failed to parse model JSON: {parse_err}")
                    #         model_result = None
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Model error: {e}")
                    model_result = None

            except Exception as e:
                logger.error(f"Error processing signature crop: {e}")

            # Append
            files_map[orig_file_idx]["signatures"].append({
                "signature_index": sig_idx,
                "confidence_score": similarity_score,
                "status": confidence_status,
                "report": raw_result,
                "image": crop_b64
            })

    try:
        gemini_cost = calculate_gemini_cost(total_gemini_calls)
    except Exception as e:
        logger.exception(f"Failed calculating gemini api cost: {e}")
        gemini_cost = 0

    try:
        total_cost = total_model_cost + gemini_cost
    except Exception as e:
        logger.exception(f"Failed calculating total cost: {e}")
        total_cost = total_model_cost

    response = {"total_cost": round(total_cost, 3),
        "files": []
        }

    # Final response structure
    for f_idx in sorted(files_map.keys()):
        response["files"].append(files_map[f_idx])

    return JSONResponse(response)


@app.post("/verify-photo")
async def compare_photos(
    file: List[UploadFile] = File(..., description="One or more images or PDFs containing passport/ID photos"),
    reference_photo: UploadFile = File(..., description="Reference photo image to compare against"),
):
    if not file or len(file) == 0:
        raise HTTPException(status_code=400, detail="No document uploaded")
    ref_raw_bytes = await reference_photo.read()
    if not ref_raw_bytes:
        raise HTTPException(status_code=400, detail="Reference photo is empty")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    pages: List[Image.Image] = []
    pages_meta: List[Tuple[int, str, int]] = [] 

    # Build pages + metadata
    for f_idx, upload in enumerate(file):
        try:
            content = await upload.read()
            if not content:
                logger.warning(f"Skipping empty upload: {upload.filename}")
                continue

            try:
                original_filepath = os.path.join(UPLOAD_PHOTOS_DIR, f"{timestamp}_{f_idx}_{upload.filename}")
                with open(original_filepath, "wb") as fh:
                    fh.write(content)
            except Exception as e:
                logger.exception(f"Failed to save original upload {upload.filename}: {e}")

            try:
                upload_pages = file_bytes_to_pil_pages(content, upload.filename or f"upload_{f_idx}")
            except Exception as e:
                logger.exception(f"Failed to convert uploaded file {upload.filename}: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file '{upload.filename}': {e}")

            for p_idx, p in enumerate(upload_pages):
                pages.append(p.convert("RGB"))
                pages_meta.append((f_idx, upload.filename or f"upload_{f_idx}", p_idx))

        except HTTPException:
            raise
        except Exception as e_outer:
            logger.exception(f"Unexpected error reading upload {getattr(upload, 'filename', None)}: {e_outer}")
            raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {e_outer}")

    if not pages:
        raise HTTPException(status_code=400, detail="No pages/images found in uploaded files")

    # Convert reference -> JPEG bytes
    try:
        ref_pages = file_bytes_to_pil_pages(ref_raw_bytes, reference_photo.filename or "reference")
        if not ref_pages:
            raise ValueError("No pages/images found in reference upload")
        ref_page = ref_pages[0].convert("RGB")
        ref_arr = np.array(ref_page)
        ref_bgr = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2BGR)
        ref_jpeg_bytes = cv2_to_jpeg_bytes(ref_bgr, quality=95)

        try:
            ref_filepath = os.path.join(UPLOAD_PHOTOS_DIR, f"{timestamp}_ref_{reference_photo.filename}.jpg")
            with open(ref_filepath, "wb") as fh:
                fh.write(ref_jpeg_bytes)
        except Exception as e:
            logger.exception(f"Failed to save converted reference image: {e}")

    except Exception as e:
        logger.exception(f"Failed to convert reference document to image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to convert reference photo to image: {e}")

    # Detect face in reference
    try:
        ref_face = await azure_detect_face(ref_jpeg_bytes)
    except Exception as e:
        logger.exception(f"Azure detect failed (reference): {e}")
        raise HTTPException(status_code=500, detail=f"Azure detect error (reference): {e}")

    if isinstance(ref_face, dict) and "error" in ref_face:
        logger.exception(f"Azure detect returned error (reference): {ref_face}")
        raise HTTPException(status_code=500, detail=f"Azure detect error (reference): {ref_face['error']}")

    ref_faceId = None
    try:
        ref_faceId = ref_face.get("faceId") if isinstance(ref_face, dict) else None
    except Exception:
        ref_faceId = None
    if not ref_faceId:
        raise HTTPException(status_code=400, detail="No face detected in reference photo")

    # Extract photos from pages
    try:
        extracted = extract_passport_photos_from_pages(pages)  # Dict[int, List[crop_items]]
    except Exception as e:
        logger.exception(f"Photo extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Photo extraction failed: {e}")

    # Helper to ensure jpeg bytes
    def ensure_jpeg_bytes(item) -> bytes:
        if isinstance(item, (bytes, bytearray)):
            arr = np.frombuffer(item, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                try:
                    pil = Image.open(io.BytesIO(item)).convert("RGB")
                    arr = np.array(pil)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    return cv2_to_jpeg_bytes(bgr, quality=95)
                except Exception:
                    raise ValueError("Could not decode returned image bytes")
            return cv2_to_jpeg_bytes(img, quality=95)

        if isinstance(item, Image.Image):
            arr = np.array(item.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return cv2_to_jpeg_bytes(bgr, quality=95)

        if isinstance(item, np.ndarray):
            return cv2_to_jpeg_bytes(item, quality=95)

        raise ValueError(f"Unsupported extracted photo type: {type(item)}")
    
    total_model_cost = 0
    total_faceapi_calls = 0

    # Build response: one entry per uploaded file, each with a flat photos array
    files_map = {}
    for f_idx, fname, _ in pages_meta:
        if f_idx not in files_map:
            files_map[f_idx] = {
                "file_index": f_idx,
                "filename": fname,
                "photos": []
            }


    # Iterate over extracted results (page_idx refers to pages list index)
    for page_idx in sorted(extracted.keys()):
        crops = extracted[page_idx]
        try:
            orig_file_idx, orig_filename, page_within_file = pages_meta[page_idx]
        except IndexError:
            orig_file_idx, orig_filename, page_within_file = (-1, "unknown", -1)
            logger.warning(f"Missing pages_meta for page_idx {page_idx}")

        for crop_idx, crop_item in enumerate(crops):
            model_result = None
            photo_b64 = None
            confidence_score = None
            confidence_status = None

            try:
                # normalize crop to jpeg bytes + base64
                try:
                    crop_jpeg = ensure_jpeg_bytes(crop_item)
                    photo_b64 = base64.b64encode(crop_jpeg).decode("ascii")
                except Exception as e:
                    logger.debug(f"Could not ensure jpeg bytes for crop (page {page_idx}, idx {crop_idx}): {e}")
                    if isinstance(crop_item, (bytes, bytearray)):
                        try:
                            photo_b64 = base64.b64encode(crop_item).decode("ascii")
                        except Exception:
                            photo_b64 = None
                    else:
                        photo_b64 = None

                if photo_b64:
                    crop_filename = f"{timestamp}_f{orig_file_idx}_photo{crop_idx}.jpg"
                    crop_filepath = os.path.join(PHOTOS_CROPS_DIR, crop_filename)
                    try:
                        with open(crop_filepath, "wb") as fh:
                            fh.write(base64.b64decode(photo_b64))
                    except Exception as e:
                        logger.exception(f"Failed saving crop to {crop_filepath}: {e}")


                    input_tokens = count_tokens(photo_prompt)
                    total_model_cost += calculate_input_cost(input_tokens)
                    # Call external model (if present)
                    try:
                        raw_result = await get_model_response(
                            photo_prompt,
                            base64.b64encode(ref_jpeg_bytes).decode("ascii"),
                            photo_b64
                        )
                        print("raw_result", raw_result)
                        output_tokens = count_tokens(raw_result)
                        total_model_cost += calculate_output_cost(output_tokens)

                        try:
                            model_result = json.loads(raw_result)
                        except Exception:
                            try:
                                cleaned = (
                                    raw_result.strip()
                                    .replace("```json", "")
                                    .replace("```", "")
                                    .strip()
                                )
                                model_result = json.loads(cleaned)
                            except Exception as parse_err:
                                logger.error(f"Failed to parse model JSON: {parse_err}")
                                model_result = None
                    except Exception as e:
                        logger.exception(f"Model response error for crop (page {page_idx}, idx {crop_idx}): {e}")
                        model_result = None

                    # Azure detect + verify
                    try:
                        crop_bytes = base64.b64decode(photo_b64)
                        crop_face = await azure_detect_face(crop_bytes)

                        if isinstance(crop_face, dict) and "error" in crop_face:
                            logger.exception(f"Azure detect returned error for crop (page {page_idx}, idx {crop_idx}): {crop_face}")
                            confidence_score = None

                        elif not (isinstance(crop_face, dict) and crop_face.get("faceId")):
                            logger.info(f"No face detected in crop (page {page_idx}, idx {crop_idx})")
                            confidence_score = None

                        else:
                            verify_result = await azure_verify(ref_faceId, crop_face["faceId"])
                            total_faceapi_calls += 1
                            if isinstance(verify_result, dict) and "error" in verify_result:
                                logger.exception(f"Azure verify returned error for crop (page {page_idx}, idx {crop_idx}): {verify_result}")
                                confidence_score = None

                            else:
                                confidence_score = verify_result.get("confidence") if isinstance(verify_result, dict) else None
                                if confidence_score is None:
                                    confidence_score = None

                                else:
                                    confidence_score = round(confidence_score * 100, 2)
                                    try:
                                        confidence_status = photo_evaluate_confidence(confidence_score)
                                    except Exception as e:
                                        logger.exception(f"evaluate_confidence failed: {e}")
                                        confidence_status = None

                    except Exception as e_azure:
                        logger.exception(f"Azure verification error for crop (page {page_idx}, idx {crop_idx}): {e_azure}")
                        confidence_score = None
                        confidence_status = None

                # Build photo entry and append directly under the file's photos list
                photo_entry = {
                    "photo_index": crop_idx,
                    "confidence_score": confidence_score,
                    "status": confidence_status,
                    "report": model_result,
                    "image": photo_b64
                }

                # Ensure file exists in map
                if orig_file_idx not in files_map:
                    files_map[orig_file_idx] = {
                        "file_index": orig_file_idx,
                        "filename": orig_filename,
                        "photos": []
                    }

                files_map[orig_file_idx]["photos"].append(photo_entry)

            except Exception as e_outer:
                logger.exception(f"Unexpected error processing crop (page {page_idx}, idx {crop_idx}): {e_outer}")
                photo_entry = {
                    "photo_index": crop_idx,
                    "confidence_score": None,
                    "status": None,
                    "report": None,
                    "image": None
                }
                if orig_file_idx not in files_map:
                    files_map[orig_file_idx] = {
                        "file_index": orig_file_idx,
                        "filename": orig_filename,
                        "photos": []
                    }
                files_map[orig_file_idx]["photos"].append(photo_entry)

    try:
        face_api_cost = calculate_faceapi_cost(total_faceapi_calls)
    except Exception as e_face_cost:
        logger.exception(f"Failed calculating face_api_cost: {e_face_cost}")
        face_api_cost = 0

    try:
        total_cost = total_model_cost + face_api_cost
    except Exception as e_total_cost:
        logger.exception(f"Failed calculating total_cost: {e_total_cost}")
        total_cost = total_model_cost

    response = {
    "total_cost": round(total_cost, 3),
    "files": []
}

    for f_idx in sorted(files_map.keys()):
        response["files"].append(files_map[f_idx])

    return JSONResponse(response)