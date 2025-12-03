import requests, os, httpx
from dotenv import load_dotenv


load_dotenv()


AZURE_FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT")  
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")


if not AZURE_FACE_ENDPOINT or not AZURE_FACE_KEY:
    raise RuntimeError("Set AZURE_FACE_ENDPOINT and AZURE_FACE_KEY environment variables")


HEADERS_IMG = {"Ocp-Apim-Subscription-Key": AZURE_FACE_KEY, "Content-Type": "application/octet-stream"}
HEADERS_JSON = {"Ocp-Apim-Subscription-Key": AZURE_FACE_KEY, "Content-Type": "application/json"}


import requests

async def azure_detect_face(image_bytes: bytes, recognition_model: str = "recognition_04"):
    try:
        url = f"{AZURE_FACE_ENDPOINT}/face/v1.2/detect"
        params = {
            "returnFaceId": "true",
            "returnFaceLandmarks": "false",
            "recognitionModel": recognition_model,
            "returnRecognitionModel": "false",
        }

        try:
            resp = requests.post(
                url,
                params=params,
                headers=HEADERS_IMG,   # should include Content-Type: application/octet-stream and key
                data=image_bytes,
                timeout=30,
            )
        except Exception as req_err:
            return {"error": f"Request to Azure failed: {req_err}"}

        if resp.status_code != 200:
            return {"error": f"Azure detect failed: {resp.status_code} {resp.text}"}

        try:
            data = resp.json()
        except Exception as json_err:
            return {"error": f"Failed to parse Azure response JSON: {json_err}"}

        if not isinstance(data, list) or len(data) == 0:
            return {"faceId": None, "faceRectangle": None}

        face = data[0]
        return {
            "faceId": face.get("faceId"),
            "faceRectangle": face.get("faceRectangle"),
        }

    except Exception as err:
        return {"error": f"Unexpected error in azure_detect_face: {err}"}

 

async def azure_verify(faceId1: str, faceId2: str):
    try:
        url = f"{AZURE_FACE_ENDPOINT}/face/v1.2/verify"  # adjust if AZURE_FACE_ENDPOINT already has /face/v1.2
        payload = {"faceId1": faceId1, "faceId2": faceId2}

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    url,
                    headers=HEADERS_JSON,
                    json=payload,
                )
            except Exception as req_err:
                return {"error": f"Request to Azure failed: {req_err}"}

        if resp.status_code != 200:
            return {"error": f"Azure verify failed: {resp.status_code} {resp.text}"}

        try:
            return resp.json()  # {"isIdentical": bool, "confidence": float}
        except Exception as json_err:
            return {"error": f"Failed to parse Azure verify response JSON: {json_err}"}

    except Exception as err:
        return {"error": f"Unexpected error in azure_verify: {err}"}