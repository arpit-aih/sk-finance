# Photo and signature verification — API Documentation

This document describes the **two POST APIs** one for **photo verification** and another for **signature verification**.

In addition, both APIs return a **confidence score**, **verification status**, a **detailed verification report**, and the **extracted image** encoded in Base64 format

Clients can upload multiple images and PDF files.

---

## Table of Contents

1. Overview
3. `POST /verify-photo` and `POST /verify-signature`
   * Purpose
   * URL
   * Request format
   * Required fields
   * File constraints
   * Example request
   * Successful response (200 OK)
   * Field-by-field description
6. Error Handling (Complete List)
---

## 1. Overview

The `/verify-photo` and `/verify-signature` endpoints accept multiple images and PDF files and perform **photo verification** and **signature verification** respectively. Each endpoint returns a **confidence score**, **verification status**, a **detailed verification report**, and the **extracted image encoded in Base64 format** as part of a synchronous response.

The endpoint responds with `200 OK` and a structured payload containing:

* Verification status
* Confidence score
* Detailed verification report
* Extracted image encoded in Base64 format

All verification is completed within the request–response cycle, and results are returned directly in the API response.

---

## 2. POST `/verify-photo` and `/verify-signature`

### Purpose

Perform **verification** of uploaded documents against a provided **reference image**.

/verify-photo verifies **passport / ID photos**
/verify-signature verifies **handwritten signatures**

Each request is processed immediately and returns verification results directly in the response.

### URL

```http
POST http://127.0.0.1:8000/verify-photo
POST http://127.0.0.1:8000/verify-signature
```

### Content Type

* `multipart/form-data`

---

### Required Fields

#### Common Fields (Both Endpoints)


| Field                                             | Type   |      Description                                                     |
| --------------------------------------------      | ------ |      --------------------------------------------------------------- |
| `file[]`                                          | File[] |      One or more image or PDF files containing photos or signatures. |
| `reference_photo` /`reference_signature`          | File   |      Reference image used for verification.                          |


Use reference_photo for `/verify-photo`
Use reference_signature for `/verify-signature`

---

### File Constraints

* **Supported formats**: PDF, JPEG, PNG
* **Multiple files** are supported per request
* Empty files or documents with no detectable photos/signatures will result in `400 Bad Request`

---

### Example `curl` Request — Photo Verification

```bash
curl -X POST "http://127.0.0.1:8000/verify-photo" \
 -F "file[]=@/path/to/id1.pdf" \
  -F "file[]=@/path/to/id2.jpg" \
  -F "reference_photo=@/path/to/reference.jpg"
```

### Example `curl` Request — Signature Verification

```bash
curl -X POST "http://127.0.0.1:8000/verify-signature" \
  -F "file[]=@/path/to/document1.pdf" \
  -F "file[]=@/path/to/document2.png" \
  -F "reference_signature=@/path/to/reference_signature.jpg"
```

---

### Successful Response -- `/verify-photo` (200 OK)

```json
{
  "total_cost": 0.004,
  "files": [
    {
      "file_index": 0,
      "filename": "pan.png",
      "photos": [
        {
          "photo_index": 0,
          "confidence_score": 9.61,
          "status": "rejected",
          "report": {
            "pixel_difference": [
              "Background color differs: reference image has a white background, document image has a blue background.",
              "Clothing color and collar style differ: reference image shows a green and white collared shirt, document image shows a blue shirt.",
              "Lighting and image quality differ: reference image is clearer and brighter, document image is darker and lower resolution."
            ],
            "jawline_shape": "Reference image shows a wider jawline with a more rounded contour, while the document image appears to have a narrower jawline with less visible contour due to lower image quality.",
            "forehead_proportion": "Reference image displays a lower hairline and a shorter visible forehead, while the document image shows a higher hairline and a taller visible forehead.",
            "lip_shape_thickness": "Lip details are not clearly visible in the document image due to blur and resolution; reference image suggests fuller lips with a more defined outline.",
            "nose_bridge_width": "Nose bridge and width are not distinctly visible in the document image; reference image suggests a broader nose base, while the document image appears to have a narrower nose due to image quality."
          },
          "image": "<BASE64_ENCODED_IMAGE>"
        }
      ]
    }
  ]
}
```

### Successful Response -- `/verify-signature` (200 OK)

```json
{
  "total_cost": 0.336,
  "files": [
    {
      "file_index": 0,
      "filename": "kj4.png",
      "signatures": [
        {
          "signature_index": 0,
          "confidence_score": 95.0,
          "status": "accepted",
          "report": {
            "pixel_difference": [
              {
                "region": "central and upper right portions of the signature",
                "description": "The document image appears lighter and more blurred compared to the reference image, resulting in less defined edges and lower contrast in the signature lines."
              },
              {
                "region": "lower right stroke ending",
                "description": "The document image's horizontal stroke ends slightly earlier and is less sharp than in the reference image."
              }
            ],
            "stroke_points": {
              "reference_signature": [
                {
                  "stroke": 1,
                  "start_point": [60, 140],
                  "end_point": [420, 110],
                  "curvature_points": [[120, 120], [250, 100], [350, 110]],
                  "description": "Main elongated stroke forming the base and lower loop of the signature."
                }
              ],
              "document_signature": [
                {
                  "stroke": 1,
                  "start_point": [65, 150],
                  "end_point": [430, 120],
                  "curvature_points": [[130, 130], [260, 110], [360, 120]],
                  "description": "Main elongated stroke forming the base and lower loop of the signature, but appears lighter and less defined."
                }
              ],
              "differences": [
                {
                  "type": "stroke definition",
                  "description": "All strokes in the document image are lighter, less defined, and appear more blurred compared to the reference image."
                }
              ]
            }
          },
          "image": "<BASE64_ENCODED_IMAGE>"
        }
      ]
    }
  ]
}
```

---

### Field-by-Field Description

#### Top-Level Fields

| Field             | Type   |      Description                                             |
| ------------      | ------ |      ------------------------------------------------------- |
| `total_cost`      | number |      Combined cost of all model and verification operations. |
| `files`           | array  |      Results grouped by uploaded file.                       |

---

#### `files[]` Object

| Field                           | Type   |      Description                               |
| --------------------------      | ------ |      ----------------------------------------- |
| `file_index`                    | number |      Index of the uploaded file.    |
| `filename`                      | string |      Original uploaded file name.              |
| `photos` / `signatures`         | array  |      Verification results per extracted image. |

---

#### `photos[]` / `signatures[]` Entry

| Field                                     | Type   |      Description                                                     |
| ------------------------------------      | ------ |      --------------------------------------------------------------- |
| `photo_index` / `signature_index`         | number |      Index of the extracted image within the file.                   |
| `confidence_score`                        | number |      Verification confidence score. |
| `status`                                  | string |      Verification status derived from the confidence score.          |
| `report`                                  | object |      Detailed verification report.            |
| `image`                                   | string |      Extracted photo or signature encoded in Base64 format.          |

---


## 3. Error Handling (Complete List)

The `/verify-photo` and `/verify-signature` endpoints return standard HTTP error responses when input validation or processing fails.


### No Files Uploaded

```json
{
  "status": 400,
  "detail": "No document uploaded"
}
```

### Empty Reference Image

```json
{
  "status": 400,
  "detail": "Reference photo is empty"
}
```

or (for signatures):

```json
{
  "status": 400,
  "detail": "Empty reference signature"
}
```

### Unsupported or Invalid File Format

```json
{
  "status": 400,
  "detail": "Failed to parse uploaded file '<filename>'"
}
```
