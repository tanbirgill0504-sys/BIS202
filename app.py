import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import unquote_plus

# Make matplotlib use Lambda's writable /tmp folder
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

TMP_DIR = Path("/tmp")
INPUT_DIR = TMP_DIR / "input"
OUTPUT_DIR = TMP_DIR / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLOV5_DIR = os.environ.get("YOLOV5_DIR", "/var/task/yolov5")
MODEL_PATH = os.environ.get("MODEL_PATH", "/var/task/yolov5s.pt")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "output")
MODEL_CONFIDENCE = float(os.environ.get("MODEL_CONFIDENCE", "0.25"))

# Helps Python find YOLOv5 internal modules like models/ and utils/
if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL = None
MODEL_LOAD_SECONDS = None


def get_model():
    """
    Load YOLOv5 only when the Lambda function is invoked.
    The loaded model is reused on warm Lambda invocations.
    """
    global MODEL
    global MODEL_LOAD_SECONDS

    if MODEL is not None:
        return MODEL

    logger.info("Loading YOLOv5 model from %s", MODEL_PATH)

    model_load_start = time.time()

    MODEL = torch.hub.load(
        YOLOV5_DIR,
        "custom",
        path=MODEL_PATH,
        source="local",
        force_reload=False,
        device="cpu"
    )

    MODEL.conf = MODEL_CONFIDENCE
    MODEL.eval()

    MODEL_LOAD_SECONDS = round(time.time() - model_load_start, 3)

    logger.info("YOLOv5 model loaded successfully in %s seconds", MODEL_LOAD_SECONDS)

    return MODEL


def make_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body, default=str)
    }


def parse_event(event):
    """
    Supports:
    1. Lambda direct test event:
       {
         "bucket": "bis202-yolov5",
         "key": "input/sample.jpg"
       }

    2. API Gateway event with body:
       {
         "body": "{\"bucket\":\"bis202-yolov5\",\"key\":\"input/sample.jpg\"}"
       }

    3. S3 trigger event.
    """

    if "Records" in event and event["Records"]:
        record = event["Records"][0]

        if "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
            return bucket, key

    if "body" in event and event["body"]:
        body = event["body"]

        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")

        if isinstance(body, str):
            body = json.loads(body)

        bucket = body.get("bucket")
        key = body.get("key")

    else:
        bucket = event.get("bucket")
        key = event.get("key")

    if not bucket:
        raise ValueError("Missing bucket. Example: bis202-yolov5")

    if not key:
        raise ValueError("Missing key. Example: input/sample.jpg")

    return bucket, key


def validate_input_key(key):
    extension = Path(key).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .jpg, .jpeg, and .png images are supported.")

    if not key.startswith("input/"):
        raise ValueError("Image must be inside the input/ folder. Example: input/sample.jpg")


def download_image(bucket, key):
    validate_input_key(key)

    extension = Path(key).suffix.lower()
    local_path = INPUT_DIR / f"{uuid.uuid4()}{extension}"

    logger.info("Downloading s3://%s/%s to %s", bucket, key, local_path)

    s3.download_file(bucket, key, str(local_path))

    return local_path


def run_detection(image_path):
    """
    Runs YOLOv5 object detection.

    Important:
    This function does NOT use results.render().
    results.render() can fail in Lambda/OpenCV because the NumPy image array
    may be read-only. So we manually draw bounding boxes using Pillow.
    """
    logger.info("Running YOLOv5 detection on %s", image_path)

    model = get_model()

    results = model(str(image_path))
    df = results.pandas().xyxy[0]

    detections = []

    annotated_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    for _, row in df.iterrows():
        name = str(row["name"])
        confidence = round(float(row["confidence"]), 4)

        xmin = round(float(row["xmin"]), 2)
        ymin = round(float(row["ymin"]), 2)
        xmax = round(float(row["xmax"]), 2)
        ymax = round(float(row["ymax"]), 2)

        detections.append({
            "name": name,
            "confidence": confidence,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

        x1 = int(xmin)
        y1 = int(ymin)
        x2 = int(xmax)
        y2 = int(ymax)

        label = f"{name} {confidence:.2f}"

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3
        )

        text_bbox = draw.textbbox((x1, y1), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_y1 = max(0, y1 - text_height - 6)
        label_y2 = y1

        draw.rectangle(
            [(x1, label_y1), (x1 + text_width + 6, label_y2)],
            fill="red"
        )

        draw.text(
            (x1 + 3, label_y1 + 2),
            label,
            fill="white"
        )

    output_image_path = OUTPUT_DIR / f"detected-{image_path.stem}.jpg"
    annotated_image.save(output_image_path, "JPEG", quality=90)

    return detections, output_image_path


def upload_outputs(bucket, input_key, detections, annotated_image_path, timings):
    original_name = Path(input_key).stem.replace(" ", "_")
    result_id = str(uuid.uuid4())[:8]

    json_key = f"{OUTPUT_PREFIX}/detections/{original_name}-{result_id}.json"
    image_key = f"{OUTPUT_PREFIX}/detected-images/{original_name}-{result_id}.jpg"

    result = {
        "message": "Object detection completed successfully",
        "bucket": bucket,
        "input_key": input_key,
        "detected_count": len(detections),
        "detected_objects": sorted(list({item["name"] for item in detections})),
        "detections": detections,
        "outputs": {
            "json_key": json_key,
            "annotated_image_key": image_key
        },
        "timings": timings
    }

    logger.info("Uploading JSON result to s3://%s/%s", bucket, json_key)

    s3.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=json.dumps(result, indent=2, default=str).encode("utf-8"),
        ContentType="application/json"
    )

    logger.info("Uploading annotated image to s3://%s/%s", bucket, image_key)

    s3.upload_file(
        str(annotated_image_path),
        bucket,
        image_key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    return result


def lambda_handler(event, context):
    request_start = time.time()

    logger.info("Received event: %s", json.dumps(event, default=str)[:1000])

    try:
        bucket, key = parse_event(event)

        image_path = download_image(bucket, key)

        detection_start = time.time()
        detections, annotated_image_path = run_detection(image_path)
        detection_seconds = round(time.time() - detection_start, 3)

        timings = {
            "model_load_seconds": MODEL_LOAD_SECONDS,
            "detection_seconds": detection_seconds,
            "total_seconds": round(time.time() - request_start, 3)
        }

        result = upload_outputs(
            bucket=bucket,
            input_key=key,
            detections=detections,
            annotated_image_path=annotated_image_path,
            timings=timings
        )

        logger.info("Detection completed successfully. Objects found: %s", len(detections))

        return make_response(200, result)

    except ClientError as error:
        logger.exception("AWS error")

        return make_response(500, {
            "message": "AWS error",
            "error": str(error)
        })

    except Exception as error:
        logger.exception("Application error")

        return make_response(400, {
            "message": "Object detection failed",
            "error": str(error)
        })