BIS202 Assignment 2 - Deployment Details

AWS Region:
ap-southeast-2

S3 Bucket:
bis202-yolov5

S3 Input Folder:
input/

S3 Output Folder:
output/

ECR Repository:
bis202-yolov5

Lambda Function:
bis202-yolov5-detector

API Gateway Route:
POST /detect

API Gateway Test Body:
{
  "bucket": "bis202-yolov5",
  "key": "input/sample.jpg"
}

Docker Build Command:
docker buildx build --platform linux/amd64 --provenance=false -t bis202-yolov5:v6 --load .

Docker Tag Command:
docker tag bis202-yolov5:v2 271434547702.dkr.ecr.ap-southeast-2.amazonaws.com/bis202-yolov5:v6

Docker Push Command:
docker push 271434547702.dkr.ecr.ap-southeast-2.amazonaws.com/bis202-yolov5:v6

Lambda Settings:
Memory: 6144 MB
Timeout: 5 minutes
Ephemeral storage: 2048 MB

Environment Variables:
MODEL_CONFIDENCE = 0.25
OUTPUT_PREFIX = output

Test Result:
The Lambda function successfully detected a car in input/sample.jpg and saved the JSON result and annotated image into the output folder.