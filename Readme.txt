Name -  Tanbir Singh Gill
Student ID - 240123

Short Description:
The current project involves creating an object detection system for images based in the cloud using AWS products. The project utilizes YOLOv5 for object detection purposes for images available in Amazon S3. The image file is uploaded into the input folder on the selected S3 bucket. Lambda executes the model and identifies the objects within the images; results are saved in the output folder of the S3 bucket as both JSON files and annotated images. A live API endpoint can be created via API Gateway.

AWS Services:

Amazon S3: Used for storing the input images and outputs.
AWS Lambda: Used for running the YOLOv5 code.
Amazon ECR: Used for storing the Lambda Docker image.
API Gateway: Used for exposing the Lambda function as API.
IAM: Used for managing the permission for Lambda to read/write to S3.
CloudWatch: Used for monitoring the Lambda function.

S3 Bucket:
bis202-yolov5

S3 Folder Structure:

input/: For storing uploaded images for object detection.
output/detections/: For storing JSON detection output.
output/detected-images/: For storing the annotated detected image.

Input Image used:
input/sample.jpg

Sample Lambda/API Test body:
{
"bucket": "bis202-yolov5",
"key": "input/sample.jpg"
}

Test result:
The system successfully detected the object within the uploaded image. YOLOv5 successfully detected a car with a confidence score of approximately 77.83%. The output JSON file and annotated detected image are stored in the output folder of S3 bucket.

Output Files:
output/detections/sample-d5b44711.json
output/detected-images/sample-d5b44711.jpg

Lambda Function name:
bis202-yolov5-detector

ECR repository:
bis202-yolov5

API Gateway POST method route:
POST /detect

URL of live API:
REPLACE_WITH_YOUR_API_GATEWAY_INVOKE_URL/detect

Link for Source code or Repository:
REPLACE_WITH_YOUR_GITHUB_OR_GOOGLE_DRIVE_LINK

Folder structure of the project:
code/

app.py
requirements.txt
Dockerfile

deploy/

Readme.txt
IAM policy.json
Lambda test and postman body.json
Docker.txt

images/
Contains all the screenshot related to the project

How to Use the Application:

1. Place your image in the S3 bucket under the input directory.
2. Send a POST request to the API Gateway using the S3 bucket and the image key.
3. Lambda will fetch the image from S3 and perform YOLOv5-based object detection.
4. The system will store a detection report in JSON format and an image with annotation in the output directory.
5. The user will have access to the results stored on Amazon S3.

Important Points to Consider:

* The image has to be located within the input/ directory.
* Only .jpg, .jpeg, and .png images are supported by the application.
* The Lambda function utilizes a Docker container image as YOLOv5 and PyTorch cannot fit in the regular Lambda ZIP package.
* The first execution of the Lambda function may take more time than usual due to model loading.