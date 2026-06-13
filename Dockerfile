FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y git curl mesa-libGL glib2 libSM libXext libXrender libgomp && yum clean all

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

RUN git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5.git ${LAMBDA_TASK_ROOT}/yolov5

RUN git config --global --add safe.directory ${LAMBDA_TASK_ROOT}/yolov5

RUN curl -L --fail --retry 5 --retry-delay 10 \
    -o ${LAMBDA_TASK_ROOT}/yolov5s.pt \
    https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

RUN python -c "import os; p='${LAMBDA_TASK_ROOT}/yolov5s.pt'; print('MODEL_SIZE_BYTES=', os.path.getsize(p)); assert os.path.getsize(p) > 10000000; print('MODEL_FILE_DOWNLOADED_OK')"

COPY app.py ${LAMBDA_TASK_ROOT}/app.py

ENV YOLOV5_DIR=${LAMBDA_TASK_ROOT}/yolov5
ENV MODEL_PATH=${LAMBDA_TASK_ROOT}/yolov5s.pt
ENV MODEL_CONFIDENCE=0.25
ENV OUTPUT_PREFIX=output
ENV MPLCONFIGDIR=/tmp/matplotlib

CMD ["app.lambda_handler"]