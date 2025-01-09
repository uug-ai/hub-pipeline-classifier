# Use the official Python 3.10 slim-bookworm as base image
FROM python:3.10-slim-bookworm

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/
    

# Install linux packages and clean up (clean-up added, last line)
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /usr/src/ultralytics

# Clone the repository
RUN git clone https://github.com/ultralytics/ultralytics /usr/src/ultralytics
# Checkout tag v8.1.0
RUN cd /usr/src/ultralytics && git checkout v8.1.0

# Add yolov8n.pt model
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /usr/src/ultralytics/

# Copy requirements.txt including the needed python packages
COPY requirements.txt /ml/requirements.txt

# Upgrade pip and install pip packages
RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install --no-cache-dir -r /ml/requirements.txt 

# Create necessary directories
RUN mkdir -p /ml/data/input /ml/data/output

# Set working directory
WORKDIR /ml

# Copy application code
COPY . .

# Environment variables
ENV MEDIA_SAVEPATH "/ml/data/input/input_video.mp4"

# Model parameters
ENV MODEL_NAME "yolov8n-seg.pt"

# Queue parameters
ENV QUEUE_NAME "" 
ENV TARGET_QUEUE_NAME ""
ENV QUEUE_EXCHANGE ""
ENV QUEUE_HOST ""
ENV QUEUE_USERNAME ""
ENV QUEUE_PASSWORD ""

# Kerberos Vault parameters
ENV STORAGE_URI ""
ENV STORAGE_ACCESS_KEY ""
ENV STORAGE_SECRET_KEY ""

# Feature parameters
ENV PLOT "False"

ENV SAVE_VIDEO "False"
ENV OUTPUT_MEDIA_SAVEPATH "/ml/data/output/output_video.mp4"

ENV CREATE_BBOX_FRAME "False"
ENV SAVE_BBOX_FRAME "False"
ENV BBOX_FRAME_SAVEPATH "/ml/data/output/output_bbox_frame.jpg"

ENV CREATE_RETURN_JSON "False"
ENV SAVE_RETURN_JSON "False"
ENV RETURN_JSON_SAVEPATH "/ml/data/output/output_json.json"

ENV TIME_VERBOSE "True"
ENV LOGGING "True"

ENV FIND_DOMINANT_COLORS "False"
ENV COLOR_PREDICTION_INTERVAL "1"
ENV MIN_CLUSTERS "3"
ENV MAX_CLUSTERS "3"

# Classification parameters
ENV CLASSIFICATION_FPS ""
ENV CLASSIFICATION_THRESHOLD ""
ENV MAX_NUMBER_OF_PREDICTIONS ""
ENV MIN_DISTANCE ""
ENV MIN_STATIC_DISTANCE ""
ENV MIN_DETECTIONS ""
ENV ALLOWED_CLASSIFICATIONS "0, 1, 2, 3, 5, 7, 14, 15, 16, 24, 26, 28"


# Run the application
ENTRYPOINT ["python" , "object_classification_yolov8.py"]


