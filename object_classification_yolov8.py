# This script is used to classify objects in a video using the YOLOv8 model.
# The script reads a video from a message queue, classifies the objects in the video, and writes the annotated video to a message queue.
# It saves the detected objects in a json file and the annotated video locally.
# For this it uses the ultralytics package to perform object detection and tracking.

# Local imports
from utils.ReturnObject import ReturnJSON
from utils.TranslateObject import translate
from utils.VariableClass import VariableClass
from utils.ColorDetector import FindObjectColors
from utils.ClassificationObject import ClassificationObject
from utils.AnnotateFrame import annotate_frame, annotate_bbox_frame
from utils.ClassificationObjectFunctions import create_classification_object, edit_classification_object, find_classification_object
from utils.kerberos_vault import KerberosVault
from utils.message_brokers import RabbitMQ

# External imports
import os
import cv2
import time
import json
import torch
import numpy as np
from ultralytics import YOLO


# Following error is thrown: [W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
# https://stackoverflow.com/questions/69711410/could-not-initialize-nnpack
# torch.backends.nnpack.enabled = False


def load_model(var):
    """Load the YOLO model once for the worker process."""

    inference_backend = var.INFERENCE_BACKEND.lower()
    if inference_backend not in {'local', 'triton'}:
        raise ValueError(f'Unsupported inference backend: {var.INFERENCE_BACKEND}')

    model_source = var.TRITON_MODEL_URL if inference_backend == 'triton' else var.MODEL_NAME
    if not model_source:
        raise ValueError(f'Model source is required for the {inference_backend} inference backend')

    device = 'remote' if inference_backend == 'triton' else ('cuda' if torch.cuda.is_available() else 'cpu')
    max_model_load_attempts = 3
    model_load_delay_seconds = 5
    model = None

    for attempt in range(1, max_model_load_attempts + 1):
        try:
            if inference_backend == 'triton':
                model = YOLO(model_source, task=var.TRITON_MODEL_TASK)
                model.predict(
                    source=np.zeros((32, 32, 3), dtype=np.uint8),
                    data=var.TRITON_DATA_CONFIG,
                    imgsz=var.INFERENCE_IMAGE_SIZE,
                    verbose=False)
            else:
                model = YOLO(model_source)
                model = model.to(device)
            break
        except Exception as exc:
            model = None
            if var.LOGGING:
                print(f'Error loading YOLO model (attempt {attempt}/{max_model_load_attempts}): {exc}')
            if attempt < max_model_load_attempts:
                time.sleep(model_load_delay_seconds)

    if model is None:
        raise RuntimeError('Unable to load YOLO model')

    if var.LOGGING:
        print(f'c) YOLO model loaded on device: {device}')

    return model


def retrieve_media_with_retries(var, kerberos_vault, message):
    max_download_attempts = 3
    download_delay_seconds = 5
    err = None

    for attempt in range(1, max_download_attempts + 1):
        err = kerberos_vault.retrieve_media(
            message=message,
            media_type='video',
            media_savepath=var.MEDIA_SAVEPATH)
        if err is None:
            if var.LOGGING:
                print('Media retrieved from Kerberos Vault')
            return True

        if var.LOGGING:
            print(err)
            print(f'Error retrieving media from Kerberos Vault (attempt {attempt}/{max_download_attempts})')
        if attempt < max_download_attempts:
            time.sleep(download_delay_seconds)

    if var.LOGGING:
        print('Skipping message and waiting for next one')
    return False


def reset_tracking_state(var, model):
    """Reset persisted YOLO tracker state so each message starts as a new video."""

    predictor = getattr(model, 'predictor', None)
    trackers = getattr(predictor, 'trackers', None) if predictor is not None else None
    if not trackers:
        return

    for tracker in trackers:
        reset = getattr(tracker, 'reset', None)
        if callable(reset):
            reset()

    vid_path = getattr(predictor, 'vid_path', None)
    if isinstance(vid_path, list):
        predictor.vid_path = [None] * len(vid_path)

    if var.LOGGING:
        print('Reset YOLO tracker state for new message')


def process_message(var, model, rabbitmq, kerberos_vault, message):
    cap = None
    video_out = None
    bbox_frame = None

    try:
        if var.LOGGING:
            print('2) Retrieving media from Kerberos Vault')
            print(message)

        if not retrieve_media_with_retries(var, kerberos_vault, message):
            return False
        rabbitmq.process_data_events()
        reset_tracking_state(var, model)

        if var.TIME_VERBOSE:
            start_time = time.time()
            total_time_preprocessing = 0
            total_time_class_prediction = 0
            total_time_color_prediction = 0
            total_time_processing = 0
            total_time_postprocessing = 0
            start_time_preprocessing = time.time()

        if var.LOGGING:
            print(f'3) Opening video file: {var.MEDIA_SAVEPATH}')
        cap = cv2.VideoCapture(var.MEDIA_SAVEPATH)
        if not cap.isOpened():
            if var.LOGGING:
                print(f'Unable to open video file: {var.MEDIA_SAVEPATH}')
            return False

        if var.SAVE_VIDEO:
            fourcc = cv2.VideoWriter.fourcc(*'avc1')
            video_out = cv2.VideoWriter(
                filename=var.OUTPUT_MEDIA_SAVEPATH,
                fourcc=fourcc,
                fps=var.CLASSIFICATION_FPS,
                frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

        if var.FIND_DOMINANT_COLORS:
            color_detector = FindObjectColors(
                downsample_factor=0.7,
                min_clusters=var.MIN_CLUSTERS,
                max_clusters=var.MAX_CLUSTERS,
            )

        classification_object_list: list[ClassificationObject] = []
        classification_object_ids: list[int] = []

        frame_number, predicted_frames = 0, 0
        frame_skip_factor = int(cap.get(cv2.CAP_PROP_FPS) / var.CLASSIFICATION_FPS)
        if frame_skip_factor <= 0:
            if var.LOGGING:
                print('Skipping message because video FPS is lower than CLASSIFICATION_FPS')
            return False

        max_frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if var.LOGGING:
            print('4) Classifying frames')
        if var.TIME_VERBOSE:
            total_time_preprocessing += time.time() - start_time_preprocessing
            start_time_processing = time.time()

        while (predicted_frames < var.MAX_NUMBER_OF_PREDICTIONS) and (frame_number < max_frame_number):
            success, frame = cap.read()
            if not success:
                break

            if var.CREATE_BBOX_FRAME and frame_number == 0:
                bbox_frame = frame.copy()

            if frame_number % frame_skip_factor == 0:
                if var.TIME_VERBOSE:
                    start_time_class_prediction = time.time()
                track_options = dict(
                    source=frame,
                    persist=True,
                    verbose=False,
                    conf=var.CLASSIFICATION_THRESHOLD,
                    imgsz=var.INFERENCE_IMAGE_SIZE,
                    classes=var.ALLOWED_CLASSIFICATIONS)
                if var.INFERENCE_BACKEND == 'triton':
                    track_options['data'] = var.TRITON_DATA_CONFIG
                try:
                    results = model.track(**track_options)
                except Exception:
                    if var.INFERENCE_BACKEND == 'triton':
                        model.predictor = None
                    raise
                if var.TIME_VERBOSE:
                    total_time_class_prediction += time.time() - start_time_class_prediction

                if results is not None:
                    for box, mask in zip(results[0].boxes, results[0].masks or [None] * len(results[0].boxes)):
                        if box.id is None:
                            break

                        object_id = int(box.id)
                        object_name = translate(results[0].names[int(box.cls)])
                        object_conf = float(box.conf)
                        object_trajectory = box.xyxy.tolist()[0]
                        object_mask = np.int32(
                            mask.xy[0].tolist()) if mask is not None else None

                        if object_id in classification_object_ids:
                            classification_object = find_classification_object(
                                classification_object_list, object_id)

                            if var.FIND_DOMINANT_COLORS and classification_object.occurences % var.COLOR_PREDICTION_INTERVAL == 0:
                                if var.TIME_VERBOSE:
                                    start_time_color_prediction = time.time()
                                main_colors_bgr, main_colors_hls, main_colors_str = color_detector.crop_and_detect(
                                    frame=frame,
                                    trajectory=object_trajectory,
                                    mask_polygon=object_mask)
                                if var.TIME_VERBOSE:
                                    total_time_color_prediction += time.time() - start_time_color_prediction
                            else:
                                main_colors_bgr, main_colors_hls, main_colors_str = None, None, None

                            edit_classification_object(
                                id=object_id,
                                object_name=object_name,
                                object_conf=object_conf,
                                trajectory=object_trajectory,
                                frame_number=frame_number,
                                classification_object_list=classification_object_list,
                                colors_bgr=main_colors_bgr,
                                colors_hls=main_colors_hls,
                                colors_str=main_colors_str)

                        else:
                            if var.FIND_DOMINANT_COLORS:
                                if var.TIME_VERBOSE:
                                    start_time_color_prediction = time.time()
                                main_colors_bgr, main_colors_hls, main_colors_str = color_detector.crop_and_detect(
                                    frame=frame,
                                    trajectory=object_trajectory,
                                    mask_polygon=object_mask)
                                if var.TIME_VERBOSE:
                                    total_time_color_prediction += time.time() - start_time_color_prediction
                            else:
                                main_colors_bgr, main_colors_hls, main_colors_str = None, None, None

                            classification_object = create_classification_object(
                                id=object_id,
                                first_object_name=object_name,
                                first_object_conf=object_conf,
                                first_trajectory=object_trajectory,
                                first_frame=frame_number,
                                frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                frame_height=int(
                                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                first_colors_bgr=main_colors_bgr,
                                first_colors_hls=main_colors_hls,
                                first_colors_str=main_colors_str)

                            classification_object_ids.append(object_id)
                            classification_object_list.append(
                                classification_object)

                if var.SAVE_VIDEO or var.PLOT:
                    annotated_frame = annotate_frame(
                        frame=frame,
                        frame_number=frame_number,
                        classification_object_list=classification_object_list,
                        min_distance=var.MIN_DISTANCE,
                        min_detections=var.MIN_DETECTIONS)

                    cv2.imshow("YOLOv8 Tracking",
                               annotated_frame) if var.PLOT else None
                    cv2.waitKey(1) if var.PLOT else None

                    video_out.write(annotated_frame) if var.SAVE_VIDEO else None

                predicted_frames += 1
            frame_number += 1

        if var.TIME_VERBOSE:
            total_time_processing += time.time() - start_time_processing
            start_time_postprocessing = time.time()

        if var.CREATE_BBOX_FRAME and bbox_frame is not None:
            if var.LOGGING:
                print('5) Annotating bbox frame')
            bbox_frame = annotate_bbox_frame(
                bbox_frame=bbox_frame,
                classification_object_list=classification_object_list)

        return_json = ReturnJSON()
        if var.CREATE_RETURN_JSON:
            if var.LOGGING:
                print('6) Creating ReturnJSON object')

            filtered_classification_object_list = []
            for classification_object in classification_object_list:
                if classification_object.occurences >= var.MIN_DETECTIONS:
                    filtered_classification_object_list.append(
                        classification_object)
                    return_json.add_detected_object(classification_object)
            if var.LOGGING:
                print(f"\t - {len(classification_object_list)} objects where detected. Of which {len(filtered_classification_object_list)} objects where detected more than {var.MIN_DETECTIONS} times.")

        return_json.save_returnjson(
            var.RETURN_JSON_SAVEPATH) if var.SAVE_RETURN_JSON else None

        cv2.imwrite(var.BBOX_FRAME_SAVEPATH,
                    bbox_frame) if var.SAVE_BBOX_FRAME and bbox_frame is not None else None

        if var.TIME_VERBOSE:
            total_time_postprocessing += time.time() - start_time_postprocessing

        rabbitmq.process_data_events()
        if var.TARGET_QUEUE_NAME != "":
            message['operation'] = return_json.return_object['operation']
            if 'data' in message:
                message['data'].update(return_json.return_object['data'])
            else:
                message['data'] = return_json.return_object['data']

            return_message = json.dumps(message)
            if var.LOGGING:
                print('7) Publishing classification result to RabbitMQ')
            rabbitmq.send_message(return_message)
            if var.LOGGING:
                print('Published classification result to RabbitMQ')

        if var.TIME_VERBOSE:
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(
                f'\t - Classification took: {round(time.time() - start_time, 1)} seconds, @ {var.CLASSIFICATION_FPS} fps.')
            print(
                f'\t\t - {round(total_time_preprocessing, 2)}s for preprocessing and initialisation')
            print(
                f'\t\t - {round(total_time_processing, 2)}s for processing of which:')
            print(
                f'\t\t\t - {round(total_time_class_prediction, 2)}s for class prediction')
            print(
                f'\t\t\t - {round(total_time_color_prediction, 2)}s for color prediction')
            print(
                f'\t\t\t - {round(total_time_processing - total_time_class_prediction - total_time_color_prediction, 2)}s for other processing')
            print(
                f'\t\t - {round(total_time_postprocessing, 2)}s for postprocessing')
            duration_seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
            print(f'\t - Original video: {round(duration_seconds, 1)} seconds, @ {round(fps, 1)} fps @ {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. File size of {round(os.path.getsize(var.MEDIA_SAVEPATH)/1024**2, 1)} MB')

        if var.LOGGING:
            print('Finished processing message')
        return True
    finally:
        if var.LOGGING:
            print('8) Releasing video writer and closing video capture')
        if video_out is not None:
            video_out.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        rabbitmq.process_data_events()
        if var.LOGGING:
            print('Cleanup complete\n\n')


def ensure_model_loaded(var, rabbitmq, model):
    """Keep the worker alive while retrying transient model initialization failures."""

    if model is not None:
        return model

    try:
        return load_model(var)
    except RuntimeError as exc:
        if var.LOGGING:
            print(f'Unable to load YOLO model, retrying: {exc}')
        rabbitmq.process_data_events()
        time.sleep(5)
        return None


def main():
    var = VariableClass()

    if var.LOGGING:
        print('a) Initializing RabbitMQ')
    rabbitmq = RabbitMQ(
        queue_name=var.QUEUE_NAME,
        target_queue_name=var.TARGET_QUEUE_NAME,
        exchange=var.QUEUE_EXCHANGE,
        host=var.QUEUE_HOST,
        username=var.QUEUE_USERNAME,
        password=var.QUEUE_PASSWORD)

    if var.LOGGING:
        print('b) Initializing Kerberos Vault')
    kerberos_vault = KerberosVault(
        storage_uri=var.STORAGE_URI,
        storage_access_key=var.STORAGE_ACCESS_KEY,
        storage_secret_key=var.STORAGE_SECRET_KEY)

    model = None

    while True:
        rabbitmq.process_data_events()
        model = ensure_model_loaded(var, rabbitmq, model)
        if model is None:
            continue

        if var.LOGGING:
            print('1) Receiving message from RabbitMQ')
        message = rabbitmq.receive_message()
        if message == []:
            if var.LOGGING:
                print('No message received, waiting for 3 seconds')
            time.sleep(3)
            rabbitmq.process_data_events()
            continue

        if var.LOGGING:
            print('Received message from RabbitMQ')
        try:
            process_message(var, model, rabbitmq, kerberos_vault, message)
        except Exception as exc:
            if var.LOGGING:
                print(f'Error processing message: {exc}')
            rabbitmq.process_data_events()


if __name__ == '__main__':
    main()
