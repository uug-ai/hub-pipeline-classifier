import cv2
from utils.ClassificationObject import ClassificationObject
import random
import os



def annotate_frame(frame, frame_number, classification_object_list: list[ClassificationObject], min_distance, min_detections):
    """ Annotate the frame with the classification objects.

    :param frame: The frame to annotate.
    :param frame_number: The current frame number.
    :param classification_object_list: The list of classification objects.
    :param min_distance: The minimum distance to be considered.
    :param min_detections: The minimum amount of detections to be considered.

    """

    # Loop over the classification objects.
    # If the last frame of the classification object is the current frame number,
    # In other words, the object is still present in the current frame.
    for classification_object in classification_object_list:
        if classification_object.frames[-1] == frame_number:

            # If the object is too far away or has too few detections, the color of the bounding box is red.
            # Otherwise, the color is green.
            color = (0, 255, 0) if classification_object.distance > min_distance and len(classification_object.trajectory) > min_detections else (0, 0, 255)
            last_trajectory = classification_object.trajectory[-1]
            trajectory_list_length = len(classification_object.trajectory_centroids)

            # Annotate the frame with the object's bounding box and trajectory.
            # Aswell as the object's name and confidence score.
            cv2.rectangle(
                img=frame,
                pt1=(int(last_trajectory[0]), int(last_trajectory[1])),
                pt2= (int(last_trajectory[2]), int(last_trajectory[3])),
                color=color,
                thickness= 2)

            cv2.putText(
                img=frame,
                text=classification_object.object_name + ' ' + str(int(100 * classification_object.object_confs[-1])) + '%',
                org=(int(last_trajectory[0]), int(last_trajectory[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=color,
                thickness=2)
            
            if classification_object.object_colors_bgr != []:
                for i, object_color in enumerate(classification_object.object_colors_bgr[-1]):
                    cv2.circle(
                        img=frame,
                        center=(int(last_trajectory[0]) + 10, int(last_trajectory[1]) - 40 - i*25),
                        radius=10,
                        color=object_color,
                        thickness=-1)

                    # Write the text on the rotated frame
                    cv2.putText(
                        img=frame,
                        text=str(classification_object.object_colors_str[-1][i]), 
                        org=(int(last_trajectory[0]) + 30, int(last_trajectory[1]) - 35 - i*25), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=object_color,
                        thickness=2)

            
            for i in range(1, trajectory_list_length):
                cv2.line(img=frame,
                        pt1=list(map(int, classification_object.trajectory_centroids[i-1])),
                        pt2=list(map(int, classification_object.trajectory_centroids[i])),
                        color = color,
                        thickness=2)
                
    return frame


def annotate_bbox_frame(bbox_frame, classification_object_list: list[ClassificationObject]):
    """ Annotate the frame with the classification objects.

    :param frame: The frame to annotate.
    :param frame_number: The current frame number.
    :param classification_object_list: The list of classification objects.

    """

    # Loop over the classification objects.
    # If the last frame of the classification object is the current frame number,
    # In other words, the object is still present in the current frame.
    for classification_object in classification_object_list:

        min_detections = int(os.getenv('MIN_DETECTIONS'))
        if len(classification_object.trajectory) >= min_detections:

            first_trajectory = classification_object.trajectory[0]
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Annotate the frame with the object's bounding box and trajectory.
            # Aswell as the object's name and confidence score.
            cv2.rectangle(
                img=bbox_frame,
                pt1=(int(first_trajectory[0]), int(first_trajectory[1])),
                pt2= (int(first_trajectory[2]), int(first_trajectory[3])),
                color=random_color,
                thickness=2)
            
            cv2.putText(
                img=bbox_frame,
                text='static' if classification_object.is_static else 'dynamic', 
                org=(int(first_trajectory[0]), int(first_trajectory[1] - 10)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=random_color,
                thickness=2)
            
            if not classification_object.is_static:
            
                for i in range(1, len(classification_object.trajectory_centroids)):
                    cv2.line(
                        img=bbox_frame,
                        pt1=list(map(int, classification_object.trajectory_centroids[i-1])),
                        pt2=list(map(int, classification_object.trajectory_centroids[i])),
                        color = random_color,
                        thickness=2)
            
    return bbox_frame

        
        
