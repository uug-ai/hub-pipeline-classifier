from utils.ClassificationObject import ClassificationObject
import numpy as np


def create_classification_object(id: str, first_object_name: str, first_object_conf: float, first_trajectory: list[float], first_frame: int, frame_width: int, frame_height: int, first_colors_bgr: np.ndarray = None, first_colors_hls: np.ndarray = None, first_colors_str: np.ndarray = None) -> ClassificationObject:
    """ Create/initialize a ClassificationObject (Only done if the id is not yet existing in the already existing ClassificationObjects)
        :param id: Identification code for detected object, starting at 1 and chronologically increasing depending on the amount of detected objects.
        :param first_object_name: First classification name for the detected object (e.g. pedestrian, car, bus, truck, ...)
        :param first_object_conf: Confidence score coupled to the first_object_name for the detected object.
        :param first_trajectory: First trajectory, i.e. first bounding box coordinates.
        :param first_frame: Frame number where the object is first detected.
        :param frame_width: Width of the frame.
        :param frame_height: Height of the frame.
        :param first_colors_bgr: First primary colors of the object in BGR format.
        :param first_colors_hls: First primary colors of the object in HLS format.
        :param first_colors_str: First primary colors of the object mapped to string.
    """
    detected_object = ClassificationObject(id, first_object_name, first_object_conf, first_trajectory, first_frame, frame_width, frame_height, first_colors_bgr, first_colors_hls, first_colors_str)

    return detected_object


def edit_classification_object(id: str, object_name: str, object_conf: float, trajectory: list[float], frame_number: int, classification_object_list: list[ClassificationObject], colors_bgr: np.ndarray = None, colors_hls: np.ndarray = None, colors_str: np.ndarray = None):
    """Edit a ClassificationObject (Only done if the id exists in the already existing ClassificationObjects)
        :param id: Identification code for detected object, starting at 1 and chronologically increasing depending on the amount of detected objects.
        :param object_name: Classification name for the detected object (e.g. pedestrian, car, bus, truck, ...)
        :param object_conf: Confidence score coupled to the object_name for the detected object.
        :param trajectory: Trajectory, i.e. bounding box coordinates.
        :param frame_number: Frame number where the object is detected, i.e. current frame number.
        :param colors_bgr: Current primary colors of the object in BGR format.
        :param colors_hls: Current colors of the object in HLS format.
        :param colors_str: Current primary colors of the object mapped to string.
        :param classification_object_list: list of already existing objects, this is used to find the correct object matching ids.
    """
    # Find object with matching ids
    classification_object = find_classification_object(
        classification_object_list, id)

    # Edit/append object variables, such as: object name, coupled confidence score, bbox coordinates, current frame number.
    classification_object.add_object_name(object_name)
    classification_object.add_object_conf(object_conf)
    classification_object.add_trajectory(trajectory)
    classification_object.add_frame_number(frame_number)

    classification_object.add_object_colors_bgr(colors_bgr) if colors_bgr is not None else None
    classification_object.add_object_colors_hls(colors_hls) if colors_hls is not None else None
    classification_object.add_object_colors_str(colors_str) if colors_str is not None else None


def find_classification_object(classification_object_list: list[ClassificationObject], target_id: str) -> ClassificationObject:
    """ Find object with matching ids from classification_object_list using target_id.
        :param classification_object_list: list of already existing objects.
        :param target_id: id to find already existing object with.
    """
    # Iterate over already existing objects. When ids match, the correct object is found
    for obj in classification_object_list:
        if obj.id == target_id:
            return obj
        # If there is no object found with the target-id, throw ValueError.
        else:
            ValueError('No object found with this target-id')