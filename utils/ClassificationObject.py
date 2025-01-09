from collections import Counter
import math
from itertools import chain
import numpy as np
import os


class ClassificationObject:
    def __init__(self, id: str, first_object_name: str, first_object_conf: float, first_trajectory: list[float], first_frame: int, frame_width: int, frame_height: int, first_object_colors_bgr: np.ndarray = None, first_object_colors_hls: np.ndarray = None, first_object_colors_str: np.ndarray = None):
        """
        :param id: Identification code for detected object, starting at 1 and chronologically increasing depending on the amount of detected objects.
        :param first_object_name: First classification name for the detected object (e.g. pedestrian, car, bus, truck, ...).
        :param object_name: Most common classification name for the detected object.
        :param first_object_conf: Confidence score coupled to the first_object_name for the detected object.
        :param first_trajectory: First trajectory, i.e. first bounding box coordinates.
        :param first_frame: Frame number where the object is first detected.
        :param frame_width: Width of the frame.
        :param frame_height: Height of the frame.
        :param first_object_colors_bgr: First primary colors of the object in BGR format.
        :param first_object_colors_hls: First primary colors of the object in HLS format.
        :param first_object_colors_str: First primary colors of the object mapped to string.

        :param frames: List of frame numbers where the object is detected, starting with first_frame.
        :param object_names: List of predicted object names, coupled to a confidence score in object_confs.
        :param object_confs: List of Confidence scores, coupled to an object_name in object_names.
                             Two above variables will be used for a final classification name.
        :param distance: Total distance object travelled on screen, measured in pixels.
        :param static_distance: Distance object travelled from first centroid to last centroid, measured in pixels.
        :param is_static: Boolean value, True if object is static, False if object is moving.
        :param occurences: Amount of occurences the object makes, equals the length of :param frames.
        :param trajectory: List[List[int]] containing 2D coordinates for 2 diagonally opposite corners of the object's bounding box for each frame.
                           [[x11, y11, x12, y12], [x21, y21, x22, y22],
                               [x31, y31, x32, y32], [...], ...]
                           x11: frame -> 1, x-coordinate of corner -> 1
                           x12: frame -> 1, x-coordinate of corner -> 2
                           x21: frame -> 2, x-coordinate of corner -> 1
        :param trajectory_centroids: List[List[int]] containing 2D coordinates for the centroid of the object's bounding box for each frame.
                                     [[x1, y1], [x2, y2], [x3, y3], ...]
                                     x1: frame -> 1, x-coordinate of centroid
                                     x2: frame -> 2, x-coordinate of centroid
        :param object_colors_bgr: Primary colors of the object in BGR format.
        :param object_colors_hls: Primary colors of the object in HLS format.
        :param object_colors_str: Primary colors of the object mapped to string.

        :param self.valid = True: Unused (inherited from the YOLOv3 pipeline)
        :param self.w = 0: Unused (inherited from the YOLOv3 pipeline)
        :param self.x = 0: Unused (inherited from the YOLOv3 pipeline)
        :param self.y = 0: Unused (inherited from the YOLOv3 pipeline)

        """

        # Instance variables needed for first initialization of an ClassificationObject.
        self.id = id
        self.first_frame = first_frame
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Instance variables initialized empty, these are filled or altered during classification process.
        self.frames = [first_frame]
        self.object_names = [first_object_name]
        self.object_name = first_object_name
        self.object_confs = [first_object_conf]
        self.distance = 0
        self.static_distance = 0
        self.is_static = True
        self.occurences = 1
        self.trajectory = [first_trajectory]
        self.trajectory_centroids = [self.find_centroid(first_trajectory)]

        self.object_colors_bgr = [
            first_object_colors_bgr] if first_object_colors_bgr is not None else []
        self.object_colors_hls = [
            first_object_colors_hls] if first_object_colors_hls is not None else []
        self.object_colors_str = [
            first_object_colors_str] if first_object_colors_str is not None else []
        self.object_color_str = []

        # Instance variables inherited from the YOLOv3 pipeline, have no use here.
        self.valid = True
        self.w = 0
        self.x = 0
        self.y = 0

    def add_frame_number(self, new_frame_number: int):
        """ Add the new frame number to the frames list.
        :param new_frame_number: The new number of the frame where object is also detected.

        """

        # Append to frames list.
        self.frames.append(new_frame_number)
        # +1 the occurences.
        self.add_occurence()

    def add_object_name(self, new_object_name: str):
        """ Add the new object name to the object_names list.
        :param new_object_name: The new object name.

        """

        # Append to object_names list.
        self.object_names.append(new_object_name)
        self.edit_object_name()

    def edit_object_name(self):
        """ Edit most common final classification name, when a new object name is added.

        """

        # Count the instances of each classification name.
        word_counts = Counter(self.object_names)
        # object name with most instances becomes the 'best' classification name.
        self.object_name = word_counts.most_common(1)[0][0]

    def add_object_conf(self, new_object_conf: float):
        """ Add the new object's confidence score to the object_confs list.
        :param new_object_conf: The confidence score of the new object name.

        """

        # Append to object_confs list.
        self.object_confs.append(new_object_conf)

    def add_trajectory(self, new_bbox_coordinates: list[float]):
        """ Add bounding box coordinates to the trajectory list.
        :param new_bbox_coordinates: The bbox coordinates of the detected object's position
                                     [x11, y11, x12, y12]

        """

        # Append to objects trajectory list.
        self.trajectory.append(new_bbox_coordinates)

        # Calculate centroid information about bbox.
        centroid_coordinates = self.find_centroid(new_bbox_coordinates)
        # Add centroid coordinates to the trajectory_centroids list.
        self.add_trajectory_centroid(centroid_coordinates)

    def find_centroid(self, bbox_coordinates: list[float]) -> list[float]:
        """ Calculate centroid information about bbox.
        :param bbox_coordinates: The bbox coordinates of the detected object's position
                                 [x11, y11, x12, y12]
        :returns: A list of integers representing the x- and y-coordinate of the centroids position

        """

        return [(bbox_coordinates[0]+bbox_coordinates[2])/2, (bbox_coordinates[1]+bbox_coordinates[3])/2]

    def add_trajectory_centroid(self, new_trajectory_centroid: list[float]):
        """ Add centroid coordinates to the trajectory-centroids list.
        :param new_trajectory_centroid: The centroid coordinates of the detected object's position
                                     [x11, y11, x12, y12]

        """

        # Append to objects trajectory_centroid list.
        self.trajectory_centroids.append(new_trajectory_centroid)
        self.add_distance()
        self.edit_static_distance()

    def add_occurence(self):
        """ +1 the occurences.

        """

        self.occurences += 1

    def add_distance(self):
        """ Calculate the Euclidean distance travelled from previous centroid to new centroid. Add to total distance travelled.

        """

        previous_centroid = self.trajectory_centroids[-2]
        new_centroid = self.trajectory_centroids[-1]

        # Calculate the Euclidean distance travelled from previous centroid to new centroid.
        new_distance = math.sqrt(
            (new_centroid[0]-previous_centroid[0])**2 + (new_centroid[1]-previous_centroid[1])**2)
        # Add newly calculated distance to total distance travelled.
        self.distance += new_distance

    def edit_static_distance(self):
        """ Calculate the Euclidean distance travelled from first centroid to last centroid. Add to total distance travelled.

        """

        first_centroid = self.trajectory_centroids[0]
        last_centroid = self.trajectory_centroids[-1]

        # Calculate the Euclidean distance travelled from first centroid to last centroid.
        static_distance = math.sqrt(
            (last_centroid[0]-first_centroid[0])**2 + (last_centroid[1]-first_centroid[1])**2)
        # Add newly calculated distance to total distance travelled.
        self.static_distance = static_distance
        self.edit_is_static()

    def edit_is_static(self):
        """ Check if static distance is smaller than a certain threshold.

        """

        if self.static_distance <= int(os.getenv('MIN_STATIC_DISTANCE')):
            self.is_static = True
        else:
            self.is_static = False

    def add_object_colors_bgr(self, new_object_colors_bgr: np.ndarray):
        """ Add the new object's colors to the object_colors_bgr list.
        :param new_object_colors_bgr: The new object's colors of the object.

        """

        # Append to object_colors_bgr list.
        self.object_colors_bgr.append(new_object_colors_bgr)

    def add_object_colors_hls(self, new_object_colors_hls: np.ndarray):
        """ Add the new object's colors to the object_colors_hls list.
        :param new_object_colors_hls: The new object's colors of the object.

        """

        # Append to object_colors_hls list.
        self.object_colors_hls.append(new_object_colors_hls)

    def add_object_colors_str(self, new_object_colors_str: np.ndarray):
        """ Add the new object's colors to the object_colors_str list.
        :param new_object_colors_str: The new object's colors of the object.

        """

        # Append to object_colors_str list.
        self.object_colors_str.append(new_object_colors_str)
        self.edit_object_color_str()

    def edit_object_color_str(self):
        """ Edit most common final colors, when a new object name is added.

        """

        # Count the instances of each color.
        flattened_list = list(chain(*self.object_colors_str))
        word_counts = Counter(flattened_list)
        # object colors with most instances become the 'best' object colors.
        most_common = word_counts.most_common(3)

        # Get colors from most common list.
        colors = [color[0] for color in most_common]
        self.object_color_str = colors
