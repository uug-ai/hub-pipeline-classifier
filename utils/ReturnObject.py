from utils.ClassificationObject import ClassificationObject
import json


class ReturnJSON:
    def __init__(self):
        """ Initialize a ReturnJSON class object, this makes sure the final json object has the correct structure.
        :params: Parameter explanations can be found in __init__ of ClassificationObject class.

        """

        self.object_count = 0
        self.properties = []
        self.details = []

        self.return_object = {'operation': 'classify',
                              'data': {
                                  'objectCount': self.object_count,
                                  'properties': self.properties,
                                  'details': self.details
                              }
                              }

    def add_detected_object(self, det_obj: ClassificationObject):
        """ Adds a detected object to the ReturnJSON class object.
        :param det_obj: ClassificationObject whose characteristics should be saved in the ReturnJSON object.

        """
        
        if not det_obj.is_static:
            self.return_object['data']['objectCount'] += 1
            self.return_object['data']['properties'].append(det_obj.object_name)

        details_dict = {'id': str(det_obj.id),
                        'classified': det_obj.object_name,
                        'distance': det_obj.distance,
                        'staticDistance': det_obj.static_distance,
                        'isStatic': det_obj.is_static,
                        'frameWidth': det_obj.frame_width,
                        'frameHeight': det_obj.frame_height,
                        'frame': det_obj.first_frame,
                        'frames': det_obj.frames,
                        'occurence': det_obj.occurences,
                        'traject': det_obj.trajectory,
                        'trajectCentroids': det_obj.trajectory_centroids,
                        'colorsBGR': det_obj.object_colors_bgr,
                        'colorsHLS': det_obj.object_colors_hls,
                        'colorsStr': det_obj.object_colors_str,
                        'colorStr': det_obj.object_color_str,
                        'valid': det_obj.valid,
                        'w': det_obj.w,
                        'x': det_obj.x,
                        'y': det_obj.y
                        }
        self.return_object['data']['details'].append(details_dict)

    def batch_add_detected_object(self, det_obj_list: list[ClassificationObject]):
        """ Batch add detected_objects from a ClassificationObject list.
        :param det_obj_list: List containing the ClassificationObjects whose characteristics should be saved in the ReturnJSON object.

        """

        for det_obj in det_obj_list:
            self.add_detected_object(det_obj)

    def save_returnjson(self, path: str):
        """ Save the ReturnJSON object to a json file.
        :param path: Path where the json file should be saved.

        """

        with open(path, 'w') as file:
            json.dump(self.return_object, file, indent='\t')
