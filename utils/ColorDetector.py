from uugai_python_color_prediction.ColorPrediction import ColorPrediction
import cv2
import numpy as np



class FindObjectColors():
    """ Class to find the colors of an object in an image.
    This class is used to crop the object from the image, segment it from the background,
    and then find the main colors of the object.
    
    """

    def __init__(self, crop_reduction = 0, min_clusters = 1, max_clusters = 8, downsample_factor = 0, increase_elbow = 0):
        """ Initialize the class with the given parameters.
        
        :param crop_reduction: The percentage to reduce the crop by.
        :param min_clusters: The minimum number of clusters to use in the KMeans algorithm.
        :param max_clusters: The maximum number of clusters to use in the KMeans algorithm.
        :param downsample_factor: The factor to downsample the image by.
        :param increase_elbow: The amount to increase the elbow by.
        
        """

        self.crop_reduction = crop_reduction
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.downsample_factor = downsample_factor
        self.increase_elbow = increase_elbow
        

    def crop_detected_object(self, frame, trajectory):
        """ Crop the image to only the detected object for color determination.
            :param image: Contains the original image of full size.
            :param crop_coords: Contains the coordinates as a list[float], these originate from
                                the bounding box trajectory.

        """

        # Extract coordinates from crop_coords.
        x1 = int(trajectory[0])
        y1 = int(trajectory[1])
        x2 = int(trajectory[2])
        y2 = int(trajectory[3])

        # Ensure the coordinates are in the correct order.
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Calculate the width and height of the rectangle.
        width = x2 - x1
        height = y2 - y1

        # Calculate the amount to reduce each side by the given percentage.
        reduction_x = int(width * self.crop_reduction / 2)
        reduction_y = int(height * self.crop_reduction / 2)

        x1 += reduction_x
        y1 += reduction_y
        x2 -= reduction_x
        y2 -= reduction_y

        # Crop the region defined by the coordinates.
        cropped_image = frame[y1:y2, x1:x2]
        return cropped_image
    

    def segment_object(self, frame, mask_polygon):
        """ Segment the object from the background.

        :param object_img: The image of the object to segment.

        """

        # Create a mask image with the same dimensions as the original image
        # This creates a np.zeros array with the same dimensions as the frame width and height
        mask_image = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Create a polygon mask in the mask_image.
        # This creates a filled polygon where the object is located.
        # The filling doesn't matter, as long as it's not 0.
        cv2.fillPoly(mask_image, [mask_polygon], 255)

        # Convert the BGR image to BGRA, adding an alpha channel
        object_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # Set pixel values to (255, 255, 255, 0) where the mask is 0
        object_bgra[mask_image == 0] = np.array([255, 255, 255, 0], dtype=np.uint8)

        return object_bgra
    

    def detect_color(self, object_image, coding):
        """ Detect the main colors of the object in the image.
        
        :param object_image: The image of the object to detect the colors of.
        :param coding: The coding of the image, either 'BGR', 'RGB', 'BGRA' or 'RGBA'.

        """

        # Find the main colors of the object using the ColorPrediction class.
        # From the uugai_python_color_prediction.ColorPrediction dependency.
        optimal_k, kmeans_data = ColorPrediction.find_main_colors(
            image = object_image,
            coding = coding,
            min_clusters = self.min_clusters, 
            max_clusters = self.max_clusters, 
            downsample_factor = self.downsample_factor, 
            increase_elbow = self.increase_elbow, 
            )
        
        # If no optimal_k is found, return an empty array.
        if optimal_k is None:
            return np.array([])
        
        # Get the BGR colors of the centroids.
        # Return the BGR colors as a numpy array.
        bgr_colors = kmeans_data[optimal_k]['centroids']
        return np.array(bgr_colors)


    def hls_to_str(self, given_hls):
        """ Convert HLS to string.
        This is done using a slightly customised version of the HSL-79 color naming system.
        A good representatin can be found at https://www.chilliant.com/colournames.html

        :param given_hls: The HLS color to convert to string.

        """

        # Extract the HLS values from the given HLS color.
        h = given_hls[0]*2
        l = given_hls[1]/255
        s = given_hls[2]/255

        # Define the saturation and lightness values for the color naming system.
        S1 = 0.28
        S2 = 0.51
        L1 = 0.12
        L2 = 0.24
        L3 = 0.44
        L4 = 1-L3
        L5 = 1-L2
        L6 = 1-L1

        # Determine the color name based on the HLS values.
        if l < L1:
            return 'black'
        elif l > L6:
            return 'white'
        
        if l < L3:
            prefix = 'dark'
        elif l > L4:
            prefix = 'light'
        else:
            prefix = ''
        
        if s < S1:
            return prefix + ' ' + 'grey' if prefix != '' else 'grey'
        elif s < S2:
            prefix = 'dull'

        if h < 15 or h > 345:
            color = 'red'
        elif 15 <= h < 45:
            color = 'orange'
        elif 45 <= h < 75:
            color = 'yellow'
        elif 75 <= h < 105:
            color = 'chartreuse'
        elif 105 <= h < 135:
            color = 'green'
        elif 135 <= h < 165:
            color = 'spring'
        elif 165 <= h < 195:
            color = 'cyan'
        elif 195 <= h < 225:
            color = 'azure'
        elif 225 <= h < 255:
            color = 'blue'
        elif 255 <= h < 285:
            color = 'violet'
        elif 285 <= h < 315:
            color = 'magenta'
        elif 315 <= h < 345:
            color = 'rose'
        
        return prefix + ' ' + color if prefix != '' else color

    
    def bgr_to_hls(self, bgr_color):
        """ Convert BGR to HLS.
        
        :param bgr_color: The BGR color to convert to HLS.

        """

        hls_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HLS)[0][0]
        return hls_color.tolist()
    

    def crop_and_detect(self, frame, trajectory, mask_polygon = None):
        """ Crop the object from the image and detect the colors of the object.

        :param frame: The image to crop the object from.
        :param trajectory: The trajectory of the object in the image.
        :param mask_polygon: The mask polygon of the object in the image.

        """

        # If no mask_polygon is given, crop the object from the image.
        # Otherwise, segment the object from the background.
        if mask_polygon is None:
            cropped_image = self.crop_detected_object(frame, trajectory)
            bgr_centroid_colors = self.detect_color(cropped_image, 'BGR').tolist()
        else:
            cropped_image = self.segment_object(frame, mask_polygon)
            bgr_centroid_colors = self.detect_color(cropped_image, 'BGRA').tolist()

        hls_centroid_colors = []
        str_centroid_colors = []

        # Convert the BGR colors to HLS and string.
        # Append the HLS and string colors to the respective lists.
        for bgr_color in bgr_centroid_colors:
            hls_color = self.bgr_to_hls(bgr_color)
            hls_centroid_colors.append(hls_color)
            color_name = self.hls_to_str(hls_color)
            str_centroid_colors.append(color_name)

        return bgr_centroid_colors, hls_centroid_colors, str_centroid_colors
