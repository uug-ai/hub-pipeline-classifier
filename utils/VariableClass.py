import os
from dotenv import load_dotenv



class VariableClass:
    """This class is used to store all the environment variables in a single class. This is done to make it easier to access the variables in the code.
    
    """

    def __init__(self):
        """This function is used to load all the environment variables and store them in the class.

        """
        
        # Environment variables
        load_dotenv()

        # Model parameters
        self.MODEL_NAME = os.getenv("MODEL_NAME")
        self.MEDIA_SAVEPATH = os.getenv("MEDIA_SAVEPATH")

        # Queue parameters
        self.QUEUE_NAME = os.getenv("QUEUE_NAME")
        self.TARGET_QUEUE_NAME = os.getenv("TARGET_QUEUE_NAME")
        self.QUEUE_EXCHANGE = os.getenv("QUEUE_EXCHANGE")
        self.QUEUE_HOST = os.getenv("QUEUE_HOST")
        self.QUEUE_USERNAME = os.getenv("QUEUE_USERNAME")
        self.QUEUE_PASSWORD = os.getenv("QUEUE_PASSWORD")

        # Kerberos Vault parameters
        self.STORAGE_URI = os.getenv("STORAGE_URI")
        self.STORAGE_ACCESS_KEY = os.getenv("STORAGE_ACCESS_KEY")
        self.STORAGE_SECRET_KEY = os.getenv("STORAGE_SECRET_KEY")

        # Feature parameters
        # The == "True" is used to convert the string to a boolean.
        self.PLOT = os.getenv("PLOT") == "True"

        self.TIME_VERBOSE = os.getenv("TIME_VERBOSE") == "True"

        self.LOGGING = os.getenv("LOGGING") == "True"

        self.CREATE_BBOX_FRAME = os.getenv("CREATE_BBOX_FRAME") == "True"
        self.SAVE_BBOX_FRAME = os.getenv("SAVE_BBOX_FRAME") == "True"
        self.BBOX_FRAME_SAVEPATH = os.getenv("BBOX_FRAME_SAVEPATH")
        if self.SAVE_BBOX_FRAME:
            self.CREATE_BBOX_FRAME = True

        self.CREATE_RETURN_JSON = os.getenv("CREATE_RETURN_JSON") == "True"
        self.SAVE_RETURN_JSON = os.getenv("SAVE_RETURN_JSON") == "True"
        self.RETURN_JSON_SAVEPATH = os.getenv("RETURN_JSON_SAVEPATH")
        if self.SAVE_RETURN_JSON:
            self.CREATE_RETURN_JSON = True

        self.SAVE_VIDEO = os.getenv("SAVE_VIDEO") == "True"
        self.OUTPUT_MEDIA_SAVEPATH = os.getenv("OUTPUT_MEDIA_SAVEPATH")

        self.FIND_DOMINANT_COLORS = os.getenv("FIND_DOMINANT_COLORS") == "True"
        self.COLOR_PREDICTION_INTERVAL = int(os.getenv("COLOR_PREDICTION_INTERVAL"))
        self.MIN_CLUSTERS = int(os.getenv("MIN_CLUSTERS"))
        self.MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS"))

        # Classification parameters
        self.CLASSIFICATION_FPS = int(os.getenv("CLASSIFICATION_FPS"))
        self.CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD"))
        self.MAX_NUMBER_OF_PREDICTIONS = int(os.getenv("MAX_NUMBER_OF_PREDICTIONS"))
        self.MIN_DISTANCE = int(os.getenv("MIN_DISTANCE"))
        self.MIN_STATIC_DISTANCE = int(os.getenv("MIN_DISTANCE"))
        self.MIN_DETECTIONS = int(os.getenv("MIN_DETECTIONS"))
        ALLOWED_CLASSIFICATIONS_STR = os.getenv("ALLOWED_CLASSIFICATIONS")
        self.ALLOWED_CLASSIFICATIONS = [int(item.strip()) for item in ALLOWED_CLASSIFICATIONS_STR.split(',')]
        TRANSLATED_CLASSIFICATIONS_STR = os.getenv("ALLOWED_CLASSIFICATIONS")
        self.TRANSLATED_CLASSIFICATIONS = [item.strip() for item in TRANSLATED_CLASSIFICATIONS_STR.split(',')]