import requests
import json


class KerberosVault:
    """A class for connecting with the Kerberos Vault.
    """

    def __init__(self, storage_uri, storage_access_key, storage_secret_key):
        """ Initializes the QueueProcessor object with necessary attributes.

        Parameters:
        -----------
        storage_uri : str
            The URI of the storage service.
        storage_access_key : str
            The access key for the storage service.
        storage_secret_key : str
            The secret key for the storage service.

        """

        self.storage_uri = storage_uri
        self.storage_access_key = storage_access_key
        self.storage_secret_key = storage_secret_key

    def update_storage_info(self, data):
        """ Updates storage-related information based on data received from message payloads.

        Parameters:
        -----------
        data : dict
            A dictionary containing storage-related information.

        """

        if "storage_uri" in data and data['storage_uri'] != "":
            self.storage_uri = data['storage_uri']
        if "storage_access_key" in data and data['storage_access_key'] != "":
            self.storage_access_key = data['storage_access_key']
        if "storage_secret" in data and data['storage_secret'] != "":
            self.storage_secret = data['storage_secret']

    def create_headers(self, file_name, storage_provider):
        """ Creates and returns headers required for accessing storage service based on message payload information.

        Parameters:
        -----------
        file_name : str
            The name of the file to be stored.
        storage_provider : str
            The name of the storage provider.

        """

        self.headers = {
            'Content-Type': 'application/json',
            'X-Kerberos-Storage-FileName': '{0}'.format(file_name),
            'X-Kerberos-Storage-Provider': '{0}'.format(storage_provider),
            'X-Kerberos-Storage-AccessKey': '{0}'.format(self.storage_access_key),
            'X-Kerberos-Storage-SecretAccessKey': '{0}'.format(self.storage_secret_key),
        }

        return self.headers

    def retrieve_media(self, message: str, media_type: str = '', media_savepath: str = ''):
        """ Fetches associated data from storage, and performs actions.

        Parameters:
        -----------
        message : str
            The message to be processed.
        media_type : str
            The type of message to be processed (image, video). Default is an empty string.
            if type is 'video', the message is processed and video file is created. Filepath is returned.
            else, the message response is returned.
        media_savepath : str
            The path where the media file is to be saved. Default is an empty string.

        """

        # Update storage-related information if available in message payload
        if "data" in message:
            self.update_storage_info(message['data'])

        # Create headers for accessing storage service
        headers = self.create_headers(
            message['payload']['key'], message['source'])

        resp = requests.get(self.storage_uri +
                            "/storage", headers=headers, timeout=10)

        if resp is None or resp.status_code != 200:
            return resp.content

        objData = resp.content
        jsonPayload = json.loads(objData)
        if 'data' in jsonPayload:
            mediaUrl = jsonPayload['data']
        else:
            mediaUrl = ""

        # check if has a data field which should contain the mediaurl
        if mediaUrl != "":
            # Parse to json
            print("mediaUrl: ", mediaUrl)

            # Read from signed url
            resp = requests.get(mediaUrl, timeout=10)
            if resp.status_code != 200:
                print("Error: an exception occurred: ", resp.content)
                return resp.content

            objData = resp.content

            # From the received requested data, reconstruct a video-file.
            # This creates a video-file in the data folder, containing the recording.
            with open(media_savepath, 'wb') as output:
                output.write(resp.content)

        return None
