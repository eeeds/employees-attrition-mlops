import json


class Keys:
    """
    This class is used to obtain the keys for the various APIs.
    """

    def __init__(self):
        pass

    def obtain_whylogs_key(self):
        """
        This function obtains the WhyLogs key.
        """
        with open('whylog_token.json', 'r') as f:
            token = json.load(f)['key']
        self.whylog_key = token
