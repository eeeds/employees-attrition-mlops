import os
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
        try:
            with open('whylog_token.json', 'r') as f:
                token = json.load(f)['key']
            self.whylog_key = token
        except FileNotFoundError:
            print('The file whylog_token.json was not found.')
            self.whylog_key = os.environ('WHYLOGS_ACCOUNT_KEY')
