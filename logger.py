import sys
import json
from datetime import datetime as dt

class Logger():
    def __init__(self, logs_folder="logs", models_folder="models",
                 output_folder="output", data_folder="data",
                 show=False, verbosity_level=0, html_output=False,
                 config_file="config.txt"):
        sys.stdout.flush()
        #print(self.get_time() + " Initialize the logger")
        self.internal_clock = dt.now()
        self.logs_folder = logs_folder
        self.models_folder = models_folder
        self.output_folder = output_folder
        self.data_folder = data_folder
        self.show = show
        self.verbosity_level = verbosity_level
        self.html_output = html_output
        self.config_file = config_file

        '''
        with open(self.config_file) as fp:
            self.config_dict = json.load(fp)
        #print(self.get_time() + " Read config file {}".format(config_file))
        print(" Read config file {}".format(config_file))
        '''

    def log(self, str_to_log, show=None, tabs=0, verbosity_level=0, show_time=False):
        sys.stdout.flush()