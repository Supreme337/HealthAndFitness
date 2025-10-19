import sys
from healthandfitness.logging import logger
class HealthAndFitnessException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message
        exc_traceback=error_details.exc_info()[2]
        self.lineno=exc_traceback.tb_lineno
        self.file_name=exc_traceback.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(self.file_name,self.lineno,str(self.error_message))    

