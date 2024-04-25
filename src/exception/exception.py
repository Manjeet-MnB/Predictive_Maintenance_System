import os
import sys

# error message , #error details
class CustomException(Exception):
    #error_details from sys module
    def __init__(self,error_message:Exception,error_details:sys):
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,error_details=error_details)

    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_details:sys)->str:

#error details

        # when returning we will get three things but only 1 of them is importaant for us
        _, _, exec_tb= error_details.exc_info()

        #we need exception like in which the error is occured
        exception_block_line_number=exec_tb.tb_frame.f_lineno
 
        #for try block
        try_block_line_number=exec_tb.tb_lineno

        # in which file we get error and which code we get issue
        file_name=exec_tb.tb_frame.f_code.co_filename

# error message

        error_message = f"""
        Error occured in script: 
        [ {file_name} ] at 
        try block line number: [{try_block_line_number}]
        and exception block line number: [{exception_block_line_number}] 
        error message: [{error_message}]
        """
        return error_message
    
#The __str__() method returns a human-readable, or informal, string representation of an object. 
# This method is called by the built-in print(), str(), and format() functions.
    def __str__(self):
        return self.error_message

#The __repr__() method returns a more information-rich, or official, string representation of an object.
#  This method is called by the built-in repr() function.
#  If possible, the string returned should be a valid Python expression that can be used to recreate the object.

    def __repr__(self) -> str:
        return CustomException.__name__.str()