# import datetime
# import time
# import os

# class Logger:

#     def __init__(self):
#         # Initialize file in /log
#         fileName = "raw_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) +'.log'
#         log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         self.log_file_path = os.path.join(log_dir, fileName)
#         self.log_out = open(self.log_file_path, 'a', buffering=1) # line buffered file write
#         self.log_out.write


#     def message(self, dataFrame):
#         formatedData = "{\"ts\": " + str(int(time.time()*1000)) +", \"dataFrame\": " + str(dataFrame) + "}"
#         self.data[str(int(time.time()*1000))] = 
#         self.log_out.write(formatedData)
#         self.log_out.write("\n")

#     def close(self):
#         self.log_out.close()