import json
import numpy as np

def make_it_dict(string):
    s = string.replace("'",'"')
    return json.loads(s)

if __name__ == "__main__":
    
    instances = []
    
    with open("pymmw_data.log",'r') as data:
        entries = data.readlines()
    
    for line in entries:
        instances.append(make_it_dict(line))
        break
    print(instances[0]["dataFrame"]["detected_points"])
    # print(make_it_dict(instances[0]["dataFrame"])["detected_points"])