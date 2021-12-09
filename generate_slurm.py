import os
import re

f = open("TE_TR_TI_control_T1.csv","r")
path = "/w/246/gzk/PPMI/MRI_"
lines = f.readlines()
for line in lines:
    variable = line.split(",")
    group = variable[1]
    if group == "Control":
        group = "control"
    subjectIdentifier = variable[2]
    seriesIdentifier = variable[5]
    folderpath = path+group+"/PPMI/"+subjectIdentifier
    pattern = "S"+seriesIdentifier
    for root, directories, filenames in os.walk(folderpath): 
        for filename in filenames:  
            filepath = os.path.join(root,filename)
            print(filepath)
            if re.match(pattern, filepath):
                print(filepath)
