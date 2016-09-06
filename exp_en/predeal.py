#!/usr/bin/env python
# coding=utf-8

'''
    local cmd, handle
    cmd = "g++ ./predeal_dataset_big-data.cpp -o predeal_dataset_big" print(cmd)
    handle = io.popen("rm predeal_dataset_big") handle = io.popen(cmd) waitFile("predeal_dataset_big")
    cmd = "g++ ./implus_x_y_.cpp -o implus" print(cmd)
    handle = io.popen("rm implus") handle = io.popen(cmd) waitFile("implus")
    waitFile("./predeal_dataset_big")
    cmd = "sudo ./predeal_dataset_big "..data_path.."/train.txt "..data_path.."/valid.txt "..data_path.."/test.txt "..data_path.."/" print(cmd)
    handle = io.popen("rm "..data_path.."/sortmapxy.txt") handle = io.popen(cmd) waitFile(data_path.."/sortmapxy.txt")
'''
import os, sys

if len(sys.argv) < 3:
    print("useage: python predeal.py ../data/ addEnd(or notAddEnd)")
    exit(1)

data_path = sys.argv[1]
end_str   = sys.argv[2]
cmd = "g++ ./predeal_dataset.cpp -o predeal_dataset" 
print(cmd)
os.system(cmd)


cmd = "./predeal_dataset "+data_path+"/train.txt "+data_path+"/valid.txt "+data_path+"/test.txt "+data_path+"/"
if end_str == "addEnd":
    cmd += " addEnd"
print(cmd)
os.system(cmd)


#cmd = "g++ ./MCMF_x_y_.cpp -o implus" 
cmd = "g++ ./implus_x_y_.cpp -o implus"
print(cmd)
os.system(cmd)

print("----------------------------------")
print("   python data prepare finished!  ")
print("----------------------------------")


