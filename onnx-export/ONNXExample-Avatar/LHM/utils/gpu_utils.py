# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-02 13:33:56
# @Function      : GPU utils 

import subprocess
import sys

try:
    import GPUtil
except:
    package_name = "GPUtil"
    print(f"{package_name} is not installed. installing now...")
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
    print(f"{package_name} has been installed.")
    import GPUtil 

def check_single_gpu_memory(threshold_gb=24, gpu_id=0):

    gpus = GPUtil.getGPUs()
    
    if not gpus:
        print("No GPU found.")
        return False

    if gpu_id >= len(gpus):
        print(f"GPU ID {gpu_id} is out of range. There are only {len(gpus)} GPUs.")
        return False

    gpu = gpus[gpu_id]  
    total_memory = gpu.memoryTotal  
    used_memory = gpu.memoryUsed   

    available_memory = total_memory - used_memory
    available_memory_gb = available_memory / 1024  

    print(f"GPU ID: {gpu.id}, Total Memory: {total_memory} MB, Used Memory: {used_memory} MB, Available Memory: {available_memory_gb:.2f} GB")
    
    if available_memory_gb < threshold_gb:
        return False 
    else:
        return True