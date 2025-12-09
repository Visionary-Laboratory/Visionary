import pdb
import subprocess
import sys

import cv2
from PIL import Image

try:
    import imagehash
except:
    package_name = "imagehash"
    print(f"{package_name} is not installed. installing now...")
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
    print(f"{package_name} has been installed.")
    import imagehash

def get_video_hash(video_path):
    cap = cv2.VideoCapture(video_path)
    hashes = []


    total_hash_codes = 5  # only sample 5 frames as hash input.

    remain_codes =5 
    cnt = 0
    hash_cnt = 0

    
    while True:
        ret, frame = cap.read()
        if hash_cnt == total_hash_codes or ret==None:
            break
        
        if cnt % remain_codes == 0:
        
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pil_image = Image.fromarray(gray_frame)
            frame_hash = imagehash.average_hash(pil_image)
            hashes.append(str(frame_hash))
            hash_cnt+=1
        
        cnt += 1


    cap.release()
    video_hash = "_".join(hashes)

    return video_hash


def check_single_gpu_memory(threshold_gb=24, gpu_id=0):
    try:
        import GPUtil
    except:
        package_name = "GPUtil"
        print(f"{package_name} is not installed. installing now...")
        
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"{package_name} has been installed.")
        import GPUtil 

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