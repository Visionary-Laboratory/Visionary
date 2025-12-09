# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-20 14:38:28
# @Function      : auto download class

import os
import pdb
import subprocess
import sys

from LHM.utils.model_card import HuggingFace_MODEL_CARD, ModelScope_MODEL_CARD

package_name='huggingface_hub'

try:
    from huggingface_hub import snapshot_download as hf_snapshot
except:
    print(f"{package_name} is not installed. installing now...")
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
    print(f"{package_name} has been installed.")

    from modelscope import snapshot_download as hf_snapshot

package_name = "modelscope"  
try:
    from modelscope import snapshot_download as ms_snapshot
except:
    print(f"{package_name} is not installed. installing now...")
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
    print(f"{package_name} has been installed.")

    from modelscope import snapshot_download as ms_snapshot

class AutoModelQuery:
    def __init__(self, save_dir='./pretrained_models', hf_kwargs=None, ms_kwargs=None):
        """ 
        :param save_dir: 
        """
        self.hf_save_dir = os.path.join(save_dir, 'huggingface')
        self.ms_save_dir = save_dir


        self.logger = lambda x: "\033[31m{}\033[0m".format(x)
    
    def query_huggingface_model(self, model_name):
        try:
            model_repo_id= HuggingFace_MODEL_CARD[model_name]
            model_path = hf_snapshot(repo_id=model_repo_id, cache_dir=self.hf_save_dir)
            return model_path
        except:
            print(self.logger("Cannot download from Hugging Face; try using ModelScope instead!"))
            raise FileNotFoundError

    def query_modelscope_model(self, model_name):
        """ model_name: query model_name
        """

        def get_max_step_folder(current_path):
            step_folders = [f for f in os.listdir(current_path) if f.startswith('step_')]
            if len(step_folders) == 0:
                return current_path
            else:
                max_folder = max(step_folders, key=lambda x: int(x.split('_')[1]), default=None)
                return os.path.join(current_path, max_folder) if max_folder else None
        try:
            model_repo_id = ModelScope_MODEL_CARD[model_name]
            model_path = get_max_step_folder(ms_snapshot(model_repo_id, cache_dir=self.ms_save_dir))
            return model_path
        except:
            raise FileNotFoundError("fail to download model, DO you download the model?")

    def query(self, model_name):
        """ model_name: query model_name
        """

        assert model_name in HuggingFace_MODEL_CARD.keys(), f"only support model_name: {HuggingFace_MODEL_CARD.keys()}!"
        try:
            model_path = self.query_huggingface_model(model_name)
        except:
            model_path = self.query_modelscope_model(model_name)
        model_path = model_path +'/' if model_path[-1] !='/' else model_path
        return model_path
        
if __name__ == '__main__':
    automodel = AutoModelQuery()
    model_path = automodel.query('LHM-1B-HF')
    print(model_path)


