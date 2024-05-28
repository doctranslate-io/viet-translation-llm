from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os 
from config.base_config import InferConfig

class Model(InferConfig):
    def __init__(self, config_path : str):
        InferConfig.__init__(self, config_path=config_path)
        login(self.hf_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname, cache_dir = self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname,device_map="auto", 
                                                        torch_dtype=torch.bfloat16, cache_dir = self.cache_dir)
        self.model.load_adapter(self.adapter)  
        
