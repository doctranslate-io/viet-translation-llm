from omegaconf import DictConfig, OmegaConf

class InferConfig:
    def __init__(self , config_path) :
        self._cfg = OmegaConf.load(config_path)
        self._infer_cfg = self._cfg.infer
        if self._infer_cfg is not None:
            self.modelname = self._infer_cfg.modelname
            self.adapter = self._infer_cfg.adapter
            self.cache_dir = self._infer_cfg.cache_dir
            self.hf_tokens = self._infer_cfg.hf_tokens
            self.device = self._infer_cfg.device
            self.top_k = self._infer_cfg.top_k
            self.top_p = self._infer_cfg.top_p
            self.temperature = self._infer_cfg.temperature
            self.repetition_penalty = self._infer_cfg.repetition_penalty
            self.max_new_tokens = self._infer_cfg.max_new_tokens
            self.min_new_tokens = self._infer_cfg.min_new_tokens
            self.length_penalty = self._infer_cfg.length_penalty
            self.num_beams = self._infer_cfg.num_beams
            self.early_stopping = self._infer_cfg.early_stopping
            self.use_cache = self._infer_cfg.use_cache
            self.do_sample = self._infer_cfg.do_sample
            self.output_path = self._infer_cfg.output_path
            
        self.translate = self._cfg.translate
        if self.translate is not None:
            self.source_lang = self.translate.source_lang
            self.target_lang = self.translate.target_lang
            self.file_path = self.translate.file_path
            self.text = self.translate.text
