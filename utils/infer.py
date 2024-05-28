import pandas as pd
import re
from tqdm import tqdm
from build_model import Model
import argparse
from loguru import logger
def postprocess(text, prompt):
    
    text = text.replace("<bos>","").replace("<eos>", "")

    text = text[len(prompt):].split("\n")[0]
    if len(text) > 1:
        text = text.split(".")[0]
    else :
        text = text
    return text.strip()

def instruction(origin_sentence, source_lang="English", target_lang="Vietnamese"):
    prompt = f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {origin_sentence}\n{target_lang}:"
    return prompt

def eval(text, model,tokenizer,
         num_beams : int = 5, early_stopping : bool = True,
        source_lang : str = "English", target_lang : str = "Vietnamese",
        max_new_tokens : int = 512, min_new_tokens : int = -1, 
        temperature : float = 0.1, no_repeat_ngram_size : int = 3,
        top_k : int = 50, top_p : float = 0.95, repetition_penalty : float = 1):
    
    prompt = instruction(str(text), source_lang, target_lang)
    max_new_tokens = len(str(text))
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    target=model.generate(**input_ids.to(model.device),
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        use_cache=True,
                        early_stopping=early_stopping,
                        penalty_alpha= 1,
                        num_beams = num_beams,
                        no_repeat_ngram_size = no_repeat_ngram_size
                        )
    
    text = tokenizer.decode(target[0])
    text = postprocess(text = text, prompt = prompt)
    
    return text

def handle_io(eval_func):
    def decorator(func):
        def wrapper(path_file, output_path, model, tokenizer, **kwargs):
            if path_file.endswith(".txt"):
                with open(path_file, "r", encoding='utf-8') as f:
                    text = f.readlines()
                content = [x.strip() for x in text]
                df = pd.DataFrame(content, columns=["origin_text"])
            elif path_file.endswith(".csv"):
                df = pd.read_csv(path_file)
                col = df.columns
                if col[0] != "origin_text":
                    df = df.rename(columns={col[0]: "origin_text"})
            else:
                raise ValueError("Unsupported file format. Please use .txt or .csv files.")
            
            df = func(df, eval_func, model, tokenizer, **kwargs)

            df.to_csv(output_path, index=False)
            return logger.info(f"Translated file is saved at {output_path}")
        return wrapper
    return decorator

@handle_io(eval_func=eval)
def eval_file(df, eval_func, model, tokenizer, **kwargs):
    for i in tqdm(range(len(df))):
        text = df['origin_text'].tolist()[i]
        res = eval_func(text, model, tokenizer, **kwargs)
        df.loc[i, "translated_text"] = res
    return df

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, default='config/infer.yaml', help='Path to config file')
    args = parser.parse_args()
    
    docs_translate = Model(args.config)
    model = docs_translate.model
    tokenizer = docs_translate.tokenizer
    
    if docs_translate.text is not None:
        logger.info("Translating text...")
        text = docs_translate.text
        text = eval(text, model, tokenizer, source_lang= docs_translate.source_lang, target_lang= docs_translate.target_lang, max_new_tokens = docs_translate.max_new_tokens,
                    min_new_tokens = docs_translate.min_new_tokens, 
                    temperature  = docs_translate.temperature, 
                    top_k = docs_translate.top_k, top_p = docs_translate.top_p , 
                    repetition_penalty = docs_translate.repetition_penalty)
        logger.info(f"Translated text: {text}")
    if docs_translate.file_path is not None : 
        logger.info("Translating file...")
        kwargs = {
            "source_lang" : docs_translate.source_lang,
            "target_lang" : docs_translate.target_lang,
            "max_new_tokens" : docs_translate.max_new_tokens,
            "num_beams" : docs_translate.num_beams,
            "early_stopping" : docs_translate.early_stopping,
            "repetition_penalty" : docs_translate.repetition_penalty,
            "min_new_tokens" : docs_translate.min_new_tokens,
            "top_k" : docs_translate.top_k,
            "top_p" : docs_translate.top_p,
            "temperature" : docs_translate.temperature
        }
        eval_file(docs_translate.file_path, docs_translate.output_path , model, tokenizer, **kwargs)
        