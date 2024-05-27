from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import torch
import re
from huggingface_hub import login
from tqdm import tqdm
login('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')
model_id = 'google/gemma-7b-it'

tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir = '/workspace/models--google--gemma-7b-it/snapshots/18329f019fb74ca4b24f97371785268543d687d2'
    )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir = '/workspace/models--google--gemma-7b-it/snapshots/18329f019fb74ca4b24f97371785268543d687d2'
    
)

peft_model_id = '/workspace/result_135/adapter'
model.load_adapter(peft_model_id)



def instruction(origin_sentence, source_lang="English", target_lang="Vietnamese"):
    prompt = f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {origin_sentence}\n{target_lang}:"
    return prompt


def generate(text):
    prompt = instruction(str(text))
    # print(prompt)
    temperature = 0.1
    top_k = 50
    top_p = 0.95
    max_new_tokens = len(str(text)) 
    # print(max_new_tokens)
    min_new_tokens = -1
    repetition_penalty = 1
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
                        early_stopping=True,
                        penalty_alpha= 1,
                          num_beams = 5,
                         )
    
    text = tokenizer.decode(target[0])
    try:
        predicted_text = text.split("\n")[-1].split(":")[1].split(".")[0]+"."
    except :
        predicted_text = text.split("\n")[-1].split(":")[0].split(".")[0]+"."
    finally:
        predicted_text = ""
    return (text, predicted_text)

if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    bad_results = []
    good_results = []
    for i in tqdm(range(len(df))):
        text = df['origin_text'].tolist()[i]
        res = generate(text) 
        
        if len(res) == 2:
            bad_res, good_res = res[0], res[1]
        elif len(res) == 1:
            bad_res, good_res = res[0], ''
        bad_results.append(bad_res)
        good_results.append(good_res)

        if i!= 0 and i % 2000 == 0:
            pd.DataFrame(
                {
                    "bad_results" : bad_results,
                    "good_resuls" : good_results
                }
            ).to_csv(f"infer135/result_{i}.csv", index=False)
        
    df["bad results"] = bad_results
    df["good results"] = good_results
    df.to_csv("infer135/results_trans.csv", index=False)
    