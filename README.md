# viet-translation-llm


This project provides a solution for translating text from one language to another using Gemma-7b-it. The project includes steps for setting up the environment, configuring the inference parameters, and running the translation.

## Getting Started

Follow the steps below to set up and run the project.

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone <repository_url>
cd <repository_directory>
```

### Install requirements
Install the necessary packages using **requirements.txt**:

```bash
pip install -r requirements.txt
```

### Custom Inference Configuration
Customize the inference configuration file **(infer.yaml)** according to your requirements. The infer.yaml file contains parameters for the translation model and other settings. Below is an example configuration:

```
infer:
  modelname: "google/gemma-7b-it"           # The name of the model to use for translation
  adapter: "path adapter"                   # Path to the adapter, if any
  output_dir: "./translated.csv"            # Output file path for the translated text
  cache_dir: "./"                           # Directory for caching model files
  hf_tokens: "hf_tokens"                    # Hugging Face API tokens
  max_new_tokens: 512                       # Maximum number of tokens to generate
  num_beams: 5                              # Number of beams for beam search
  early_stopping: True                      # Whether to stop early when the model is confident
  no_repeat_ngram_size: 3                   # Size of n-grams that should not be repeated
  repetition_penalty: 1                     # Penalty for repeating tokens
  min_new_tokens: -1                        # Minimum number of tokens to generate
  length_penalty: 2.0                       # Penalty for the length of the generated text
  device: "cuda"                            # Device to use for inference (e.g., 'cuda' or 'cpu')
  top_k: 50                                 # The number of highest probability vocabulary tokens to keep for top-k-filtering
  top_p: 0.95                               # Cumulative probability for top-p-filtering
  temperature: 0.1                          # Sampling temperature

translate:
  source_lang: "English"                    # Source language
  target_lang: "Vietnamese"                 # Target language
  file_path:                                # Path to the file (txt or csv) with the text to translate
  text: "Hi, I'm doc translate bot"         # Text to translate if no file is provided

```

### Running the Translation
Run the translation script with the customized configuration file:

```bash
python utils/infer.py --config config/infer.yaml
```

## Repository Structure
config/: Contains configuration files.
utils/: Contains utility scripts, including the inference script.
requirements.txt: Lists the required Python packages for the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Special thanks to all contributors and the open-source community for their support and contributions.