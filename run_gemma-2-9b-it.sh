git clone https://github.com/02lyx/E-dpo.git
cd E-dpo

bash env_setup.sh

export WANDB_API_KEY=6f9e1eaf73cd08b4f0cd4674c7856201f2453428
# export HF_HOME="/projects/p32646/hf_cache"
huggingface-cli login  --token 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

bash gemma-2-9b-it.sh

huggingface-cli upload Yuanxin-Liu/Gemma-2-9b-it-e-dpo Gemma-2-9b-it_iter3 --token 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