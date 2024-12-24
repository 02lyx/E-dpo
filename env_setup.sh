conda init
conda env create -f gen-eval.yaml
conda init
eval "$(conda shell.bash hook)"

conda create -n anno-train python=3.10.9 --yes
conda activate anno-train

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.34.0
pip install huggingface-hub==0.24.7
pip install transformers==4.45.0
pip install trl==0.11.1
pip install wandb==0.17.7
pip install numpy==1.26.4
pip install antlr4-python3-runtime
pip install distro
pip install deepspeed==0.15.4
cd ..