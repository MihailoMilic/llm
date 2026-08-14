conda create -n llm python=3.11 -y
conda activate llm
pip install "torch==2.2.2" "numpy<2" tiktoken matplotlib
pip freeze > requirements.txt