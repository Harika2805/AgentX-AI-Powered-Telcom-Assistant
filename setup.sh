#!/bin/bash

# ========================================================
# 🚀 Telecom Chatbot Environment Setup Script
# ========================================================

echo "=== Creating Conda environment ==="
conda create --name chat_bot python==3.10 python-dotenv -y

echo "=== Initializing Conda ==="
conda init

echo "=== Activating environment ==="
# Note: If running inside Codespaces or CI, you may not need to close terminal

source ~/.bashrc
conda activate chat_bot

echo "=== Installing Jupyter tools ==="
conda install -y ipykernel jupyter jupyterlab

echo "=== Installing LlamaIndex and related dependencies ==="
pip install \
  llama-index==0.14.6 \
  llama-index-core==0.14.6 \
  llama-index-embeddings-openai==0.5.1 \
  llama-index-llms-openai==0.6.6 \
  llama-index-readers-file==0.5.4 \
  openai==1.109.1

echo "=== Installing Qdrant client and LlamaIndex Qdrant connector ==="
pip install \
  qdrant-client==1.15.1 \
  llama-index-vector-stores-qdrant==0.8.6

echo "=== Installing Cohere integrations ==="
pip install -U cohere
pip install -U \
  llama-index-llms-cohere \
  llama-index-embeddings-cohere \
  llama-index-postprocessor-cohere-rerank

echo "=== Registering environment as Jupyter kernel ==="
python -m ipykernel install --user --name=basic_chatbot --display-name "Basic_chatbot"

echo "=== Installing Streamlit ==="
pip install streamlit

echo "✅ Setup complete!"
echo "You can now activate your environment using:"
echo "👉 conda activate chat_bot"
echo "Then run your Streamlit app with:"
echo "👉 streamlit run app.py"