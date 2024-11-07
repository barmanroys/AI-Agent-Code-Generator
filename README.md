## Goal

Run an agentic RAG

# Steps:

Install OLLAMA followed by pip dependencies

```commandline
curl -fsSL https://ollama.com/install.sh | sh # This installation fires up the server at port 11434
time python3 -m pip install --requirement requirement.txt
# Pull the model locally 
ollama pull mistral
ollama pull codellama 
```
