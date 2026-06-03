# nanoNLA

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/2e6dc1c6-d998-4e57-807a-4ce4a5f288e2" />

(enjoy gpt image 2.0 reading of this repo)

This is a minimal reimplementatoin of [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html). 

Starting from their [implementation](https://github.com/kitft/natural_language_autoencoders), I here share a minimal version that is sufficient to train NLAs on small models (no SGLang) and should lead to significantly reduced infra hassle.

**The code contains the code to warmstart an AV and an AR, and co-train them using RL**

**The warmstart dataset can be found [here](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k)** 

Hyperparameters are identical to the paper.

A Natural Language Autoencoder is a pair of fine-tuned LMs that map
residual-stream activation vectors to natural language and back.

(I may or may not extend this with evals, for ease of hillclimbing)

## Claude instructions for usage, extra changes, and info for agents

TODO claude.
