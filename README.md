# NLP
Transformer Language Model

Overview

This project implements a character-level transformer language model using PyTorch. It is trained on the Tiny Shakespeare dataset to generate Shakespeare-style text. The implementation includes tokenization, model architecture, training, and inference.

Features
	•	Character-level Tokenization: Maps unique characters to integers and vice versa.
	•	Transformer-based Architecture: Uses multi-head self-attention and feed-forward layers.
	•	Training Pipeline: Implements dataset preparation, batching, loss estimation, and training.
	•	Text Generation: Generates new text sequences from trained models.
	•	Byte Pair Encoding (BPE) Support: Implements a subword tokenization technique.
	•	Rotary Positional Encoding (RoPE): Improves long-range dependencies in transformer models.

Requirements
	•	Python 3.x
	•	PyTorch 2.2.0 
	•	NumPy 1.26.4

Installation
	1.	Clone the repository:

.....


	2.	Install dependencies:

pip install torch numpy 1.26.4



Usage

Download and Prepare Data

Download the Tiny Shakespeare dataset:

wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --no-check-certificate

Load and tokenize the dataset:

text = open('input.txt', 'r').read()

Train the Model

Run the training script:

python train.py

Training parameters include:
	•	batch_size: Number of sequences processed in parallel.
	•	block_size: Maximum context length for predictions.
	•	n_head: Number of attention heads.
	•	n_layer: Number of transformer layers.

Generate Text

Generate text from a trained model:

python generate.py --max_tokens 500

or in Python:

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))

Model Details

Transformer Components
	•	Multi-Head Attention: Computes attention across multiple heads for better representation learning.
	•	Feed-Forward Networks: Expands and projects embeddings using non-linear transformations.
	•	Layer Normalization (RMSNorm): Normalizes input activations for stable training.
	•	Rotary Positional Encoding (RoPE): Enhances positional representations.

Training Configuration
	•	Optimizer: AdamW
	•	Learning Rate: 1e-3
	•	Dropout: 0.1
	•	Loss Function: CrossEntropyLoss

File Structure

.
├── data/
│   ├── input.txt             # Shakespeare dataset
├── model/
│   ├── transformer.py        # Transformer model implementation
│   ├── tokenizer.py          # Character-level tokenizer
├── train.py                  # Training script
├── generate.py               # Text generation script
├── README.md                 # Project documentation

