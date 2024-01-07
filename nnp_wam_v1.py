import numpy as np
import torch
import torch.nn.functional as F

# Introduction to the script
input("Welcome to the interactive tutorial on Neural Network Processing with Attention Mechanism. Press Enter to start...")

print("\nStep 1: Checking PyTorch Version")
print("PyTorch version:", torch.__version__)
input("\nPress Enter to proceed to the next step...")

# Embeddings: Introduction
print("\nStep 2: Understanding Embeddings")
print("Embeddings are vector representations of words. They convert text data into numerical form for processing by neural networks.")
input("Press Enter to see an example of word embeddings...")

# Example embeddings
embeddings = {
    "The": np.array([1, 0, 0]),
    "cat": np.array([0, 1, 0]),
    "sat": np.array([0, 0, 1]),
    "on": np.array([1, 1, 0]),
    "mat": np.array([1, 0, 1])
}
print("\nEmbeddings dictionary:\n", embeddings)
input("Press Enter to proceed to the next step...")

# Converting sentence to embeddings
print("\nStep 3: Converting a Sentence to Embeddings")
sentence = ["The", "cat", "sat", "on", "The", "mat"]
print("Example sentence:", ' '.join(sentence))
sentence_embeddings = np.array([embeddings[word] for word in sentence])
print("Sentence embeddings:\n", sentence_embeddings)
input("Press Enter to proceed to the next step...")

# Tensor conversion
print("\nStep 4: Converting Embeddings to Tensor")
print("Tensors are multi-dimensional arrays used in PyTorch for efficient computation.")
sentence_tensor = torch.tensor(sentence_embeddings, dtype=torch.float32)
print("Sentence tensor:\n", sentence_tensor)
input("Press Enter to proceed to the next step...")

# Attention mechanism
print("\nStep 5: Applying the Attention Mechanism")
def attention(query, key, value):
    print("\nCalculating attention scores (dot product of query and key)...")
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key.size(-1))
    print("Scores:\n", scores)

    print("\nApplying softmax to scores to get probabilities...")
    probabilities = F.softmax(scores, dim=-1)
    print("Probabilities:\n", probabilities)

    print("\nCalculating final output as weighted sum of values...")
    return torch.matmul(probabilities, value)

# Applying attention function
query, key, value = sentence_tensor, sentence_tensor, sentence_tensor
attention_output = attention(query, key, value)
print("\nAttention output:\n", attention_output.numpy())

input("\nTutorial completed. Press Enter to exit...")
