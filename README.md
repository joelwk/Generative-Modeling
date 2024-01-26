# Generative Modeling
Experiment and showcase custom or pre-built generative models with ClearML experiment management.
## Generative Architecture
### 1. **General text generation (`/generative_text/general_tnn_generative`)**

#### `tnn.py` - Transformer-based neural network, specifically designed for generative text tasks in TensorFlow. 
1. **TransformerBlock**: A custom layer that implements the Transformer mechanism. It uses multi-head attention (`MultiHeadAttention`), followed by layer normalization and a feed-forward network. This block handles the key aspect of the Transformer architecture, focusing on the relationships between different parts of the input.

2. **TokenAndPositionEmbedding**: Another custom layer that creates embeddings for the tokens and their positions within the sequence. This is crucial for the Transformer model as it doesn't inherently understand the sequence order.

Key parameters `max_len`, `vocab_size`, `embedding_dim` are configurable, allowing for flexibility in adapting the model to various text generation tasks. These are modified in `configkeras.ini`

#### `train.py` - Highly configurable training script with parameters such as maximum sequence length, vocabulary size, and dimensions set through `configkeras.ini`

1. **Custom Learning Rate Schedule**: Implements a custom learning rate schedule (CustomSchedule) for the Adam optimizer, which adjusts the learning rate based on the number of training steps and warmup steps.

2. **Model Construction**: Constructs a Transformer model using TokenAndPositionEmbedding and multiple TransformerBlock layers. The model predicts the next word in a sequence, making it suitable for generative text tasks.

3. **Loading Pre-Existing Models**: Includes functionality to preload a model from a specified path, allowing for model retraining or transfer learning.

4. **TrainTextGenerator Callback**: A custom callback class for generating text at the end of each training epoch. It uses a temperature-based sampling method to generate text, which can be used for monitoring the model's performance during training.

Important features for analysis and debugging include **TensorBoard logging**, returned attention scores, **temperature-based sampling** which allows for controlling the randomness of the generated text.


## Dataset Curation Overview
### 1. **Context Pairing (`/generative_text/general_tnn_generative/utils/fnContextPairing.py`)**
The `ContextPairing.py` is designed for processing and correlating text data with relevant Wikipedia articles. Its primary function is to analyze a given dataset, extract specific entities, and match these entities with the summaries of corresponding Wikipedia articles. Each matched entity in the data is linked to the summary of a relevant Wikipedia article, thereby enriching the original dataset with contextual information.