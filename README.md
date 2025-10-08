we will train a simple model from scratch (base) — not a huge LLM like Llama or GPT, but something you can actually run and understand on your computer.

We’ll make a tiny text model that learns to predict the next word in a sentence — the simplest version of how LLMs work.

### Goal:

Train a small model to learn language patterns from a few example sentences, like:

```text
“I like cats.”
“I like dogs.”
“You like animals.”
```

We’ll use this to predict the next word given a few previous words.

### Tools We’ll Use:

- Python
- torch (PyTorch) — for neural networks
- numpy — for arrays
- Just a few lines of text data.

#### 1. Install requirements:
```python
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
```

##### 2. Create program: 1.py
```python
import torch
import torch.nn as nn
import torch.optim as optim

# small training data
sentences = [
    "i like cats",
    "i like dogs",
    "you like animals",
    "we love coding",
    "they love music"
]

# build vocabulary
words = set(" ".join(sentences).split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)
print("Vocabulary:", word2idx)

# prepare training data: (input_word → next_word)
data = []
for s in sentences:
    tokens = s.split()
    for i in range(len(tokens) - 1):
        x = torch.tensor([word2idx[tokens[i]]])
        y = torch.tensor([word2idx[tokens[i + 1]]])
        data.append((x, y))

# simple neural network model
class TinyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x

model = TinyLM(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
for epoch in range(200):
    total_loss = 0
    for x, y in data:
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# test prediction
def predict_next(word):
    with torch.no_grad():
        x = torch.tensor([word2idx[word]])
        out = model(x)
        predicted_idx = torch.argmax(out).item()
        return idx2word[predicted_idx]

print("\nNext word after 'i' might be:", predict_next("i"))
print("Next word after 'love' might be:", predict_next("love"))

```
#### 3. run it:

python3 1.py

#### what just happened?
- “i → like”, “love → coding”, “love → music”.
- This is a tiny version of how LLMs learn language patterns.
- Real LLMs do the same — but with billions of parameters and terabytes of text.

## Now try to do this with interactive data from user.
1. Create program : tiny_rnn_chat.py

```python
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1️⃣ Training sentences
# -----------------------------
sentences = [
    "i like cats",
    "i like dogs",
    "you like animals",
    "we love coding",
    "they love music"
]

# -----------------------------
# 2️⃣ Build vocabulary
# -----------------------------
words = set(" ".join(sentences).split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)
print("Vocabulary:", word2idx)

# -----------------------------
# 3️⃣ Prepare training data (two words -> next word)
# -----------------------------
data = []
for s in sentences:
    tokens = s.split()
    for i in range(len(tokens) - 2):
        x = torch.tensor([[word2idx[tokens[i]], word2idx[tokens[i+1]]]])
        y = torch.tensor([word2idx[tokens[i+2]]])
        data.append((x, y))

# -----------------------------
# 4️⃣ Define Tiny RNN model
# -----------------------------
class TinyRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # take the last output
        return out

model = TinyRNN(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 5️⃣ Training loop
# -----------------------------
for epoch in range(300):
    total_loss = 0
    for x, y in data:
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# 6️⃣ Prediction function
# -----------------------------
def predict_next_word(word1, word2):
    x = torch.tensor([[word2idx[word1], word2idx[word2]]])
    with torch.no_grad():
        out = model(x)
        predicted_idx = torch.argmax(out).item()
        return idx2word[predicted_idx]

# -----------------------------
# 7️⃣ Interactive loop
# -----------------------------
print("\nType two words (like 'i like') and I will predict the next word.")
print("Type 'quit' to exit.")

while True:
    user_input = input("You: ").lower().strip()
    if user_input == "quit":
        break

    words_input = user_input.split()
    if len(words_input) != 2:
        print("Please type exactly two words.")
        continue

    next_word = predict_next_word(words_input[0], words_input[1])
    print("Model predicts:", next_word)
```
## Take data from file and provide interactive output:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1️⃣ Read text data from file
# -----------------------------
with open("data.txt", "r") as f:
    sentences = [line.strip().lower() for line in f.readlines() if line.strip()]

print("Sentences loaded:", sentences)

# -----------------------------
# 2️⃣ Build vocabulary
# -----------------------------
words = set(" ".join(sentences).split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)
print("Vocabulary:", word2idx)

# -----------------------------
# 3️⃣ Prepare training data (two words -> next word)
# -----------------------------
data = []
for s in sentences:
    tokens = s.split()
    if len(tokens) < 3:
        continue  # need at least 3 words for two-word input
    for i in range(len(tokens) - 2):
        x = torch.tensor([[word2idx[tokens[i]], word2idx[tokens[i+1]]]])
        y = torch.tensor([word2idx[tokens[i+2]]])
        data.append((x, y))

# -----------------------------
# 4️⃣ Define Tiny RNN model
# -----------------------------
class TinyRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = TinyRNN(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 5️⃣ Training loop
# -----------------------------
for epoch in range(300):
    total_loss = 0
    for x, y in data:
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# 6️⃣ Prediction function
# -----------------------------
def predict_next_word(word1, word2):
    if word1 not in word2idx or word2 not in word2idx:
        return "<unknown>"
    x = torch.tensor([[word2idx[word1], word2idx[word2]]])
    with torch.no_grad():
        out = model(x)
        predicted_idx = torch.argmax(out).item()
        return idx2word[predicted_idx]

# -----------------------------
# 7️⃣ Interactive loop
# -----------------------------
print("\nType two words (like 'i like') and I will predict the next word.")
print("Type 'quit' to exit.")

while True:
    user_input = input("You: ").lower().strip()
    if user_input == "quit":
        break

    words_input = user_input.split()
    if len(words_input) != 2:
        print("Please type exactly two words.")
        continue

    next_word = predict_next_word(words_input[0], words_input[1])
    print("Model predicts:", next_word)
```
