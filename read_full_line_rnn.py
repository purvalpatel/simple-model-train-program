import torch
import torch.nn as nn
import torch.optim as optim
import random

# -----------------------------
# 1️⃣ Read text data from file
# -----------------------------
with open("data.txt", "r") as f:
    text = f.read().lower().strip()

# Split text into words
tokens = text.split()
print(f"Loaded {len(tokens)} words.")

# -----------------------------
# 2️⃣ Build vocabulary
# -----------------------------
words = sorted(list(set(tokens)))
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)
print(f"Vocabulary size: {vocab_size}")

# -----------------------------
# 3️⃣ Prepare training data (3-gram style)
# -----------------------------
seq_len = 3  # input size (2 words → next word)
data = []
for i in range(len(tokens) - seq_len):
    x = torch.tensor([[word2idx[tokens[i]], word2idx[tokens[i+1]]]])
    y = torch.tensor([word2idx[tokens[i+2]]])
    data.append((x, y))

print(f"Prepared {len(data)} training samples.")

# -----------------------------
# 4️⃣ Define Tiny RNN model
# -----------------------------
class TinyRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_size=32):
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
# 6️⃣ Text generation function
# -----------------------------
def generate_text(start_words, num_words=10):
    words_input = start_words.lower().split()
    if len(words_input) < 2:
        print("Please enter at least two words to start.")
        return ""

    for _ in range(num_words):
        w1, w2 = words_input[-2], words_input[-1]
        if w1 not in word2idx or w2 not in word2idx:
            break
        x = torch.tensor([[word2idx[w1], word2idx[w2]]])
        with torch.no_grad():
            out = model(x)
            predicted_idx = torch.argmax(out).item()
            next_word = idx2word[predicted_idx]
            words_input.append(next_word)
    return " ".join(words_input)

# -----------------------------
# 7️⃣ Interactive chat loop
# -----------------------------
print("\n✨ TinyRNN Text Generator ✨")
print("Type two words (like 'i like') and I’ll complete the sentence.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").lower().strip()
    if user_input == "quit":
        break

    generated = generate_text(user_input, num_words=10)
    print("Model:", generated)
