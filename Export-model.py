import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# -----------------------------
# 1️⃣ Read text data
# -----------------------------
with open("data.txt", "r") as f:
    text = f.read().lower().strip()

tokens = text.split()
words = sorted(list(set(tokens)))
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(words)

# -----------------------------
# 2️⃣ Prepare training data
# -----------------------------
seq_len = 3  # 2 words -> next word
data = []
for i in range(len(tokens) - seq_len):
    x = torch.tensor([[word2idx[tokens[i]], word2idx[tokens[i+1]]]])
    y = torch.tensor([word2idx[tokens[i+2]]])
    data.append((x, y))

# -----------------------------
# 3️⃣ Define Tiny RNN model
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

# -----------------------------
# 4️⃣ Train the model
# -----------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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
# 5️⃣ Save model and vocab
# -----------------------------
torch.save(model.state_dict(), "tiny_rnn_model.pth")

with open("word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)

with open("idx2word.pkl", "wb") as f:
    pickle.dump(idx2word, f)

print("\n✅ Model and vocabulary saved successfully!")
