from torch import nn


class RNNModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim=128, hidden_dim=16) -> None:
		super().__init__()

		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
		self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim)

		self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

		self.sigmoid = nn.Sigmoid()
	
	def forward(self, seq):
		word_embedding = self.embedding(seq)

		rnn_in = word_embedding.view(len(seq), 1, -1)
		rnn_out, _ = self.rnn(rnn_in)

		scores = self.linear(rnn_out[-1])

		return self.sigmoid(scores)
