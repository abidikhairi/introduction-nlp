from torch import nn


label_names = {
	0: "negative",
	1: "positive"
}

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

	def inference(self, seq):
		score = self(seq)
		label = (score > 0.5).long()[0].squeeze().item()
		label_name = label_names[label]
		confidence = score[0].squeeze().item()
		
		return label_name, confidence
