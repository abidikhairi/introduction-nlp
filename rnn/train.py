import torch
import pandas as pd
import torchmetrics.functional as metrics

from models import RNNModel
from utils import prepare_sequence, load_or_build_vocab


if __name__ == '__main__':
    batch_size = 2048
    num_epochs = 10
    learning_rate = 0.01
    embedding_dim = 128
    hidden_dim = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    word2idx = load_or_build_vocab([train_data, test_data], '../data/vocab.json')
    
    model = RNNModel(vocab_size=len(word2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    epoch_loss = []
    epoch_accuracy = []

    num_batches = len(train_data)

    for epoch in range(num_epochs):
        model.train()

        batch_loss = []
        batch_accuracy = []
		
        preds = []
        labels = []
		
        for batch_idx, (idx, row) in enumerate(train_data.iterrows()):
            seq = prepare_sequence(row['text'], word2idx).to(device)
            label = torch.tensor(row['target']).float().unsqueeze(-1).to(device)
            
            score = model(seq).view(1)

            loss = loss_fn(score, label)

            loss.backward()

            preds.append(score.detach().cpu().tolist())
            labels.append(label.detach().cpu().item())

            if batch_idx > 0 and batch_idx % batch_size == 0:
                b_loss = loss_fn(torch.tensor(preds).squeeze(1), torch.tensor(labels)).item()
                b_accuracy = metrics.accuracy(torch.tensor(preds).squeeze(1), torch.tensor(labels).long()).item()
				
                template_message = f'training:\tepoch: [{epoch + 1}/{num_epochs}]\tprocessed: [{batch_idx}/{num_batches}]\ttrain loss: {b_loss:.4f}\ttrain accuracy: {b_accuracy*100:.2f}%'
                print(template_message, flush=True)
				
                batch_loss.append(b_loss)
                batch_accuracy.append(b_accuracy)

                optimizer.step()
                optimizer.zero_grad()
        
        # Evaluation
        with torch.no_grad():
            model.eval()
                    
            test_preds = []
            test_labels = []

            test_losses = []
            test_accuracies = []
            
            for idx, row in test_data.iterrows():
                seq = prepare_sequence(row['text'], word2idx).to(device)
                label = torch.tensor(row['target']).float().unsqueeze(-1).to(device)
                    
                scores = model(seq).view(1)
                        
                test_preds.append(scores.tolist())
                test_labels.append(label.item())
                    
                test_loss = loss_fn(torch.tensor(test_preds).squeeze(1), torch.tensor(test_labels)).item()
                test_accuracy = metrics.accuracy(torch.tensor(test_preds).squeeze(1), torch.tensor(test_labels).long()).item()
                
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

            
            template_message = f'testing:\tepoch: {epoch + 1}\ttest loss: {torch.tensor(test_losses).mean():.4f}\ttest accuracy: {torch.tensor(test_accuracies).mean()*100:.2f}%'
            print(template_message, flush=True)

            if torch.tensor(test_accuracies).mean() > 0.6:
                torch.save(model.state_dict(), f'../data/rnn-{torch.tensor(test_accuracies).mean()}.pt')
