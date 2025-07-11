import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse

# Define the JointBERT model
class JointBERT(nn.Module):
    def __init__(self, bert_model, num_intent_labels, num_slot_labels, dropout_rate=0.1):
        super(JointBERT, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.intent_classifier = nn.Linear(bert_model.config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(bert_model.config.hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        pooled_output = outputs[1]   # (batch_size, hidden_size)

        # Intent classification
        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        # Slot filling
        slot_logits = self.slot_classifier(self.dropout(sequence_output))

        return intent_logits, slot_logits

# Dataset class for ATIS
class ATISDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self.load_data(data_path)
        self.intent_labels = sorted(list(set([item['intent'] for item in self.data])))
        self.slot_labels = sorted(list(set([slot for item in self.data for slot in item['slots']])))
        self.intent_label_map = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.slot_label_map = {label: idx for idx, label in enumerate(self.slot_labels)}

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        intent = item['intent']
        slots = item['slots']

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert labels
        intent_label = self.intent_label_map[intent]
        slot_labels = [self.slot_label_map.get(slot, self.slot_label_map['O']) for slot in slots]
        slot_labels = slot_labels + [self.slot_label_map['O']] * (self.max_seq_length - len(slot_labels))

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'intent_label': torch.tensor(intent_label),
            'slot_labels': torch.tensor(slot_labels)
        }

# Training function
def train_model(model, train_loader, optimizer, device, slot_loss_coef=1.0):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels = batch['intent_label'].to(device)
        slot_labels = batch['slot_labels'].to(device)

        optimizer.zero_grad()
        intent_logits, slot_logits = model(input_ids, attention_mask)

        intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)
        slot_loss = nn.CrossEntropyLoss()(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
        loss = intent_loss + slot_loss_coef * slot_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Main function
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Load dataset
    dataset = ATISDataset(args.data_path, tokenizer, args.max_seq_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = JointBERT(bert_model, len(dataset.intent_labels), len(dataset.slot_labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        loss = train_model(model, train_loader, optimizer, device, args.slot_loss_coef)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Save model
    torch.save(model.state_dict(), args.model_dir + '/joint_bert_model.pt')

    # Quantization
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), args.model_dir + '/quantized_joint_bert_model.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='atis_data.json')
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)