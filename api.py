from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
from JointBERT_Implementation import JointBERT
import os

app = FastAPI()

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize JointBERT model
# Note: Adjust num_intent_labels and num_slot_labels based on your ATIS dataset
model = JointBERT(bert_model, num_intent_labels=26, num_slot_labels=129).to(device)
model.load_state_dict(torch.load('model/quantized_joint_bert_model.pt', map_location=device))
model.eval()

# Input model for FastAPI
class TextInput(BaseModel):
    text: str

# Helper function to process input
def process_input(text, max_seq_length=128):
    encoding = tokenizer(
        text,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

# Helper function to map indices to labels
# Note: These should be loaded from your dataset or saved during training
intent_label_map = {i: label for i, label in enumerate(['flight_booking', 'flight_status', 'other'])}  # Example
slot_label_map = {i: label for i, label in enumerate(['O', 'B-destination', 'I-destination', 'B-departure'])}  # Example

@app.post("/predict")
async def predict(input: TextInput):
    try:
        input_ids, attention_mask = process_input(input.text)
        with torch.no_grad():
            intent_logits, slot_logits = model(input_ids, attention_mask)
        
        # Get intent prediction
        intent_pred = torch.argmax(intent_logits, dim=1).item()
        intent_label = intent_label_map.get(intent_pred, 'unknown')

        # Get slot predictions
        slot_preds = torch.argmax(slot_logits, dim=2).squeeze().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
        slot_labels = [slot_label_map.get(pred, 'O') for pred in slot_preds[:len(tokens)]]

        return {
            "intent": intent_label,
            "slots": [{"token": token, "slot": slot} for token, slot in zip(tokens, slot_labels) if token not in ['[PAD]', '[CLS]', '[SEP]']]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)