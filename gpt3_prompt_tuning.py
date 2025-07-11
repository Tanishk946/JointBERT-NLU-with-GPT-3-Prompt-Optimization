import openai
import json
from dotenv import load_dotenv
import os
import argparse

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load ATIS dataset for prompt tuning
def load_atis_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

# Define prompt templates
prompt_templates = [
    "Classify the intent of this query: '{query}' Options: flight_booking, flight_status, other.",
    "Given the query '{query}', identify the intent from: flight_booking, flight_status, other.",
    "Query: '{query}' What is the intent? Choose from: flight_booking, flight_status, other."
]

# Function to evaluate a prompt
def evaluate_prompt(prompt_template, query, true_intent):
    prompt = prompt_template.format(query=query)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for intent classification."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.0
    )
    predicted_intent = response.choices[0].message.content.strip().lower()
    return predicted_intent == true_intent.lower()

# Main function for prompt tuning
def main(args):
    data = load_atis_data(args.data_path)
    results = {i: {'correct': 0, 'total': 0} for i in range(len(prompt_templates))}

    # Evaluate each prompt template
    for item in data[:args.sample_size]:  # Use a subset for testing
        query = item['text']
        true_intent = item['intent']
        
        for i, template in enumerate(prompt_templates):
            if evaluate_prompt(template, query, true_intent):
                results[i]['correct'] += 1
            results[i]['total'] += 1

    # Print results
    for i, result in results.items():
        accuracy = result['correct'] / result['total'] if result['total'] > 0 else 0
        print(f"Prompt {i+1} Accuracy: {accuracy:.4f} ({result['correct']}/{result['total']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='atis_data.json')
    parser.add_argument('--sample_size', type=int, default=100)
    args = parser.parse_args()
    main(args)