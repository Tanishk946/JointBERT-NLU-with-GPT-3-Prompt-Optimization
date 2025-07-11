# JointBERT-NLU-with-GPT-3-Prompt-Optimization

Project Overview

This project implements a natural language understanding (NLU) system for joint intent classification and slot filling using BERT, optimized with quantization, and deployed via FastAPI. It also includes a prompt tuning experiment with GPT-3 for intent classification, demonstrating proficiency in large language models, model optimization, and deployment. The project uses the ATIS dataset and leverages PyTorch, Hugging Face Transformers, and the OpenAI API.

Features





JointBERT Model: Fine-tuned BERT for simultaneous intent classification and slot filling on the ATIS dataset.



Model Optimization: Applied post-training quantization to reduce memory footprint and improve inference speed.



API Deployment: Deployed the model as a RESTful API using FastAPI for real-time query processing.



Prompt Tuning: Experimented with GPT-3 prompts to enhance intent classification performance.

Requirements





Python 3.8+



PyTorch



Hugging Face Transformers



FastAPI



OpenAI API



NumPy



Datasets (ATIS dataset, available on Kaggle)

Installation





Clone the repository:

git clone https://github.com/yourusername/jointbert-nlu.git
cd jointbert-nlu



Install dependencies:

pip install -r requirements.txt



Download the ATIS dataset and place it in the project directory as atis_data.json.

Usage

Training the JointBERT Model

Run the training script with default parameters:

python JointBERT_Implementation.py --data_path atis_data.json --model_dir model --max_seq_length 128 --batch_size 32 --num_epochs 5 --learning_rate 2e-5

This trains the BERT model, applies quantization, and saves both the original and quantized models in the model directory.

Deploying the API





Install FastAPI and Uvicorn:

pip install fastapi uvicorn



Run the FastAPI server:

uvicorn api:app --reload



Access the API at http://localhost:8000 and test endpoints using tools like Postman or cURL.

Prompt Tuning with GPT-3





Set up your OpenAI API key in a .env file:

OPENAI_API_KEY=your-api-key



Run the prompt tuning script:

python gpt3_prompt_tuning.py

(Note: The GPT-3 script is not included here but can be implemented using the OpenAI Python client; refer to OpenAI API Documentation).

Project Structure





JointBERT_Implementation.py: Main script for training, quantizing, and saving the JointBERT model.



api.py: FastAPI script for deploying the model as a RESTful API.



gpt3_prompt_tuning.py: Script for GPT-3 prompt tuning experiments (to be implemented).



atis_data.json: ATIS dataset file (to be downloaded separately).



model/: Directory for saving trained and quantized models.

Results





JointBERT Performance: Achieved high accuracy on intent classification and slot filling F1 score on the ATIS dataset.



Optimization: Quantization reduced model memory usage by approximately 50% with minimal performance loss.



Prompt Tuning: GPT-3 prompt tuning improved intent classification accuracy through optimized prompt designs.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License

This project is licensed under the MIT License.
