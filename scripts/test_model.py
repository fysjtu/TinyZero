#!/usr/bin/env python3
"""
Test script for evaluating trained model on Countdown dataset
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils.reward_score.countdown import compute_score, evaluate_equation


def load_test_data(data_path):
    """Load test data from parquet file"""
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} test samples")
    return df


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer, device


def extract_prompt_content(prompt_item):
    """Extract text content from prompt item"""
    if isinstance(prompt_item, list):
        # Handle array of dicts
        content = prompt_item[0]['content'] if len(prompt_item) > 0 else ""
    elif isinstance(prompt_item, dict):
        # Handle single dict
        content = prompt_item['content']
    else:
        # Handle string
        content = str(prompt_item)
    return content


def generate_response(model, tokenizer, prompt_text, device, max_new_tokens=512):
    """Generate response using the model"""
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
    # Decode the response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_model_performance(model_path, data_path, num_samples=100):
    """Evaluate model performance on test set"""
    # Load data and model
    df = load_test_data(data_path)
    model, tokenizer, device = load_model(model_path)
    
    # Sample test data if needed
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
        print(f"Sampling {num_samples} test samples for evaluation")
    
    # Initialize metrics
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    print("Starting model evaluation...")
    
    for idx, row in df.iterrows():
        try:
            # Extract prompt
            prompt_content = extract_prompt_content(row['prompt'])
            
            # Generate response
            response = generate_response(model, tokenizer, prompt_content, device)
            
            # Extract ground truth
            ground_truth = row['reward_model']['ground_truth']
            target = ground_truth['target']
            numbers = ground_truth['numbers']
            
            # Compute score
            score = compute_score(response, ground_truth)
            
            # Check if prediction is correct
            if score == 1.0:
                correct_predictions += 1
            
            total_predictions += 1
            
            # Store results
            results.append({
                'idx': idx,
                'prompt': prompt_content[:100] + "..." if len(prompt_content) > 100 else prompt_content,
                'response': response,
                'target': target,
                'numbers': numbers,
                'score': score
            })
            
            # Print progress
            if (idx + 1) % 10 == 0:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Processed {idx + 1}/{len(df)} samples, Current Accuracy: {accuracy:.2%}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate final metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples evaluated: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*50)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = f"model_evaluation_results_step_{os.path.basename(model_path)}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")
    
    return accuracy, results_df


def main():
    # Configuration
    model_checkpoint = "/root/autodl-tmp/checkpoints/actor/global_step_100"  # Latest checkpoint
    test_data_path = "/root/TinyZero/data/test.parquet"
    num_test_samples = 100  # Adjust based on your needs
    
    # Check if paths exist
    if not os.path.exists(model_checkpoint):
        print(f"Model checkpoint not found at {model_checkpoint}")
        sys.exit(1)
        
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        sys.exit(1)
    
    # Run evaluation
    print("Starting model evaluation...")
    accuracy, results = evaluate_model_performance(
        model_path=model_checkpoint,
        data_path=test_data_path,
        num_samples=num_test_samples
    )
    
    print(f"\nEvaluation completed. Final accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()