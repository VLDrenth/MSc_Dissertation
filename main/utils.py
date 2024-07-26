import pickle
import time
import json
import os

def generate_experiment_id():
    return time.strftime("%Y%m%d-%H%M%S")

def save_experiment(config, results, experiment_id, base_dir='experiments'):
    # Create a directory for this experiment
    exp_dir = os.path.join(base_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Save results
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Experiment saved in {exp_dir}")
    return exp_dir

def log_experiment(experiment_id, config, results, log_file='experiment_log.jsonl'):
    summary = {
        'id': experiment_id,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'config': {k: str(v) for k, v in config.__dict__.items()},  # Convert all values to strings
        'final_accuracy': results['test_accuracies'][-1] if results['test_accuracies'] else None,
        'final_loss': results['test_losses'][-1] if results['test_losses'] else None,
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(summary) + '\n')

