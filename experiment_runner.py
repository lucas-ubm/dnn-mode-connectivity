import os
import json
import jsonschema
import torch
from train_hf import run as train_run
from argparse import Namespace

def load_and_validate_experiment(experiment_path):
    """Load and validate experiment configuration"""
    with open(experiment_path, 'r') as f:
        experiment = json.load(f)
    
    # Load schema
    schema_path = os.path.join(os.path.dirname(__file__), 'experiment_schema.json')
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Validate experiment against schema
    jsonschema.validate(instance=experiment, schema=schema)
    return experiment

def setup_early_stopping(config):
    """Configure early stopping from experiment config"""
    early_stopping = config['developer']['stopping_criteria']['early_stopping']
    return {
        'metric': early_stopping['metric'],
        'threshold': early_stopping['threshold']
    }

def create_args_from_experiment(experiment):
    """Create argument namespace from experiment configuration"""
    args = Namespace()
    
    # Convert experiment config to training arguments
    dev_config = experiment['developer']
    
    # Set loss function
    args.loss_config = dev_config['loss_fn'] if isinstance(dev_config['loss_fn'], str) else None
    args.loss_spec = None if isinstance(dev_config['loss_fn'], str) else dev_config['loss_fn']
    
    # Set metrics
    args.metrics = dev_config['metrics']
    
    # Set training parameters
    args.max_epochs = dev_config['stopping_criteria']['max_epochs']
    args.early_stopping = setup_early_stopping(experiment)
    
    # Set test split parameters
    test_split = dev_config['test_split']
    args.random_seed = test_split['random_seed']
    args.test_size = test_split['test_size']
    
    # Set evaluation requirements
    eval_config = experiment['evaluator']
    args.accuracy_requirements = eval_config['accuracy_requirement']
    
    # Set default values for other parameters
    args.dir = '/app/curve/'
    args.experiment_name = experiment.get('experiment_name', 'default_experiment')
    args.dataset = 'CIFAR10'
    args.model = 'VGG19'
    args.batch_size = 128
    args.num_workers = 8
    args.use_test = True
    args.data_path = '/app/data'
    return args

def run_experiment(experiment_path):
    """Run an experiment defined in a JSON file"""
    # Load and validate experiment configuration
    experiment = load_and_validate_experiment(experiment_path)
    
    # Create argument namespace
    args = create_args_from_experiment(experiment)
    args.experiment_path = experiment_path
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    
    # Run training
    results = train_run(args)
        
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run experiment from config file')
    parser.add_argument('experiment_path', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()
    
    run_experiment(args.experiment_path)
