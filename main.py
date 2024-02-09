from config import Config
from transformers import HfArgumentParser
from dataset import DataProcessor, get_loaders
from train import train_model
from test import eval_model_on_metrics, eval_model_on_metrics_with_action_optimize
from utils import set_seeds

# Revisting direct reward maximization in Portfolio Management 
# Multi-Market

def run(args):

    ''' Load Datasets '''
    processor = DataProcessor(args)
    processor.set_data_args()

    ''' Get Loaders '''
    loader_dict = get_loaders(args, processor.tech_dict, processor.close_dict, processor.date_dict, processor.cov_dict)

    ''' Model Training '''
    best_model, save_path = train_model(args, loader_dict)

    ''' Model Testing '''
    eval_model_on_metrics(args, loader_dict, best_model)
    # eval_model_on_metrics_with_action_optimize(args, loader_dict, best_model)

if __name__ == "__main__":

    parser = HfArgumentParser(Config)
    args = parser.parse_args_into_dataclasses()[0]
    set_seeds(args.seed)
    run(args)