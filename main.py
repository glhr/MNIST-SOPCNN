# main.py

import torch
from config import config
from data.dataset import *
from models.model import SOPCNN
from train import train
from test import test

import argparse

def main():
    parser = argparse.ArgumentParser(description="SOPCNN Training and Evaluation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode to run: train or test")
    args = parser.parse_args()

    # Load data
    if args.mode == "train":
        train_loader, val_loader = get_mnist_loaders(config.batch_size, config.data_path, mode=args.mode)
    else:
        train_loader, test_loader = get_mnist_loaders(config.batch_size, config.data_path, mode=args.mode)
        inverted_test_loader = get_inverted_mnist_test_loader(config.batch_size, config.data_path)
        usps_test_loader = get_usps_test_loader(config.batch_size, config.data_path)
        fashionmnist_test_loader = get_fashionmnist_test_loader(config.batch_size, config.data_path)
    
    # Initialize model, optimizer
    model = SOPCNN(num_classes=config.num_classes).to(config.device)
    
    if args.mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        train(model, train_loader, val_loader, optimizer, config)
        
    else:
        # only load "model_state_dict" key from the checkpoint
        checkpoint = torch.load("best_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test(model, train_loader, config, split_name='train')
        test(model, test_loader, config, split_name='test')
        test(model, inverted_test_loader, config, split_name='inverted_test')
        test(model, usps_test_loader, config, split_name='usps_test')
        test(model, fashionmnist_test_loader, config, split_name='fashionmnist_test')

if __name__ == '__main__':
    main()
