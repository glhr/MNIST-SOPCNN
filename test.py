# test.py

import os
import torch
import torch.nn.functional as F
import pandas as pd
import torchvision
import numpy as np

def test(model, test_loader, config, split_name='test'):
    model.eval()
    test_loss = 0
    correct = 0
    test_log = []
    device = config.device
    
    model.to(device)

    results_path = os.path.join(config.results_path, split_name)
    # create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    with torch.no_grad():
        # paths for incorrect images
        img_paths_gather = []
        incorrect_preds_gather = []
        incorrect_targets_gather = []
        for batch_n, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum').item()
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_log.append((loss, correct))
            
            # get incorrect data points and their predictions
            incorrect_indices = (pred != target.view_as(pred)).nonzero(as_tuple=True)[0]
            if len(incorrect_indices) > 0:
                incorrect_data = data[incorrect_indices]
                incorrect_preds = pred[incorrect_indices]
                incorrect_targets = target[incorrect_indices]
                
                
                for i, (img, pred, target) in enumerate(zip(incorrect_data, incorrect_preds, incorrect_targets)):
                    img_path = os.path.join(results_path, f'incorrect_{batch_n}-{i}_pred{pred.item()}_target{target.item()}.png')
                    if i < 5:
                        torchvision.utils.save_image(img.cpu(), img_path)
                    img_paths_gather.append(img_path)
                    
                if len(incorrect_preds) > 0:
                    incorrect_preds_gather.extend(incorrect_preds.cpu().numpy().tolist())
                    incorrect_targets_gather.extend(incorrect_targets.cpu().numpy().tolist())
                
        # Save incorrect predictions to a CSV file
        incorrect_df = pd.DataFrame({
            #'data': [d.cpu().numpy() for d in incorrect_data],
            'image_path': img_paths_gather,
            'predicted': incorrect_preds_gather,
            'target': incorrect_targets_gather
        })
        incorrect_df.to_csv(os.path.join(results_path, 'incorrect_predictions.csv'), index=False)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'{split_name} Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.5f}%')
    
    # Save test log
    test_log_df = pd.DataFrame(test_log, columns=['loss', 'correct'])
    test_log_df.to_csv(os.path.join(results_path, 'test_log.csv'), index=False)

    return test_loss, test_accuracy
