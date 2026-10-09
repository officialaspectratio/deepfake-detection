# uncertainty.py
# Monte Carlo Dropout Uncertainty Estimation Module
# For: Deepfake Detection Project — Ephraim, FUTA
# Purpose: Calculates how confident the model actually is by
#          running the image multiple times with dropout active.

import torch
import numpy as np


def enable_dropout(model):
    """
    Turns on the Dropout layers inside the model.
    
    Normally, when we call model.eval(), PyTorch automatically turns
    off Dropout so the model makes a single, stable prediction.
    For Monte Carlo Dropout, we WANT the randomness. This function
    finds all Dropout layers and puts them back into train mode.
    """
    for module in model.modules():
        # Check if the current layer is a Dropout layer
        if isinstance(module, torch.nn.Dropout):
            # .train() turns the dropout layer back on
            module.train()


def mc_dropout_predict(model, image_tensor, n_passes=10, class_idx=None):
    """
    Runs the image through the model multiple times to measure uncertainty.
    
    Args:
        model:        The trained PyTorch model
        image_tensor: The preprocessed image tensor (1, 3, 224, 224)
        n_passes:     How many times to run the image through the model.
                      10 passes balances speed and statistical stability for web apps.
        class_idx:    Class index to evaluate (0 = REAL, 1 = FAKE). Defaults to 1 (FAKE).
                      
    Returns:
        mean_confidence: The average probability of the target class
        uncertainty:     The standard deviation (how much the predictions varied)
        probs_list:      A list of all raw predictions
    """
    # Put the whole model in eval mode first (good practice)
    model.eval()
    
    # Turn dropout back on just for this prediction
    enable_dropout(model)

    probs_list = []
    target_idx = 1 if class_idx is None else class_idx
    
    # torch.no_grad() tells PyTorch not to save memory gradients, 
    # which speeds up the process because we are only predicting, not training.
    with torch.no_grad():
        for _ in range(n_passes):
            # Pass the image through the model
            output = model(image_tensor)
            
            # Apply softmax to turn the raw output into a percentage (0.0 to 1.0)
            prob = torch.softmax(output, dim=1)[0][target_idx].item()
            
            # Save this prediction to our list
            probs_list.append(prob)

    # Calculate the average (mean) of all predictions
    mean_confidence = float(np.mean(probs_list))
    
    # Calculate the standard deviation (how spread out the numbers are)
    uncertainty = float(np.std(probs_list))

    return mean_confidence, uncertainty, probs_list


def get_uncertainty_tier(uncertainty):
    """
    Maps the uncertainty number to a colour-coded tier for the web interface.
    
    Args:
        uncertainty: The standard deviation from mc_dropout_predict()
        
    Returns:
        tier_name:    Text like "High Confidence"
        colour:       "green", "amber", or "red" (used in HTML/CSS later)
        message:      A plain-English explanation for the user
    """
    # If the standard deviation is very low (under 0.05), the model 
    # gave almost the exact same answer 30 times. It is very sure.
    if uncertainty < 0.05:
        return "High Confidence", "green", \
               "The model is highly certain of this result."
               
    # If the standard deviation is moderate (between 0.05 and 0.15),
    # the model wavered a bit. It is probably right, but worth checking.
    elif uncertainty < 0.15:
        return "Moderate Confidence", "amber", \
               "Moderate certainty. Consider cross-checking this image."
               
    # If the standard deviation is high (0.15 or above), the model gave
    # wildly different answers. It is essentially guessing.
    else:
        return "Low Confidence", "red", \
               "The model is uncertain. Manual inspection is recommended."