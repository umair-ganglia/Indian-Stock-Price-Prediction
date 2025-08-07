"""
Data preprocessing utilities for stock prediction
"""

import numpy as np
import pandas as pd

def preprocess_features(data):
    """
    Preprocess features to handle infinite and NaN values
    
    Args:
        data: numpy array or pandas DataFrame to preprocess
    Returns:
        Preprocessed data with handled infinities and NaNs
    """
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to numpy array
        data_np = data.values
    else:
        data_np = data
        
    # Replace inf and -inf with maximum and minimum finite values
    data_np = np.nan_to_num(
        data_np,
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).min
    )
    
    return data_np
