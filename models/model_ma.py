import numpy as np
from scipy.fft import fft, ifft



class mv_model:
    def __init__(self) -> None:
        pass
    def predict(self, sequence, window_size=10):
        if len(sequence) < window_size:
            return 0

        window_sum = sum(sequence[-window_size:])
        average = window_sum / window_size

        return 1 if average >= 0.5 else 0
