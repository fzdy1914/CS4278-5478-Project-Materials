import numpy as np
from scipy.signal import find_peaks


def find_highest_peak(scores):
    peaks, _ = find_peaks(scores, distance=5)
    highest_peak_index = peaks[np.argmax(np.array(scores)[peaks])]
    start = max(0, highest_peak_index - 1)
    end = min(len(scores), highest_peak_index + 2)
    peak_positions = np.arange(start, end)
    peak_scores = np.array(scores[start:end])
    mean = np.average(peak_positions, weights=peak_scores)
    return int(mean)
