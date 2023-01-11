from collections import deque
import numpy as np
from omnisafe.typing import List

class RecordQueue():

    def __init__(self, *names, maxlen=100) -> None:
        """Initialize the RecordQueue."""
        self.queues = {}
        self._create_deques(*names, maxlen=maxlen)
    
    def _create_deques(self, *names, maxlen=100) -> None:
        """Create deques by names."""
        for name in names:
            self.queues[name] = deque(maxlen=maxlen)
        
    def append(self, **kwargs) -> None:
        """Add values to the deques."""
        for key, value in kwargs.items():
            assert key in self.queues.keys(), f'{key} has not been set in queues {self.queues.keys()}'
            self.queues[key].append(value)
    
    def non_empty_mean(self, name) -> np.ndarray:
        """Get the mean of the non-empty values."""
        return np.mean(self.queues[name]) if len(self.queues[name])  else 0.0

    def get_mean(self, *names) -> List:
        """Get the means of needed deque names."""
        assert all(name in self.queues.keys() for name in names), f'{names} has not been set in queues {self.queues.keys()}'
        if len(names) == 1:
            return self.non_empty_mean(names[0])
        return [self.non_empty_mean(name) for name in names]

    def reset(self, *names) -> None:
        """Reset the needed deque."""
        assert all(name in self.queues.keys() for name in names), f'{names} has not been set in queues {self.queues.keys()}'
        for name in names:
            self.queues[name].clear()