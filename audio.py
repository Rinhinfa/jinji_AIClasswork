import numpy as np

class SimpleAudioSimulator:
    def __init__(self):
        self.events = ['说话声', '咳嗽声', '笑声', '移动声', '耳语声']
    
    def get_random_event(self, frame_count, fps):
        if frame_count % (5 * fps) == 0 and np.random.random() < 0.1:
            return np.random.choice(self.events)
        return None