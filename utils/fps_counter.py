# utils/fps_counter.py
# Tracks and displays real-time FPS accurately

import time
import collections

class FPSCounter:
    def __init__(self, buffer_size=30):
        """
        Args:
            buffer_size : number of recent frames to average FPS over
        """
        self.buffer_size = buffer_size
        self.timestamps = collections.deque(maxlen=buffer_size)

    def update(self):
        """Call once per frame to record timestamp."""
        self.timestamps.append(time.time())

    def get_fps(self) -> float:
        """Returns average FPS over last buffer_size frames."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def get_fps_string(self) -> str:
        """Returns formatted FPS string for display."""
        return f"FPS: {self.get_fps():.1f}"

    def reset(self):
        """Reset FPS counter."""
        self.timestamps.clear()