
class PointDetector:

    def __init__(self):
        self.prev_bounce_in_bounds = False
        self.side_of_court = 0 # <0 means bottom, >0 means top
        self.buffer = -1 # prevents a single bounce being detected as multiple
        self.points = 0

    def reset(self):
        self.side_of_court = 0
        #self.buffer = -1
        self.prev_bounce_in_bounds = False
        #self.points = 0