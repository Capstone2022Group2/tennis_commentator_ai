
class PointDetector:

    def __init__(self):
        self.prev_bounce_in_bounds = False
        self.side_of_court = 0 # <0 means bottom, >0 means top
        self.prev_side_of_court = 0
        self.side_of_last_bounce = 0

        self.buffer = -1 # prevents a single bounce being detected as multiple
        self.points = 0
        self.frames_out_of_bounds = 0
        self.point_was_scored = False

    def reset(self):
        self.side_of_court = 0
        self.side_of_last_bounce = 0
        #self.buffer = -1
        self.prev_bounce_in_bounds = False
        #self.points = 0
    
    def bounce_detected(self, ball_in_bounds):
        self.prev_bounce_in_bounds = ball_in_bounds
        self.side_of_last_bounce = self.side_of_court
        self.buffer = 0
    
    def update(self):
        self.buffer = self.buffer + 1 if self.buffer < 6 and self.buffer >= 0 else -1
        self.prev_side_of_court = self.side_of_court #if side_of_court != None else prev_side_of_court