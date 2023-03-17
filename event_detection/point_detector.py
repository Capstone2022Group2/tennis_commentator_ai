
class PointDetector:

    def __init__(self):
        self.prev_bounce_in_bounds = False
        self.side_of_court = 0 # <0 means bottom, >0 means top
        self.side_of_last_bounce = 0
        self.buffer = -1 # prevents a single bounce being detected as multiple
        self.points = 0
        self.frames_out_of_bounds = 0
        self.frames_no_ball_detected = 0
        self.point_was_scored = False

    def reset(self):
        self.side_of_court = 0
        self.side_of_last_bounce = 0
        #self.buffer = -1
        self.prev_bounce_in_bounds = False
        #self.points = 0