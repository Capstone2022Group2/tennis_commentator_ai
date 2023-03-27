
class ReplayDetector:

    def __init__(self):
       # the number of points expected to be scored before a replay happens
       self.expected_points = 1
       self.replay_frames = 0
    
    def check_game_status(self, data, model):
        results = model.predict(data.reshape(1, -1))

        return results

    def confirm_if_replay(self, current_points):

        if current_points < self.expected_points:
            return False

        # a replay has to be detected for 60 frames in a row to prevent false positives
        # sometimes, the model detects a replay for a frame or 2
        if self.replay_frames >= 30:
            return True
        else:
            return False
    