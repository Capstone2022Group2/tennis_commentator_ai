import cv2
import openai

import openai
import os
from dotenv import load_dotenv


class Commentator:

    text_location = (250,680)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_thickness = 3
    text_color = (255,255, 255)
    line_type = 3

    current_commentary = ''

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.events = {'point': 'A player has just scored a point', 
                       'serve': 'A player has just served the ball'}

    def get_commentary(self, event):
        
        # The system parameter sets up the chat agent and gives it context for the conversation
        # user parameter is is a prompt from the user to the chat agent
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful commentator for a tennis match."},
                        {"role": "user", "content": f'{self.events[event]}'},
                    ]
                )
            self.current_commentary = response['choices'][0]['message']['content']
            print(self.current_commentary)
        except Exception as e:
            print(e)
        
        
    # TODO format this text nicely on screen
    def display_commentary(self, image):
        cv2.putText(image, self.current_commentary, 
          self.text_location, 
          cv2.FONT_HERSHEY_SIMPLEX, 
          self.font_scale,
          self.text_color,
          self.text_thickness,
          self.line_type)

# commentator = Commentator()
# commentator.display_event('point')
# print('--------------------------')
# commentator.display_event('serve')
