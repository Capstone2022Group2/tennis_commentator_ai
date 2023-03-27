
import numpy as np
def get_objects_with_highest_conf(det_objects):
    labels, coord = det_objects.xyxyn[0][:, -1].numpy(), det_objects.xyxyn[0][:, :-1].numpy()

    # ball
    i = 0
    highest_conf = 0
    i_value = -1
    ball_data = []
    for label in labels:
        # only care about checking the ball
        if label == 0:
            if coord[i][4] > highest_conf:
                highest_conf = coord[i][4]
                i_value = i
        i += 1
    if i_value > -1:
        ball_data = coord[i_value][:-1]
    else:
        ball_data = [0,0,0,0]

    # net
    i = 0
    highest_conf = 0
    i_value = -1
    net = []
    for label in labels:
        # only care about checking the ball
        if label == 2:
            if coord[i][4] > highest_conf:
                highest_conf = coord[i][4]
                i_value = i
        i += 1
    if i_value > -1:
        net = coord[i_value][:-1]
    else:
        net = [0,0,0,0]

    # players
    player_data = []
    highest_conf = 0
    second_conf = 0
    i_value = -1
    s_value = -2
    i = 0
    for label in labels:
        # only care about checking the ball
        if label == 3:
            if coord[i][4] > highest_conf:
                second_conf = highest_conf
                s_value = i_value
                highest_conf = coord[i][4]
                i_value = i
            elif coord[i][4] > second_conf:
                second_conf = coord[i][4]
                s_value = i
            
        i += 1
    
    player1 = []
    player2 = []

    if i_value < 0:
        player1 = [0,0,0,0]
    else:
        player1 = coord[i_value][:-1]
    if s_value < 0:
        player2 = [0,0,0,0]
    else:
        player2 = coord[s_value][:-1]
    
    player_data = np.concatenate([player1, player2])

    # courts
    court_data = []
    highest_conf = 0
    second_conf = 0
    i_value = -1
    s_value = -2
    i = 0
    for label in labels:
        # only care about checking the ball
        if label == 1:
            if coord[i][4] > highest_conf:
                second_conf = highest_conf
                s_value = i_value
                highest_conf = coord[i][4]
                i_value = i
            elif coord[i][4] > second_conf:
                second_conf = coord[i][4]
                s_value = i
            
        i += 1
    
    court1 = []
    court2 = []
    
    
    if i_value < 0:
        court1 = [0,0,0,0]
    else:
        court1 = coord[i_value][:-1]
    if s_value < 0:
        court2 = [0,0,0,0]
    else:
        court2 = coord[s_value][:-1]

    court_data = np.concatenate([court1, court2])

    # print(f'ball: {ball_data}')
    # print(f'player1: {player1}')
    # print(f'player2: {player2}')
    # print(f'court1: {court1}')
    # print(f'court2: {court2}')
    # print(f'net: {net}')

    return np.concatenate([ball_data, player_data, court_data, net])