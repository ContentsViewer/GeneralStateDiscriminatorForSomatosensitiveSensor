
from pygame.locals import *


class Config():

    baudrate = 19200

    port = '/dev/ttyACM0'

    sensor_data_labels = [
        'timestamp',
        'pulse_width',
        'flow_amount'
    ]

    sensor_feature_labels = [
        'pulse_width',
        'flow_amount'
    ]

    model_path = 'lib.models.lstm'

    # path to checkpoint for saving the model parameter.
    # if checkpoint is already exists, load the model parameter before application start.
    checkpoint = './checkpoints/default/checkpoint'

    model_sensor_data_inputs = 64

    precedents_maxlen = 100

    # [s]
    precedent_interval = 10

    sampling_rate = 32
    frame_rate = 30

    possible_states = {
        'not_touched': {
            'key': K_n
        },
        'toched': {
            'key': K_m
        }
    }

    id2color = [
        (0xFF, 0xFF, 0x00),
        (0xFF, 0x00, 0xFF),
        (0xFF, 0xFF, 0xFF),
        (0x00, 0x00, 0xFF),
        (0xFF, 0x00, 0x00),
        (0x00, 0xFF, 0x00),
    ]
