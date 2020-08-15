
import pygame
from pygame.locals import *

import os
import sys
import time
import threading
import serial
import tqdm
import argparse
import importlib
from collections import deque
import numpy as np
from multiprocessing import Process, Queue

import tensorflow as tf
from tensorflow.keras import Model

from lib.layers import Duplicate
import lib.sensor as sensor
import lib.facenet as facenet
from lib.gui import GUI

from lib.stdlib.stopwatch import Stopwatch, stopwatch, stopwatch_scope
from lib.stdlib.fixedloop import FixedLoop
from lib.stdlib.collections import dotdict


def main(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    config = importlib.import_module(args.config).Config

    app_timer = Stopwatch()
    app_timer.start()

    # --- Setup Serial ---
    print('Serial Start up...')
    sensor_output = Queue()
    serial_process = Process(
        target=sensor.reader,
        daemon=True,
        args=(config.port, config.baudrate, config.sampling_rate,
              config.sensor_data_labels, sensor_output)
    )
    serial_process.start()

    if sensor_output.get() is sensor.FAIL:
        raise RuntimeError('Serial Process Fail.')

    # --- Setup GUI ---
    print('GUI is launching...')
    pygame.init()
    GUI.init()
    screen = pygame.display.set_mode((720, 483))
    pygame.display.set_caption("STEM")
    pygame.display.set_icon(GUI.make_text('|†|', font=pygame.font.Font(
        "fonts/Ubuntu Mono derivative Powerline Bold.ttf", 64), color=GUI.color.black))
    # End Setup GUI ---

    # --- Setup Model ---
    print('Model Setup...')
    base_model = importlib.import_module(config.model_path).Model()
    inputs = tf.keras.Input(
        shape=(config.model_sensor_data_inputs *
               len(config.sensor_feature_labels),),
        batch_size=None
    )
    x = base_model(inputs)
    outputs = Duplicate()(x)
    model = Model(inputs, outputs)
    model.compile(loss=facenet.triplet_loss(), optimizer='adam')

    base_model.summary()
    model.summary()
    if os.path.exists(config.checkpoint):
        print('last checkpoint found. Loading the weights...')
        model.load_weights(config.checkpoint)

    # model.save_weights(args.checkpoint)

    # End Setup Model ---

    is_running = True
    profile = {
        'keyboard_input': 0,
        'sensor_read': 0,
        'gui_update': 0,
        'screen_update': 0
    }
    input_queue = deque(
        maxlen=config.model_sensor_data_inputs *
        len(config.sensor_feature_labels)
    )

    # 0 -> 'label A', 1 -> 'label B', ...
    id2label = [None for _ in range(len(config.possible_states))]
    label2id = {}

    precedents_dict = [deque(maxlen=config.precedents_maxlen)
                       for _ in range(len(config.possible_states) + 1)]

    estimator = facenet.Estimator(
        model=model,
        precedents_dict=precedents_dict,
    )

    trainor = facenet.Trainor(
        precedents_dict=precedents_dict
    )

    prev_saving_precedent_time = app_timer.elapsed
    fixed_loop = FixedLoop(1 / config.frame_rate)
    fixed_loop.reset()
    fixed_loop.sync()
    estimated = None
    while is_running:
        screen.fill(GUI.color.screen_backgorund)
        submitted_estimator = False
        submitted_trainor = False
        supervised_state_label = None
        app_timer.lap()

        # --- Keyboard Input ---
        pressed_keys = pygame.key.get_pressed()
        for label, settings in config.possible_states.items():
            if pressed_keys[settings['key']]:
                supervised_state_label = label

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    is_running = False

        profile['keyboard_input'] = app_timer.lap()
        # End Keyboard Input ---

        # --- Control Module Behaviors ---
        # print(sensor.read_latest_data(ser))
        # sensor_data = sensor.read_latest_data(ser, config.sensor_data_labels)
        # sensor_data = dotdict(
        #     {'timestamp': time.time(), 'pulse_width': 100, 'flow_amount': 200})

        received_size = sensor_output.qsize()
        for _ in range(received_size):
            sensor_data = sensor_output.get()
            if sensor_data is sensor.DROP:
                input_queue.clear()
            elif sensor_data is sensor.FAIL:
                raise RuntimeError('Serial Process Fail.')
            else:
                for label in config.sensor_feature_labels:
                    input_queue.append(sensor_data[label])

        profile['sensor_read'] = app_timer.lap()

        if len(input_queue) >= input_queue.maxlen:
            input_list = np.array([input_queue])
            if not estimator.is_running:
                submitted_estimator = True
                estimator.run(inputs=input_list,
                              supervised_state_label=supervised_state_label)

        count = 0
        for state_id in range(len(precedents_dict) - 1):
            count += len(precedents_dict[state_id])
        if count > len(config.possible_states):
            precedents_dict[-1].clear()

        if not estimator.results.empty():
            estimated = estimator.results.get()

            if (estimated.supervised_state_label is not None) and (estimated.estimated_state < len(config.possible_states)):
                if id2label[estimated.estimated_state] is None:
                    id2label[estimated.estimated_state] = estimated.supervised_state_label
                    label2id[estimated.supervised_state_label] = estimated.estimated_state

                if label2id.get(estimated.supervised_state_label) is None:
                    aligned_id = 0
                    for i, label in enumerate(id2label):
                        if label is None:
                            aligned_id = i
                            break

                    id2label[aligned_id] = estimated.supervised_state_label
                    label2id[estimated.supervised_state_label] = aligned_id

                estimated.supervised_state = label2id[estimated.supervised_state_label]
                # print(estimated.supervised_state)
            if (estimated.supervised_state_label is not None) or app_timer.elapsed > prev_saving_precedent_time + config.precedent_interval:
                if not trainor.is_running:
                    submitted_trainor = True
                    trainor.run(model=model, anchor=estimated)
                
                major_state = facenet.get_major_state(estimated)
                
                precedents_dict[major_state].append(estimated)
                prev_saving_precedent_time = app_timer.elapsed

        if not trainor.results.empty():
            pass

        # print(len(input_queue))
        profile['submodule_control'] = app_timer.lap()
        # End Control Module Behaviors ---

        # --- Update GUI Elements ---
        if supervised_state_label is None:
            screen.blit(GUI.make_text(
                'Self Learning...', GUI.font.large), (400, 0))
        else:
            screen.blit(GUI.make_text('Supervised... {0}'.format(
                supervised_state_label), GUI.font.large), (400, 0))

        screen.blit(GUI.make_text('State: ', GUI.font.large), (0, 0))
        if estimated is not None and estimated.estimated_state < len(config.possible_states):
            current_state = facenet.get_major_state(estimated)

            screen.blit(GUI.make_text(
                '{0} {1}'.format(current_state, '?' if id2label[current_state] is None else id2label[current_state]), GUI.font.large), (80, 0))
        else:
            screen.blit(GUI.make_text(
                '?', GUI.font.large), (80, 0))

        GUI.begin_multilines((400, 30))
        GUI.draw_multiline_text(screen, "Sensor:")
        if sensor_data is sensor.DROP:
            GUI.draw_multiline_text(screen, '  DROP!')
        else:
            GUI.draw_multiline_text(screen,
                                    "\n".join(["  {0}: {1}".format(
                                        label, sensor_data[label]) for label in config.sensor_data_labels])
                                    )

        GUI.draw_multiline_text(screen,
                                (
                                    "App:\n"
                                    "  Time            : {0:.3f}\n"
                                    "  input_queue.size: {1}\n"
                                ).format(
                                    app_timer.elapsed,
                                    len(input_queue),
                                ))
        GUI.draw_multiline_text(screen,
                                (
                                    "  precedents.size :" +
                                    (", ".join(
                                        ["{0}".format(len(precedents)) for precedents in precedents_dict]))
                                ))
        GUI.draw_multiline_text(screen, "Estimator:")
        GUI.draw_multiline_text(screen,
                                '  o',
                                color=GUI.color.green if submitted_estimator else GUI.color.red)
        GUI.draw_multiline_text(screen, "Trainor:")
        GUI.draw_multiline_text(screen,
                                '  o',
                                color=GUI.color.green if submitted_trainor else GUI.color.red)
        GUI.draw_multiline_text(screen,
                                (
                                    "Profile:\n"
                                    "  Keyboard Input   : {1:.4f}\n"
                                    "  Sensor Read      : {2:.4f}\n"
                                    "  Submodule Control: {3:.4f}\n"
                                    "  GUI Update       : {4:.4f}\n"
                                    "  Screen Update    : {5:.4f}\n"
                                ).format(
                                    estimator.is_running,
                                    profile['keyboard_input'],
                                    profile['sensor_read'],
                                    profile['submodule_control'],
                                    profile['gui_update'],
                                    profile['screen_update']
                                ))
        if fixed_loop.last_delay_time >= 0:
            screen.blit(GUI.make_text('Frame: Sync ({0:.3f})'.format(
                fixed_loop.last_delay_time), color=GUI.color.green), (0, 463))
        else:
            screen.blit(GUI.make_text('Frame: Busy ({0:.3f})'.format(
                fixed_loop.last_delay_time), color=GUI.color.red), (0, 463))

        if sensor_data is sensor.DROP:
            screen.blit(GUI.make_text('Sensor: Busy',
                                      color=GUI.color.red), (240, 463))
        else:
            screen.blit(GUI.make_text('Sensor: Sync',
                                      color=GUI.color.green), (240, 463))

        pygame.draw.rect(screen, (0x11, 0x11, 0x11),
                         pygame.Rect(10, 30, 380, 380))
        meta, plots = facenet.make_visualized_graph_plots(precedents_dict, estimated)
        if meta is not None:
            scale = meta.max - meta.min
            a = 190 / scale.max()
            root = np.array([10 + 190, 30 + 190])
            for plot in plots:
                position = root + plot.position * a
                position = position.astype(np.int64)
            
                if plot.supervised_state is not None:
                    pygame.draw.circle(
                        screen, config.id2color[plot.supervised_state], position, 6)

                if plot.estimated_state is not None:
                    pygame.draw.circle(
                        screen, config.id2color[plot.estimated_state], position, 4)

        profile['gui_update'] = app_timer.lap()
        # End Update GUI Elements ---

        pygame.display.update()
        profile['screen_update'] = app_timer.lap()
        fixed_loop.sync()

    pygame.quit()
    exit()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        help="Path to the config file.",
        default='configs.default'
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
