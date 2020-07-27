import queue
import serial
# import threading
import time
from multiprocessing import Process, Queue
from .stdlib.collections import dotdict
from .stdlib.fixedloop import FixedLoop

READY = 0
FAIL = 1
DROP = 2


def read_latest_data(ser, labels):
    line = read_latest_line(ser)
    return parse_line(line, labels)


def clear_sensor_input(ser):
    ser.reset_input_buffer()
    ser.readline()


def read_latest_line(ser):
    clear_sensor_input(ser)
    return ser.readline()


def parse_line(line, labels):
    line = line.decode()
    segments = line.split()
    if len(segments) != len(labels):
        return None
    data = {label: int(segment) for label, segment in zip(labels, segments)}
    return data


def reader(port, baudrate, sampling_rate, data_labels, output):
    try:
        ser = serial.Serial()
        ser.baudrate = baudrate
        ser.port = port
        ser.parity = serial.PARITY_ODD
        ser.open()
        ser.close()
        ser.parity = serial.PARITY_NONE
        ser.open()
        time.sleep(2)

        data = read_latest_data(ser, data_labels)
        if data is None:
            output.put(FAIL)
            return

        output.put(READY)
        fixed_loop = FixedLoop(1 / sampling_rate)
        fixed_loop.reset()

        while True:
            if fixed_loop.last_delay_time >= 0:
                output.put(read_latest_data(ser, data_labels))
            else:
                output.put(DROP)

            fixed_loop.sync()

    except serial.SerialException:
        output.put(FAIL)
    except:
        output.put(FAIL)
