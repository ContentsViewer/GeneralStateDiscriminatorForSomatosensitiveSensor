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


# class SensorStreamReader():
#     def __init__(self, port='/dev/ttyS9', baudrate=19200, samples_interval_ms=10):
#         self.data_queue = queue.Queue()
#         self.ser = serial.Serial(port, baudrate, timeout=None)
#         self.is_running = True
#         self.reader_thread = threading.Thread(target=self.reader)
#         self.reader_thread.daemon = True
#         self.reader_thread.start()

#     def reader(self):
#         self.ser.reset_input_buffer()

#         while self.is_running:
#             for line in self.ser:
#                 segments = line.split()
#                 if len(segments) == 3:
#                     self.data_queue.put(segments)
#                     # timestamp, pulse_width, flow_amount = segments
#                     # print(segments)

#     def __del__(self):
#         self.is_running = False
#         self.reader_thread.join()
#         self.ser.close()
