import tkinter as tk
import threading
import serial

import tensorflow as tf
from tensorflow.keras import Model 
from lib.models import NN
from lib.layers import Duplicate
import lib.facenet as facenet

from lib.stdlib.keyboard import KeyboardInput

class Application():
    def __init__(self):
        self.window_wroker_thread = threading.Thread(target=self.window_worker)
        self.window_wroker_thread.start()

    def window_worker(self):
        self.window = tk.Tk()
        self.window.geometry("900x400")
        self.window.title("STEM")
        self.window.mainloop()

def main():

    key_input = KeyboardInput()

    # for i in range(5):
    #     sleep(2)
    #     Input.update()
    print('TEST')
    app = Application()

    exit()
    inputs = tf.keras.Input(shape=(256, ))
    x = NN()(inputs)
    outputs = Duplicate()(x)
    model = Model(inputs, outputs)
    model.compile(loss=facenet.triplet_loss(), optimizer='adam')

    model.summary()

    ser = serial.Serial('/dev/ttyS9', 19200, timeout=None)
    
    line=ser.readline()
    print(line)
    ser.close()


if __name__ == "__main__":
    main()
