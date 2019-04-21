import array
import os
import wave
import matplotlib.pyplot as plt
import time
import numpy as np

class WaveRead:
    '''
    A class I put togther for reading wave files. Experimental.
    '''
    def __init__(self, filename, read=True, debug=False):
        mode = 'r' if read else 'w'
        sizes = {1: 'B', 2: 'h', 4: 'i'}
        self.wav = wave.open(filename, mode)
        self.filename = filename
        self.channels = self.wav.getnchannels()
        self.rate = self.wav.getframerate()
        self.count_frames = self.wav.getnframes()
        self.fmt_size = sizes[self.wav.getsampwidth()]
        self.fmt = "<" + self.fmt_size * self.channels
        # self.size = int(os.path.getsize(self.filename)/a.itemsize)
        # self.data = a.fromfile(open(self.filename, 'rb'), size)
    
    def read_whole(self):
        start = time.clock()
        print("Start: ", time.clock() - start)
        a = array.array(self.fmt_size)
        print("Array allocated: ", time.clock() - start)
        a.fromfile(open(self.filename, 'rb'), os.path.getsize(self.filename)//a.itemsize)
        print("Array from file: ", time.clock() - start)
        # calculate offset for 44B of header
        a = a[44//self.wav.getsampwidth():] # wave data starts at offset 44
        print("Offset calculated: ", time.clock() - start)
        # logging.debug("WH: {0}".format(len(a)))
        avg = lambda x, y: (x + y) / 2
        if self.channels == 2:
            left = a[::2] 
            # right = a[1::2]
            result = self.rectify(left)
            
            # data = np.array([np.frombuffer(left, dtype="int32"),np.frombuffer(right, dtype="int32")])
            # result = np.average(data, axis=0)
            # result = [abs(x + y)/2 for x, y in zip(left, right)]
            # result = list(map(avg, a[::2], a[1::2]))
            print("Slicing applied: ", time.clock() - start)
            return  self.normalize(result)
        return a

    def rectify(self, data):
        result = [abs(x) for x in data]
        return result

    def normalize(self, data):
        max_val = [max(data), min(data)][max(data) < abs(min(data))]
        result = [x/max_val for x in data]
        return result

    def running_mean(self, data, N):
        pad = N // 2
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        result =  (cumsum[N:] - cumsum[:-N]) / float(N)
        # restore initial length:
        result = np.pad(result, (pad, pad), 'constant', constant_values=(0,0))
        return self.normalize(result)

    def raise_to_power(self, data, pow):
        data = [x**pow for x in data]
        # max_val = max(data)
        # data = [x/max_val for x in data] # normalize
        return self.normalize(data)
    
    def average(self, data):
        avg = sum(data) / len(data)
        return [avg for x in data]

    def derivative(self, data, width):
        center = width // 2
        pad = [0] * center
        result = pad
        for i in range(center, len(data) - center):
            dydx = (data[i + center] - data[i - center]) / width
            result.append(dydx)
        result = self.rectify(result)
        result = self.normalize(result)
        result += [0] * center
        return result

    def test(self, data, width):
        # running_mean = self.running_mean(data, width)
        avg = self.average(data)
        center = width // 2
        result = [0] * len(data)
        last = 0
        next = 0
        for i in range(width, len(data) - 1):
            if data[i] > avg[i]*2 and data[i-width] > avg[i-width]*2:
                last = data[i] - data[i-width]
                next = data[i + 1] - data[i - width + 1]
                if last > 0 and next < 0:
                    pick = [i, i+1][last > abs(next)]
                    result[pick - center] = .5
        return result

# path = "C:\\Users\Elite22\Downloads\DeathDealer_Mix1rev1.wav"
path = "C:\\Users\Elite22\Downloads\DeathDealer_Mix1rev1 resampled 8kHz.wav"

if __name__ == '__main__':
    wr = WaveRead(path)
    data = wr.read_whole()
    exp = wr.raise_to_power(data, 4)
    rm100 = wr.running_mean(exp, 100)
    tst = wr.test(rm100, 100)
    # rm = wr.running_mean(data, 200)
    # deriv5 = wr.derivative(data, 5)
    # deriv200 = wr.derivative(data, 200)
    # deriv50 = wr.derivative(data, 50)
    avg = wr.average(rm100)
    # rm1k = wr.running_mean(data, 1000)
    # rm10k = wr.running_mean(10000)
    # rm100k = wr.running_mean(100000)
    # pad = (len(rm1k) - len(rm10k))//2
    # diff = rm1k - rm10k
    # plt.plot(avg)
    # plt.plot(wr.raise_to_power(10))
    # plt.plot(rm100)
    # plt.plot(rm1k)
    plt.plot(data)
    plt.plot(rm100)
    plt.plot(tst)
    plt.plot(avg)
    # plt.plot(deriv50)

    # plt.plot(wr.read_whole())
    plt.show()
    # print(wr.channels, wr.filename, wr.fmt, wr.fmt_size)
    # print('done')