from math import sin, cos
import math
import matplotlib.pyplot as plt
import random

"""
Simple 1-D convolution
* functions to generate waveforms of various kinds
* functions to generate various kernels
* function to add noise
* a simple 1-D convolution function
* matplotlib.pyplot imported as a conveience
* TO DO - hysteresis threshold, transient detection 
"""
class Convolve:
    def __init__(self, array, kernel):
        self.array = array
        self.kernel = kernel

    def convolve(self, shift=True):
        """
        convolve(self.array, self.kernel)

        Parameters
        ----------
        self.array : list [1, 2, 3...]
            a list of numeric types
        self.kernel : list [1, 2, 3...]
            a list of numeric types
        Returns
        -------
        list
            a list containing the convolution of the 
            input self.array with the self.kernel. 
        """
        result_length = len(self.array) + len(self.kernel)
        result = [0] * result_length
        for i, elt in enumerate(self.array):
            for j, eltj in enumerate(self.kernel):
                result[i + j] += (elt * eltj)
        if shift:
            result = self.shift_left(result, len(self.kernel)//2)
            return result
        return result
    
    def shift_left(self, array, shift_ammount):
        return array[shift_ammount:]


class Waveform:
    def __init__(self, samplerate = 44100, frequency=1000):
        self.sample_rate = samplerate
        self.sample_interval = 1/samplerate
        self.fequency = frequency
        self.period = 1 / frequency
        self.samples_per_period = samplerate // frequency
        print(f"Waveform object created, f={frequency}, sample rate = {samplerate}")

    def sine(self, n_periods = 5):
        result = []
        amplitude = 0
        samples = self.samples_per_period * n_periods
        for i in range(samples):
            amplitude = math.sin(2 * math.pi * i / self.samples_per_period)
            result.append(amplitude)
        return result 

    def square(self, n_periods = 5):
        one_cycle = [1] * (self.samples_per_period // 2) + [-1] * (self.samples_per_period // 2)
        return one_cycle * n_periods

    def triangle(self, n_periods = 5, centered = True):
        up = []
        down = []
        seg_length = self.samples_per_period // 4 
        for i in range(seg_length):
            up.append(i / seg_length)
        for i in range(seg_length):
            down.append((seg_length - i) / seg_length)
        pos = up + down
        neg = [-x for x in pos]
        cycle = pos + neg
        result = cycle * n_periods
        return result

    def pulse(self, n_periods=5, duty_cycle=.1, centered=True):
        seg_length = self.samples_per_period // 2
        high = [1] * int(seg_length * duty_cycle)
        if centered:
            low = [-1] * int(seg_length * (1-duty_cycle))
        else:
            low = [0] * int(seg_length * (1-duty_cycle))
        cycle = low + high
        result = cycle * n_periods
        return result

    def sawtooth(self, n_periods=5, centered = True):
        result = []
        seg_length = self.samples_per_period // 2
        if centered:
            for i in range(-seg_length, seg_length):
                result.append(i / seg_length)
        else:
            for i in range(self.samples_per_period):
                result.append(i / self.samples_per_period)
        result = result * n_periods
        return result
    
    def step_1second(self, seconds = 1):
        """
        step()
            a 1 second step cut into 3
            segments; low|high|low

        Parameters
        ----------
        length: int
            the length of the returned array
        Returns
        -------
        list
            a list containing the step data 
        """
        length = self.sample_rate * seconds
        chunk = length // 3
        result = []
        low = [0] * chunk
        high = [1] * chunk 
        result = low + high + low
        return result

    def ramp_1second(self, seconds = 1):
        """
        ramp(length = 1000)
            a ramp of a given length cut into 3
            segments; low|ramp|high

        Parameters
        ----------
        length: int
            the length of the returned array
        Returns
        -------
        list
            a list containing the step data 
        """
        length = self.sample_rate * seconds    
        chunk = length // 3
        low = [0] * chunk
        high = [1] * chunk 
        ramp = []
        for i in range(chunk):
            ramp.append(i / chunk)
        result = low + ramp + high
        return result

    def impulse_1second(self, seconds = 1):
        """
        impulse(length = 1000)
            an impulse of a given length cut into 3
            segments; all zeros with 
            a single high value (1) low|high|high

        Parameters
        ----------
        length: int
            the length of the returned array
        Returns
        -------
        list
            a list containing the step data 
        """
        length = self.sample_rate * seconds
        chunk = length // 2
        low = [0] * chunk
        high = [1]
        return low + high + low

    def get_time_axis(self, obj):
        return [self.sample_interval * s for s in range(len(obj))]
    
    def get_average(self, arr):
        return sum([abs(x) for x in arr]) / len(arr)
    
    def get_rms(self, arr):
        return math.sqrt(sum([x**2 for x in arr]) / len(arr))

    def add_noise(self, arr, noise_percent):
        '''
        add_noise(arr, noise_percent)
        
        Parameters
        ----------
        arr :list
            the input array
        noise_percent : int or float
            determines the amount of noise added to the
            input array as a percent (1 = 100%) of the 
            RMS average of the input array
        Returns
        -------
        list
            a new list with added noise.
        '''
        # noise_percent 0.0 - 1.0
        avg_level = self.get_rms(arr)
        noise_scale_factor = avg_level * noise_percent
        result = [x + (random.random() - .5) * noise_scale_factor for x in arr]
        return result

class Kernel:

    def __init__(self, n_samples = 21, samplerate = 44100):
        self.n_samples = [n_samples, n_samples + 1][n_samples % 2 == 0]
        self.samplerate = samplerate
        print(f'Kernel created, length {n_samples}, sample rate {samplerate}')

    def normalize_sum(self, arr):
        _sum = sum([abs(x) for x in arr])
        result = [x / _sum for x in arr]
        return result
    
    def flip_left_right(self, arr):
        return arr[::-1]


    def gaussian(self,  width_factor = 1):
        """
        gaussian( width_factor = 1)

        Parameters
        ----------
        width_factor : int or float
            determines the width of the bell
            .001 is narrow, 1 is approximately 
            the width of the kernel. Width is similar
            to variance
        Returns
        -------
        list
            a list containing a gaussian kernel. 
        """
        kernel = [0] * self.n_samples
        for i in range(self.n_samples):
            x = (i-self.n_samples//2) / (self.n_samples//2)
            kernel[i] = math.pow(math.e, -(2 * math.pi * x**2)/ width_factor)
        kernel = self.normalize_sum(kernel)
        return kernel

    def gaussian_hp(self,  width_factor = 1):
        """
        gaussian_hp( width_factor = 1)
        TODO: Make scale_factor into a parameter

        Parameters
        ----------
        width_factor : int or float
            determines the width of the bell
            .001 is narrow, 1 is approximately 
            the width of the kernel. Width is similar
            to variance
        Returns
        -------
        list
            a list containing a gaussian high pass kernel. 
        """
        kernel = [0] * self.n_samples
        scale_factor = 5 # make this into a parameter
        for i in range(self.n_samples):
            x = (i-self.n_samples//2) / (self.n_samples//2)
            if x == 0:
                kernel[i] = scale_factor * math.pow(math.e, -(2 * math.pi * x**2)/ width_factor)
            else:
                kernel[i] = -1 / scale_factor * math.pow(math.e, -(2 * math.pi * x**2)/ width_factor)
        kernel = self.normalize_sum(kernel)
        return kernel


    def gaussian2(self, sigma = 1):

        result = []
        start = -self.n_samples // 2
        end = self.n_samples // 2
        for n in range(start, end):
            i = (n / end)
            amplitude = (1 / (2 * math.pi * sigma)**.5) / (math.pow(math.e, (i**2 / 2 * sigma )))
            result.append(amplitude)
        result = self.normalize_sum(result)
        return result

    # def gaussian_derivative(self,  width_factor = 1):
    #     kernel = gaussian(self.n_samples, width_factor)
    #     kernel = convolve(kernel, [1,-1])
    #     kernel = normalize_sum(kernel[:self.n_samples]) 
    #     kernel = [2*x for x in kernel]
    #     return kernel


    def parabolic(self):
        """
        parabolic(n_samples = 100)

        Parameters
        ----------
        n_samples : int
            length of the kernel
        Returns
        -------
        list
            an array containing a parabolic (Epanechnikov) kernel. 
        """
        kernel = []
        bound = self.n_samples // 2
        for i in range(-bound, bound + 1):
            data_point = .75 * (1 - (i / bound )**2)
            data_point /= bound
            kernel.append(data_point)
        return kernel

    def parabolic_hp(self):
        """
        parabolic_hp(n_samples = 100)

        Parameters
        ----------
        self.n_samples : int
            length of the kernel
        Returns
        -------
        list
            an array containing a parabolic (Epanechnikov) kernel. 
        """
        kernel = []
        bound = self.n_samples // 2
        for i in range(-bound, bound + 1):
            data_point = .75 * (1 - (i / bound )**2)
            data_point /= bound
            if i == 0:
                kernel.append(data_point)
            else:
                kernel.append(-data_point)
            
        return kernel

    def triangular(self):
        """
        triangular()

        Parameters
        ----------
        n_samples : int
            length of the kernel
        Returns
        -------
        list
            an array containing a triangular kernel. 
        """
        kernel = []
        bound = self.n_samples // 2
        for i in range(-bound, bound + 1):
            data_point = 1 - abs(i / bound)
            data_point /= bound
            kernel.append(data_point)
        return kernel

    def rectangular(self):
        """
        rectangular()

        Parameters
        ----------
        n_samples : int
            length of the kernel
        Returns
        -------
        list
            an array containing a rectangular kernel. 
        """
        kernel = [1/self.n_samples] * self.n_samples
        return kernel

    def rectangular_hp(self):
        """
        rectangular_hp()

        Parameters
        ----------
        self.n_samples : int
            length of the kernel
        Returns
        -------
        list
            an array containing a rectangular high pass kernel. 
        """
        middle_sample_index = self.n_samples // 2
        kernel = [-1/self.n_samples] * self.n_samples
        kernel[middle_sample_index] = 1
        kernel = self.normalize_sum(kernel)
        return kernel

    def sinc(self, frequency=1000, n_periods = 11):
        samples_per_period = self.samplerate // frequency
        samples = samples_per_period * n_periods
        start = -samples // 2
        end = samples // 2
        result = [math.sin(math.pi * x * n_periods / end) / (math.pi * x * n_periods / end) if x != 0 else 1 for x in range(start, end)]
        result = self.normalize_sum(result)
        result = [2*x for x in result]
        return result
    



    def sinc_hp(self, frequency=1000, n_periods = 11):
        samples_per_period = self.samplerate // frequency
        samples = samples_per_period * n_periods
        start = -samples // 2
        end = samples // 2
        result = [math.sin(math.pi * x * n_periods / start) / (math.pi * x * n_periods / end) if x != 0 else 1 for x in range(start, end)]
        result = self.normalize_sum(result)
        result = [2*x for x in result]
        return result

    def exp_hp(self):
        result = [1]
        for i in range(self.n_samples - 1):
            result.append(-math.e**(-i/20))
        result = self.normalize_sum(result)
        return result




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test = Waveform(frequency=100)
    # sin = test.sine()
    # sqr = test.square()
    # tri = test.triangle()
    # saw = test.sawtooth()
    step = test.impulse_1second()
    step = test.add_noise(step,1)
    plt.plot(step)
    
    # plt.plot(sin)
    # plt.plot(sqr)
    # plt.plot(tri)
    # plt.plot(saw)
    # plt.show()
    # # print([x for x in saw])
    # k = Kernel(n_samples=250)
    k = [0,0,0,0,1,-1,1,-1,0,0,0,0]
    # k_sinc = k.sinc(frequency=3000)
    # k_sinc_hp = k.sinc_hp(frequency=3000)
    # k_rect = k.rectangular()
    # k_para = k.parabolic()
    # k_exp_hp = k.exp_hp()
    # k_rect_hp = k.rectangular_hp()
    # k_para_hp = k.parabolic_hp()
    # k_gauss = k.gaussian()
    # k_gauss_hp = k.gaussian_hp()

    # w = Waveform(frequency=100)
    # sig = w.square(n_periods=10)
    a = Convolve(step, k)
    a_cnv = a.convolve()
    plt.plot(a_cnv)
    # a = Convolve(k_rect, k_rect)
    # a_cnv = a.convolve()
    # b = Convolve(a_cnv, k_rect)
    # b_cnv = b.convolve()
    # c = Convolve(b_cnv, k_rect)
    # c_cnv = c.convolve()
    # plt.plot(a_cnv)
    # plt.plot(b_cnv)
    # plt.plot(c_cnv)

    # c1 = Convolve(sig, k_para)
    # cnv_para = c1.convolve()
    # c2 = Convolve(sig, k_rect)
    # cnv_rect = c2.convolve()
    # c3 = Convolve(sig, k_sinc)
    # cnv_sinc = c3.convolve()
    # c4 = Convolve(sig, k_sinc_hp)
    # cnv_sinc_hp = c4.convolve()
    # c5 = Convolve(sig, k_exp_hp)
    # cnv_exp_hp = c5.convolve()
    # c6 = Convolve(sig,k_rect_hp)
    # cnv_rect_hp = c6.convolve()
    # c7 = Convolve(sig, k_para_hp)
    # cnv_para_hp = c7.convolve()
    # c8 = Convolve(sig, k_gauss)
    # cnv_gauss = c8.convolve()
    # c9 = Convolve(sig, k_gauss_hp)
    # cnv_gauss_hp = c9.convolve()
    # plt.plot(k_sinc)
    # plt.plot(k_sinc_hp)
    # plt.plot(sig)
    # plt.plot(k_gauss)
    # plt.plot(k_gauss_hp)
    # plt.plot(cnv_gauss)
    # plt.plot(cnv_gauss_hp)
    # plt.plot(cnv_rect)
    # plt.plot(cnv_sinc)
    # plt.plot(cnv_sinc_hp)

    # plt.plot(cnv_rect)
    # plt.plot(cnv_rect_hp)
    plt.show()
    # plt.plot(k.gaussian())
    # plt.plot(k.gaussian2())
    # plt.plot(k.gaussian2(sigma=.5))
    # plt.plot(k.gaussian2(sigma=math.pi**2))
    # plt.plot(k.gaussian2(sigma=100))
    # plt.plot(k.triangular())
    # plt.plot(k.rectangular())
    # plt.plot(k.parabolic())
    # plt.plot(k.sinc(frequency=1000, n_periods=7))
    # plt.show()

#     def waveform(self, type = "square", cycles=5, samples_per_cycle = 100, duty_cycle=.5):
#         '''
#         wave_generator(type = "sine", cycles=5, samples_per_cycle = 100):
        
#         Parameters
#         ----------
#         type : str
#             sine, square, impulse, pulse, triangle
#         cycles : int 
#             the number of wave cycles to generate
#         samples_per_cycle : int
#             the number of data points per wave cycle
#         Returns
#         -------
#         list
#             a list containing the output data
#         '''
#         total_samples = cycles * samples_per_cycle
#         result = []
#         if type.lower() in ["square", "sqr", "sq"]:
#             high = [1] * int(samples_per_cycle * duty_cycle)
#             low = [-1] * int(samples_per_cycle * (1-duty_cycle))
#             cycle = low + high
#             result = cycle * cycles
#             return result
#         elif type.lower() in ['pulse', 'pls']:
#             seg_length = samples_per_cycle // 2
#             high = [1] * int(seg_length * duty_cycle)
#             low = [0] * int(seg_length * (1-duty_cycle))
#             cycle = low + high
#             result = cycle * cycles
#             return result
#         elif type.lower() in ['imp', 'impulse']:
#             length = cycles * samples_per_cycle
#             low = [0] * (length // 2)
#             high = [1]
#             return low + high + low
#         elif type.lower() in ['sin', 'sine']:
#             for i in range(total_samples):
#                 result.append(math.sin(2 / samples_per_cycle * math.pi * i % samples_per_cycle))
#             return result
#         elif type.lower() in ['tri', 'triangle']:
#             up = []
#             down = []
#             seg_length = samples_per_cycle // 4 
#             for i in range(seg_length):
#                 up.append(i / seg_length)
#             for i in range(seg_length):
#                 down.append((seg_length - i) / seg_length)
#             pos = up + down
#             neg = [-x for x in pos]
#             cycle = pos + neg
#             result = cycle * cycles
#             return result

#     def step(length = 1000):
#         """
#         step(length = 1000)
#             a step of a given length cut into 3
#             segments; low|high|low

#         Parameters
#         ----------
#         length: int
#             the length of the returned array
#         Returns
#         -------
#         list
#             a list containing the step data 
#         """
#         chunk = length // 3
#         result = []
#         low = [0] * chunk
#         high = [1] * chunk 
#         result = low + high + low
#         return result

#     def ramp(length = 1000):
#         """
#         ramp(length = 1000)
#             a ramp of a given length cut into 3
#             segments; low|ramp|high

#         Parameters
#         ----------
#         length: int
#             the length of the returned array
#         Returns
#         -------
#         list
#             a list containing the step data 
#         """    
#         chunk = length // 3
#         low = [0] * chunk
#         high = [1] * chunk 
#         ramp = []
#         for i in range(chunk):
#             ramp.append(i / chunk)
#         result = low + ramp + high
#         return result

#     def impulse(length=1000):
#         """
#         impulse(length = 1000)
#             an impulse of a given length cut into 3
#             segments; all zeros with 
#             a single high value (1) low|high|high

#         Parameters
#         ----------
#         length: int
#             the length of the returned array
#         Returns
#         -------
#         list
#             a list containing the step data 
#         """
#         chunk = length // 2
#         low = [0] * chunk
#         high = [1]
#         return low + high + low

#     def get_average(arr):
#         return sum([abs(x) for x in arr]) / len(arr)
    
#     def get_rms(arr):
#         return math.sqrt(sum([x**2 for x in arr]) / len(arr))

#     def add_noise(arr, noise_percent):
#         '''
#         add_noise(arr, noise_percent)
        
#         Parameters
#         ----------
#         arr :list
#             the input array
#         noise_percent : int or float
#             determines the amount of noise added to the
#             input array as a percent (1 = 100%) of the 
#             RMS average of the input array
#         Returns
#         -------
#         list
#             a new list with added noise.
#         '''
#         # noise_percent 0.0 - 1.0
#         avg_level = get_rms(arr)
#         noise_scale_factor = avg_level * noise_percent
#         result = [x + (random.random() - .5) * noise_scale_factor for x in arr]
#         return result








# def normalize_peak(arr):
#     _max = [max(arr), abs(min(arr))][max(arr) < abs(min(arr))]
#     result = [x / _max for x in arr]
#     return result




# def threshold(arr):
#     result = [0] * len(arr)
#     for i, sample in enumerate(arr):
#         if sample > .75:
#             result[i] = sample
#     return result

# def chunk_max(arr, chunk_length=5000):
#     result = [0] * len(arr)
#     start = 0
#     end = chunk_length
#     chunk_number = 0
#     last_idx = 0
#     n_chunks = len(arr) // chunk_length
#     last_chunk_len = len(arr) % chunk_length
#     for i in range(n_chunks):
#         chunk = arr[start:end]
#         _max = max(chunk)
#         idx = chunk.index(_max)
#         # print(f'start = {start}, end = {end}, index found = {idx}')
#         print(f'idx = {idx}, last_idx = {last_idx}, idx - last_idx = {idx- last_idx}')
#         if not last_idx:
#             result[chunk_number * chunk_length + idx] = 1
#             last_idx = chunk_number * chunk_length +  idx
#         if chunk_number * chunk_length + idx - last_idx >= chunk_length // 4:
#             result[chunk_number * chunk_length + idx] = 1
#             last_idx = chunk_number * chunk_length +  idx
#         start += chunk_length
#         end += chunk_length
#         chunk_number += 1
#     chunk = arr[end : end + last_chunk_len]
#     if chunk:
#         _max = max(chunk)
#         idx = chunk.index(_max)
#         result[chunk_number * chunk_length + idx] = 1
#     return result




# def high_pass(arr, alpha):
#     # alpha is equivalent to an RC time constant
#     # small alpha (.1) has a wider stop band
#     # higher alpha (1) has a narrower stop band
#     # for i from 1 to n
#     #  y[i] := α * (y[i-1] + x[i] - x[i-1])
#     result = [0]* len(arr)
#     for i in range(1, len(arr)):
#         result[i] = alpha * (result[i-1] + arr[i] - arr[i-1])
#     result = normalize_peak(result)
#     return result

# def find_reversals(arr):
#     result = [0] * len(arr)
#     for i in range(1, len(arr)-1):
#         if arr[i-1] < arr[i] > arr[i+1]:
#             if abs(arr[i]) > .25:
#                 result[i] = 1
#     return result



