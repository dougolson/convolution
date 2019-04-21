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
def shift_left(arr, shift_ammount):
    return arr[shift_ammount:]

def convolve(array, kernel, shift=True):
    """
    convolve(array, kernel)

    Parameters
    ----------
    array : list [1, 2, 3...]
        a list of numeric types
    kernel : list [1, 2, 3...]
        a list of numeric types
    Returns
    -------
    list
        a list containing the convolution of the 
        input array with the kernel. 
    """
    result_length = len(array) + len(kernel)
    result = [0] * result_length
    for i, elt in enumerate(array):
        for j, eltj in enumerate(kernel):
            result[i + j] += (elt * eltj)
    if shift:
        result = shift_left(result, len(kernel)//2)
        return result
    return result

def normalize_sum(arr):
    _sum = sum([abs(x) for x in arr])
    result = [x / _sum for x in arr]
    return result

def normalize_peak(arr):
    _max = [max(arr), abs(min(arr))][max(arr) < abs(min(arr))]
    result = [x / _max for x in arr]
    return result

def kernel_gaussian(n_samples=100, width_factor = 1):
    """
    kernel_gaussian(n_samples=100, width_factor = 1)

    Parameters
    ----------
    n_samples : int
        length of the kernel
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
    n_samples = [n_samples, n_samples + 1][n_samples % 2 == 0]
    kernel = [0] * n_samples
    width = width_factor #n_samples * width_factor
    for i in range(n_samples):
        x = (i-n_samples//2) / (n_samples//2)
        kernel[i] = math.pow(math.e, -(2 * math.pi * x**2)/ width_factor)
    kernel = normalize_sum(kernel)
    return kernel



def kernel_gaussian_derivative(n_samples=100, width_factor = 1):
    kernel = kernel_gaussian(n_samples, width_factor)
    kernel = convolve(kernel, [1,-1])
    kernel = normalize_sum(kernel[:n_samples]) 
    kernel = [2*x for x in kernel]
    return kernel


def kernel_parabolic(n_samples = 100):
    """
    kernel_parabolic(n_samples = 100)

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
    bound = n_samples // 2
    for i in range(-bound, bound + 1):
        data_point = .75 * (1 - (i / bound )**2)
        data_point /= bound
        kernel.append(data_point)
    return kernel

def kernel_triangular(n_samples = 100):
    """
    kernel_triangular(n_samples = 100)

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
    bound = n_samples // 2
    for i in range(-bound, bound + 1):
        data_point = 1 - abs(i / bound)
        data_point /= bound
        kernel.append(data_point)
    return kernel

def kernel_rectangular(n_samples = 100):
    """
    kernel_rectangular(n_samples = 100)

    Parameters
    ----------
    n_samples : int
        length of the kernel
    Returns
    -------
    list
        an array containing a rectangular kernel. 
    """
    kernel = [1/n_samples] * n_samples
    return kernel

def waveform(type = "square", cycles=5, samples_per_cycle = 100, duty_cycle=.5):
    '''
    wave_generator(type = "sine", cycles=5, samples_per_cycle = 100):
    
    Parameters
    ----------
    type : str
        sine, square, impulse, pulse, triangle
    cycles : int 
        the number of wave cycles to generate
    samples_per_cycle : int
        the number of data points per wave cycle
    Returns
    -------
    list
        a list containing the output data
    '''
    total_samples = cycles * samples_per_cycle
    result = []
    if type.lower() in ["square", "sqr", "sq"]:
        high = [1] * int(samples_per_cycle * duty_cycle)
        low = [-1] * int(samples_per_cycle * (1-duty_cycle))
        cycle = low + high
        result = cycle * cycles
        return result
    elif type.lower() in ['pulse', 'pls']:
        seg_length = samples_per_cycle // 2
        high = [1] * int(seg_length * duty_cycle)
        low = [0] * int(seg_length * (1-duty_cycle))
        cycle = low + high
        result = cycle * cycles
        return result
    elif type.lower() in ['imp', 'impulse']:
        length = cycles * samples_per_cycle
        low = [0] * (length // 2)
        high = [1]
        return low + high + low
    elif type.lower() in ['sin', 'sine']:
        for i in range(total_samples):
            result.append(math.sin(2 / samples_per_cycle * math.pi * i % samples_per_cycle))
        return result
    elif type.lower() in ['tri', 'triangle']:
        up = []
        down = []
        seg_length = samples_per_cycle // 4 
        for i in range(seg_length):
            up.append(i / seg_length)
        for i in range(seg_length):
            down.append((seg_length - i) / seg_length)
        pos = up + down
        neg = [-x for x in pos]
        cycle = pos + neg
        result = cycle * cycles
        return result

def step(length = 1000):
    """
    step(length = 1000)
        a step of a given length cut into 3
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
    chunk = length // 3
    result = []
    low = [0] * chunk
    high = [1] * chunk 
    result = low + high + low
    return result

def ramp(length = 1000):
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
    chunk = length // 3
    low = [0] * chunk
    high = [1] * chunk 
    ramp = []
    for i in range(chunk):
        ramp.append(i / chunk)
    result = low + ramp + high
    return result

def impulse(length=1000):
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
    chunk = length // 2
    low = [0] * chunk
    high = [1]
    return low + high + low

def get_average(arr):
    return sum([abs(x) for x in arr]) / len(arr)
   
def get_rms(arr):
    return math.sqrt(sum([x**2 for x in arr]) / len(arr))

def add_noise(arr, noise_percent):
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
    avg_level = get_rms(arr)
    noise_scale_factor = avg_level * noise_percent
    result = [x + (random.random() - .5) * noise_scale_factor for x in arr]
    return result

def threshold(arr):
    result = [0] * len(arr)
    for i, sample in enumerate(arr):
        if sample > .75:
            result[i] = sample
    return result

def chunk_max(arr, chunk_length=5000):
    result = [0] * len(arr)
    start = 0
    end = chunk_length
    chunk_number = 0
    last_idx = 0
    n_chunks = len(arr) // chunk_length
    last_chunk_len = len(arr) % chunk_length
    for i in range(n_chunks):
        chunk = arr[start:end]
        _max = max(chunk)
        idx = chunk.index(_max)
        # print(f'start = {start}, end = {end}, index found = {idx}')
        print(f'idx = {idx}, last_idx = {last_idx}, idx - last_idx = {idx- last_idx}')
        if not last_idx:
            result[chunk_number * chunk_length + idx] = 1
            last_idx = chunk_number * chunk_length +  idx
        if chunk_number * chunk_length + idx - last_idx >= chunk_length // 4:
            result[chunk_number * chunk_length + idx] = 1
            last_idx = chunk_number * chunk_length +  idx
        start += chunk_length
        end += chunk_length
        chunk_number += 1
    chunk = arr[end : end + last_chunk_len]
    if chunk:
        _max = max(chunk)
        idx = chunk.index(_max)
        result[chunk_number * chunk_length + idx] = 1
    return result




def high_pass(arr, alpha):
    # alpha is basically an RC time constant
    # for i from 1 to n
    #  y[i] := α * (y[i-1] + x[i] - x[i-1])
    result = [0]* len(arr)
    for i in range(1, len(arr)):
        result[i] = alpha * (result[i-1] + x[i] - x[i-1])
    return result



