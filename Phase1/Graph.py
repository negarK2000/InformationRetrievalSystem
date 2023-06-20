import matplotlib.pyplot as plt
import math as m
import  numpy as np


def plot_zipf_law(freq_dict, title):
    x = []
    y = []

    i = 1
    for word in freq_dict:
        x.append(m.log10(i))
        y.append(m.log10(freq_dict[word]))

        i += 1

    plt.plot(x, y, 'b')

    log_k = m.log10(freq_dict[list(freq_dict.keys())[0]])
    log_i = np.log10(np.linspace(2, len(freq_dict), len(freq_dict) - 1))
    log_cf = log_k - log_i

    plt.plot(log_i, log_cf, 'r')

    plt.xlabel('log10_index')
    plt.ylabel('log10_frequency')
    plt.title(title)
    plt.legend(["Actual Values", "Zipf's Law"])

    plt.show()

    return


def plot_heap_law(heap_dict, title):
    log_t = []
    log_m = []

    print(title)
    print('real values of tokens:')
    for num in heap_dict:
        log_t.append(m.log10(num))
        log_m.append(m.log10(heap_dict[num]))
        print('{} , {}'.format(num, heap_dict[num]))

    print()

    plt.plot(log_t, log_m, 'b')

    b = (log_m[1] - log_m[0])/(log_t[1] - log_t[0])
    log_k = log_m[0] - b * log_t[0]
    approx_log_m = log_k + b * np.array(log_t)

    print('approximate values of tokens:')
    for i in range(len(log_t)):
        tt = 10 ** log_t[i]
        mm = 10 ** approx_log_m[i]
        print('{} , {}'.format(tt, mm))

    print()

    plt.plot(log_t, approx_log_m, 'r')

    plt.xlabel('log10_T')
    plt.ylabel('log10_M')
    plt.title(title)
    plt.legend(["Actual Values", "Heap Law Approximate"])

    plt.show()

    return
