from transmitter import *
import numpy as np

T = 20
N = 2**12
dt = T/N
t = np.linspace(-T/2, T/2, N)

F = 1/dt
df = 1/T
f = np.linspace(-F/2, F/2, N)

M = 16
ns = 3
nb = 1024
p = 0.5
b = source(nb, p)
cnt = QamConstellation(M, 1)
s = bit_to_symb(b, cnt)
#s = np.array([1, 1*1j, 5, -1, 2+2*1j])
B = 1e2
q0 = mod(t, s ,B)

s_projection = demod(q0, t, len(s), B)
s_hat = [cnt.closest_neighboor(s) for s in s_projection]
print(s== s_hat)

""" plt.subplot(2,1,1)
plt.plot(t, np.abs(q0))
plt.title('Time domain')
plt.subplot(2,1,2)
plt.plot(f, np.fft.fftshift(np.abs(np.fft.fft(q0))))
plt.title('FFT')
plt.show() """