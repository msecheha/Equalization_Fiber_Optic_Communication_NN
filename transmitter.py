import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
import itertools

def source(N,p):
    # Returns the outcome of N successive trials of a binomial variable with probability p of success
    return np.array(random.rand(N)<=p, dtype=int)

class QamConstellation:
    # Creates a QAM Constellation with a Gray code assigned to every symbols
    # M : Number of symbols
    # P : Energy of the constellation
    def __init__(self, M, P):
        self.M = M
        self.P = P
        self.points = []
        # Basic QAM with unitary minimum distance between symbols
        coord_list = np.arange(-np.sqrt(self.M)+1, np.sqrt(self.M), step=2) 
        for x,y in itertools.product(coord_list, coord_list):
            self.points.append((x + 1j * y))
        # Scaling of the constellation to obtain the desired energy
        self.points = np.sqrt(self.P/self.average_energy())*np.array(self.points)
        # Points are sorted to obtain an easy-to-use indexing
        self.points = sorted(self.points, key=lambda x:x.imag)

        self.neighboors = {i: [] for i in range(len(self.points))}
        self.find_neighboors()

        self.gray_mapping = {i: 'No Code' for i in range(len(self.points))}
        self.gray_mapping[0] = 0
        self.gray_coding()

        self.gray_to_point = {}
        for i in range(len(self.points)):
            self.gray_to_point[bin(self.gray_mapping[i])[2:].zfill(np.log2(self.M).astype(int))] = self.points[i]
        
    def average_energy(self):
    # Computes the average energy of the constellation before scaling
        return (1/self.M)*np.sum(np.abs(self.points)**2)
    
    def find_neighboors(self):
    # Find the neighboors of each point in the constellation to create a Gray code 
        for i in range(len(self.points)):
            # We use grid coordinates here to take borders into account
            row = i//np.sqrt(self.M)
            col = i%np.sqrt(self.M)
            for dx in [-1, 1]:            
                if (row + dx >=0) and (row + dx <= np.sqrt(self.M)-1): 
                    self.neighboors[i].append(((row+dx)*np.sqrt(self.M) + col).astype(int))
            for dy in [-1, 1]:
                if (col + dy >=0) and (col + dy <= np.sqrt(self.M)-1): 
                    self.neighboors[i].append((row*np.sqrt(self.M) + col + dy).astype(int))
    
    def gray_coding(self):
    # Assigns to each point of the constellation a code such that the constellation is Gray coded
        possible_codes = [k for k in range(1,self.M)]
        for i in range(1,len(self.points)):
            if self.gray_mapping[i]=='No Code':
                already_coded_neighboors = [k for k in self.neighboors[i] if self.gray_mapping[k] != 'No Code']
                for code_index,code in enumerate(possible_codes):
                    # A code is possible for Gray coding iff it is only one bit different w.r.t each already coded neighboors
                    is_possible_gray_code = np.sum([bin(self.gray_mapping[neighboor]^code).count('1')==1 
                                                    for neighboor in already_coded_neighboors]) == len(already_coded_neighboors)
                    if is_possible_gray_code:
                        assigned_code = possible_codes.pop(code_index)
                        self.gray_mapping[i] = assigned_code
                        break

    def closest_neighboor(self, symbol):
        points_mat = np.array(self.points)
        dist = np.abs(symbol-points_mat)
        idx = np.argmin(dist)
        return self.points[idx]

    def plot(self):
        x = [point.real for point in self.points]
        y = [point.imag for point in self.points]
        plt.scatter(x,y)
        plt.grid()
        for i in range(len(self.points)):
            plt.text(x=self.points[i].real, y=self.points[i].imag, s=bin(self.gray_mapping[i])[2:].zfill(np.log2(self.M).astype(int)))
        plt.show()

def bit_to_symb(b, cnt):
# Converts a bitstream to complex symbols using QAM
# b : bitstream
# cnt : QAM constellation
    n_s = len(b)//np.log2(cnt.M).astype(int)
    s = []
    for k in range(0, n_s):
        subsequence = b[k*np.log2(cnt.M).astype(int):(k+1)*np.log2(cnt.M).astype(int)]
        code = ''.join(str(bit) for bit in subsequence)
        symbol = cnt.gray_to_point[code]
        s.append(symbol)
    return s

def mod(t, s ,B):
    Ns = len(s)
    l1 = - np.floor(Ns/2).astype(int)
    l2 = (np.ceil(Ns/2) - 1).astype(int)
    x_t = np.zeros(len(t), dtype=complex)
    for l in range(l1, l2 + 1):
        x_t += s[l-l1] * np.sinc(B * t - l)
    return x_t

def demod(x_t, t, ns, B):
    l1 = - np.floor(ns/2).astype(int)
    l2 = (np.ceil(ns/2) - 1).astype(int)
    s = np.zeros(ns, dtype = complex)
    dt = t[1] - t[0]
    for l in range(l1, l2+1):
        s[l-l1] = np.sum(x_t * np.sinc(B*t - l))*dt*B
    return s

if __name__ == "__main__":
    constellation = QamConstellation(16,1)
    constellation.plot()
    bitstream = source(1024, 0.5)
    code = bit_to_symb(bitstream, constellation)