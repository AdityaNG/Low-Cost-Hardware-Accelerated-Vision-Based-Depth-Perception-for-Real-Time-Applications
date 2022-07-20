import matplotlib.pyplot as plt

headers = input()

data = {
    'SCALE_FACTOR': [],
    'CPU': [],
    'OMP': [],
    'CUDA': []
}
data_s = {
    'SCALE_FACTOR': [],
    'CPU': [],
    'OMP': [],
    'CUDA': []
}
while True:
    try:
        SCALE_FACTOR,SUBSAMPLING_MODE,CPU,OMP,CUDA = list(map(float, input().split(',')))
        d = data
        if SUBSAMPLING_MODE==1:
            d = data_s
        d['SCALE_FACTOR'].append(SCALE_FACTOR)
        d['CPU'].append(CPU)
        d['OMP'].append(OMP)
        d['CUDA'].append(CUDA)
    except:
        break

print(data)
print(data_s)

plt.plot(data['SCALE_FACTOR'], data['CPU'], label='CPU')
plt.plot(data['SCALE_FACTOR'], data['OMP'], label='OMP')
plt.plot(data['SCALE_FACTOR'], data['CUDA'], label='CUDA')

plt.plot(data_s['SCALE_FACTOR'], data_s['CPU'], label='CPU_S')
plt.plot(data_s['SCALE_FACTOR'], data_s['OMP'], label='OMP_S')
plt.plot(data_s['SCALE_FACTOR'], data_s['CUDA'], label='CUDA_S')

plt.legend()

plt.show()
