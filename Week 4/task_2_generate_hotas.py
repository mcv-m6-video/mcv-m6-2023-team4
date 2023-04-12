import matplotlib.pyplot as plt

cams = ['c015', 'c014', 'c011', 'c013', 'c010', 'c012', 'c001', 'c002', 'c003', 'c004', 'c005', 'c029', 'c021', 'c032', 'c035', 'c018', 'c030', 'c017', 'c024', 'c020', 'c034', 'c027', 'c025', 'c028', 'c019', 'c023', 'c031', 'c038', 'c022', 'c036', 'c016', 'c033', 'c040', 'c026', 'c037', 'c039']
hotas = [6.351, 32.09, 26.16, 37.15, 32.74, 22.96, 27.78, 31.68, 28.28, 32.55, 14.16, 32.09, 37.92, 18.36, 36.8, 33.53, 29.7, 40.28, 35.79, 22.47, 26.69, 21.72, 26.46, 33.47, 61.88, 23.59, 19.22, 28.4, 36.08, 31.82, 31.54, 37.42, 24.96, 30.14, 42.27, 33.26]
idfs = [2.335, 44.15, 37.82, 56.06, 47.62, 24.75, 42.11, 43.4, 33.14, 46.36, 10.79, 40.13, 58.3, 27.21, 57.89, 55.83, 40.21, 62.31, 53.14, 43.22, 35.03, 25.52, 42.39, 48.0, 87.13, 42.36, 19.34, 47.24, 48.73, 43.93, 37.24, 59.84, 38.8, 43.09, 57.88, 51.0]


for i in range(len(cams)):
    if cams[i] in ['c016', 'c017', 'c018', 'c019', 'c020', 'c021']:
        print(f'{cams[i]}-{hotas[i]}-{idfs[i]}')

# plt.title('Performance of our tracking algorithm on the cameras of sequence 03')
# plt.plot(list(range(5)), list(reversed(hotas[:5])), label='hota')
# plt.plot(list(range(5)), list(reversed(idfs[:5])), label='idf')
# plt.xticks(list(range(5)), list(reversed(cams[:5])))
# plt.xlabel('Cameras')
# plt.ylabel('Performance')
# plt.legend()
# plt.show()