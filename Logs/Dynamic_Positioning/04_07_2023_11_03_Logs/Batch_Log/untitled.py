import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted, ns

file_list = glob.glob('*.npy')

file_list = natsorted(file_list, key=lambda y: y.lower())
print(file_list)

print(int(file_list[95][11:-9]))

loss_log = []

#0000_batch_-6809_loss.npy
for i in file_list:
    loss_log.append(int(i[11:-9]))

plt.figure(figsize=(10, 4))
plt.title('Loss history')
plt.plot(loss_log, 'r-.', alpha=0.5)
plt.savefig('loss_plot_test.pdf')
