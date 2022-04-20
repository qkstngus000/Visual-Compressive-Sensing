import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_c = '../models/checkpoints_classical/training_log.csv'
data_s = '../models/checkpoints_structured/training_log_run3.csv'

df_c = pd.read_csv(data_c)
df_s = pd.read_csv(data_s)

plt.ion()
plt.style.use('my.mplstyle')
fig, ax = plt.subplots(figsize=(8,6))
df_c.plot('epoch', ['train_loss', 'val_loss'], ax=ax, logy=False)
df_s.plot('epoch', ['train_loss', 'val_loss'], ax=ax, logy=False)
ax.legend(['train: classical', 'test: classical',
           'train: structured', 'test: structured'])
plt.xlabel('epochs')
plt.ylabel('loss')

plt.xlim([0.8,10])
ax.set_xticks(np.arange(1,11))
plt.ylim([2,7])
plt.tight_layout()
plt.savefig('learning_dnn.png')

# plt.xlim([80,90])
# plt.tight_layout()
# plt.savefig('learning_dnn_2.png')
