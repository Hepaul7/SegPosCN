import pandas as pd
import matplotlib.pyplot as plt

ctb7_data = pd.read_csv('model_metrics.csv')
ctb9_data = pd.read_csv('model_metrics9')


max_batch_ctb7 = ctb7_data['Batch'].max()
max_batch_ctb9 = ctb9_data['Batch'].max()


ctb7_data['Step'] = (ctb7_data['Epoch'] - 1) * max_batch_ctb7 + ctb7_data['Batch']
ctb9_data['Step'] = (ctb9_data['Epoch'] - 1) * max_batch_ctb9 + ctb9_data['Batch']

ctb7_data = ctb7_data[ctb7_data['Step'] <= 500]
ctb9_data = ctb9_data[ctb9_data['Step'] <= 500]

plt.figure(figsize=(10, 6))
plt.plot(ctb7_data['Step'], ctb7_data['Loss'], label='CTB7 Loss')
plt.plot(ctb9_data['Step'], ctb9_data['Loss'], label='CTB9 Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs. Step for CTB7 and CTB9')
plt.legend()
plt.show()
