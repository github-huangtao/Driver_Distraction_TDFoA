import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()
data_file='C:/Users/lenovo/Desktop/seff_data/test_TDFoA_predict/test_result.csv'
data=pd.read_csv(data_file)
true_label=data['0']
prediction=data['1']
fig=plt.figure()
ax1=plt.subplot()


con_mat = confusion_matrix(list(true_label), list(prediction))
# con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=0)[:, np.newaxis]     # 归一化
# con_mat_norm=con_mat_norm*100
# con_mat_norm=con_mat
# con_mat_norm1 = np.around(con_mat_norm, decimals=3)
# print(con_mat_norm)
sns.heatmap(con_mat,ax=ax1, annot=True, cmap='Oranges',annot_kws={'size':32, 'color':'black'},fmt ='.0f')

cbar = ax1.collections[0].colorbar
cbar.ax.tick_params(labelsize=32)

plt.tick_params(labelsize=32)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Palatino Linotype') for label in labels]
#设置刻度的朝向
plt.rcParams['xtick.direction'] = 'out'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'

plt.ylim(0,2)
font1 = {'family' : 'Palatino Linotype',
'weight' : 'normal',
'size'   : 40,
}
plt.xlabel('Predicted Driver behavior',font1)
plt.ylabel('True Driver behavior',font1)
plt.rcParams['font.sans-serif'] = 'Palatino Linotype'

plt.show()