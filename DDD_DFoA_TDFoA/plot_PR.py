from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize

# 调用sklearn库，计算每个类别对应的precision和recall
data_file='C:/Users/lenovo/Desktop/seff_data/test_TDFoA_predict/test_result.csv'

# data_file='D:/博士研究方向/论文/基于深度残差收缩网络的驾驶行为检测/程序/最新训练/泛化实验/mAP_result.csv'
data=pd.read_csv(data_file)
true_label=data['0']
true_label=np.array(true_label)
# prediction_score=data[['12','13','14','15','16','17','18','19','20','21']]
prediction_score=data[['2','3']]

prediction_score=np.array(prediction_score)
#进行one-hot处理
true_label=label_binarize(true_label,classes=[i for i in range(4)])
num_class=2
precision_dict = dict()
recall_dict = dict()
average_precision_dict = dict()
for i in range(num_class):
    precision_dict[i], recall_dict[i], _ = precision_recall_curve(true_label[:, i], prediction_score[:, i])
    average_precision_dict[i] = average_precision_score(true_label[:, i], prediction_score[:, i])
    print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])
#
# micro
# precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(true_label.ravel(),
#                                                                           prediction_score.ravel())
# average_precision_dict["micro"] = average_precision_score(true_label, prediction_score, average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))
#计算macro-average
all_precision_dict = np.unique(np.concatenate([precision_dict[i] for i in range(num_class)]))
mean_recall_dict = np.zeros_like(all_precision_dict)
for i in range(num_class):
    mean_recall_dict += np.interp(all_precision_dict, precision_dict[i], recall_dict[i])
mean_recall_dict /= num_class
precision_dict["macro"]=all_precision_dict
recall_dict["macro"]=mean_recall_dict
average_precision_dict["macro"] = average_precision_score(true_label, prediction_score, average="macro")
print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["macro"]))

# 绘制所有类别平均的pr曲线
figure, ax = plt.subplots()

# plt.figure()
# plt.step(recall_dict['micro'], precision_dict['micro'], color='blue',linestyle='-',lw=2, where='post')
# plt.stackplot(recall_dict['micro'],precision_dict['micro'],colors='lightcyan')
plt.step(recall_dict["macro"], precision_dict["macro"], color='blue',linestyle='-',lw=2, where='post')
plt.stackplot(recall_dict["macro"],precision_dict["macro"],colors='lightcyan')
#设置字体大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 40,
}
plt.xlabel('Recall',font1)
plt.ylabel('Precision',font1)
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=32)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
#设置刻度的朝向
plt.rcParams['xtick.direction'] = 'out'  # in; out; inout
plt.rcParams['ytick.direction'] = 'out'
# plt.title('AP={0:0.2f}'.format(average_precision_dict["micro"]),font1)
plt.title('AP={0:0.4f}%'.format(average_precision_dict["macro"]*100),font2)

plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.show()