import pandas as pd
import numpy as  np
import keras
from keras.models import load_model

#生成寻训练数据的形式,分心类别为0，不分心类别为1
def generate_data(distraction_path,non_distraction_path):
    data_x = []
    data_y = []
    distraction_data = pd.read_csv(distraction_path, header=None)
    non_distraction_data = pd.read_csv(non_distraction_path, header=None)
    for i in range(len(distraction_data)):
        data_x.append(distraction_data.iloc[i, :])
        data_y.append(0)
    for i in range(len(non_distraction_data)):
        data_x.append(non_distraction_data.iloc[i, :])
        data_y.append(1)
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    # data_y = keras.utils.to_categorical(data_y, 2)

    return data_x,data_y

weight_path='D:/博士研究方向/论文/基于驾驶员注意力区域的驾驶员分心/our/our_DDD/DDD_model/models/our_DDD.h5'
model=load_model(weight_path)

test_distraction_path = 'C:/Users/lenovo/Desktop/seff_data/test/distraction.csv'
test_non_distraction_path = 'C:/Users/lenovo/Desktop/seff_data/test/non-distraction.csv'
test_data_x, test_data_y = generate_data(test_distraction_path, test_non_distraction_path)
prediction=[]
prediction_score=[]

for i in range(len(test_data_x)):
    x=test_data_x[i]
    x=np.array(x)
    x=x.flatten()
    x=x.reshape(1,-1)
    pred = model.predict(x)
    class_idx = np.argmax(pred[0])
    prediction.append(class_idx)
    prediction_score.append(pred)
n=len(prediction)
data=np.zeros((n,4))
data[:,0]=test_data_y
data[:,1]=prediction
data[:,2:4]=prediction_score
csv_filename='D:/博士研究方向/论文/基于驾驶员注意力区域的驾驶员分心/our/our_DDD/DDD_model/test_result.csv'
data=pd.DataFrame(data)
data.to_csv(csv_filename)