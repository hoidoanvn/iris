from sklearn.datasets import load_iris
# hỗ trợ chia data set thành 75% train và 25% test
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


iris_dataset = load_iris()
# iris_dataset.data # xem data dưới dựng nào (ma trận 2 chiều)
# iris_dataset.target # xem nó được dán nhãn nào (từ 0-2)
# len(dataset.target) # xem độ dài dữ liệu (số lượng hoa)

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset.data, iris_dataset.target, random_state=1)  # để là 0 thì giống nhau, 1 thì random
# X là dataset
# y là nhãn dán tương ứng

model = DecisionTreeClassifier()  # xây dựng model

mymodel = model.fit(X_train, y_train)  # train model

print(mymodel.predict(X_test))  # dự đoán

# dự đoán một data mới thuộc nhãn nào
#import numpy as np
#X_New = np.array([[6.2 ,2.2 ,5.1, 1.9]])
# print(mymodel.predict(X_New))

print(mymodel.score(X_test, y_test))  # kiếm tra độ chính xác
