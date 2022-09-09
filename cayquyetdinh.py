from sklearn import tree

#thu thập dữ liệu
#xử lí dữ liệu
mytree = tree.DecisionTreeClassifier()
dactrung = [
    [1,3,3,7],
    [5,2,4,6],
    [1,2,4,6],
    [5,4,4,3],
    [1,4,4,7],
    [3,2,3,7],
    [3,3,3,6],
    [5,2,2,7]
]
nhan = [0,1,1,0,0,0,0,1]

#xây dựng model
result = mytree.fit(dactrung, nhan)

#dự đoán kết quả
x = result.predict([[1,4,3,6],[1,4,3,7]])
print(x)
#đánh giá model có hiệu quả hay không



