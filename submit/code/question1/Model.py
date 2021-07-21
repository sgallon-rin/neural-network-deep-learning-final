# training and test model
import numpy as np
import pickle
import tensorflow.compat.v1 as tf
import sklearn
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import tree
from Feature_model import Tight_frame_classifier

Tight_frame_model = pickle.load(open('data_parts/model.pkl','rb')) # pre_training model
Featur_index = Tight_frame_model.best_feature()

Train_set = pickle.load(open('data_parts/Train_set.pkl','rb'))
Test_set = pickle.load(open('data_parts/Test_set.pkl','rb'))
Train_label = pickle.load(open('data_parts/Train_label.pkl','rb'))
Test_label = pickle.load(open('data_parts/Test_label.pkl','rb'))
Distrit = pickle.load(open('data_parts/tight_frame_D.p','rb'))

Train = Train_set[:,Featur_index]
Test = Test_set[:,Featur_index]
Distr = Distrit[:,Featur_index]

## Training model
clf = SVC(C=1000, kernel='rbf', gamma=10, decision_function_shape='ovo')
knn = neighbors.KNeighborsClassifier()
DT = tree.DecisionTreeClassifier()

clf.fit(Train,Train_label)
knn.fit(Train,Train_label)
DT.fit(Train,Train_label)

# Test model
# predict = clf.predict(Test)
predict = knn.predict(Distr)
# predict = DT.predict(Test)
# predict = Tight_frame_model.predict(Test_set)
print(predict)

# #  Define neural network
# tf.disable_v2_behavior()
# configuration = tf.ConfigProto()
# configuration.gpu_options.allow_growth = True
# session = tf.Session(config=configuration)
# Step = 1e-4
# epoch = 100000
# layer1 = 7
# keep_drop = 0.5 # drop out probility
# # transfer train label to 2-dimension
# Label_tempt = np.zeros([len(Train_label),2])
# for i in range(len(Train_label)):
#     Label_tempt[i][int(Train_label[i])] = 1.0

# X = tf.placeholder('float',[None,len(Featur_index)])
# Y = tf.placeholder('float',[None,2])
# Wh = tf.Variable(0.1*tf.random_normal([len(Featur_index),layer1]))
# bh = tf.Variable(0.1*tf.random_normal([layer1]))
# h1 = tf.add(tf.matmul(X,Wh),bh) # hidden layer
# drop_out = tf.nn.dropout(h1,keep_drop) # drop out layer
# W_out = tf.Variable(0.1*tf.random_normal([layer1,2]))
# b_out = tf.Variable(0.1*tf.random_normal([2]))
# out = tf.nn.softmax(tf.add(tf.matmul(drop_out,W_out),b_out)) # prediction layer
# #out = tf.nn.softmax(tf.add(tf.matmul(h1,W_out),b_out)) # prediction layer
# loss = tf.reduce_sum(tf.square(Y - out)) # loos function
# train_step1 = tf.train.GradientDescentOptimizer(Step).minimize(loss)

# # Train neural network
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# loss_pre = 40000 # record latest loss for step-size change
# Count = 0 # continue not decrease times
# for i in range(epoch):
#     # 步长更新
#     if(i%50 == 0): # check loss per 50 epochs
#         loos_tempt = sess.run(loss, feed_dict={X: Train, Y: Label_tempt})
#         if (loss_pre <= loos_tempt):
#             if(Count < 4):
#                 Count += 1
#             else: # the value not decrease for continuous 4 times
#                 Step /= 1.5
#                 Count = 0
#                 loos_pre = loos_tempt
#         else:
#             loos_pre = loos_tempt
#             Count = 0
#     else:
#         loss_pre = loos_tempt

#     if(i%1000 == 0):
#         print('proceeding: %.1f%%'%(i*100/epoch))
#         print('loos: ',loos_tempt )
#         print('step size %.5f:'%Step)
#     sess.run(train_step1,feed_dict={X:Train, Y:Label_tempt})

# # use neural network to do prediction
# predict_tempt= sess.run(out,feed_dict={X:Test})
# sess.close()
# predict = []
# for i in range(len(Test_label)):
#     predict += [(predict_tempt[i,1] > predict_tempt[i,0])]

# precision = sklearn.metrics.precision_score(Test_label, predict)
# recall = sklearn.metrics.recall_score(Test_label, predict)
# accuracy = sklearn.metrics.accuracy_score(Test_label, predict)
# TP = 0; TN = 0
# for i in range(len(Test_label)):
#     if(predict[i] == Test_label[i] == 1):
#         TP += 1
#     elif(predict[i] == Test_label[i] == 0):
#         TN += 1
# TPR = TP / sum(Test_label)
# TNR = TN / (len(Test_label)-sum(Test_label))

# print('\naccuracy: %.2f%%' % (100 * accuracy))
# print('TPR: %.2f%%\nTNR: %.2f%%' % (100 * TPR, 100 * TNR))
# print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))
