from sklearn import svm
import numpy as np
import cPickle as pickle

fullTrainData = np.load('train_data1_orb.npy')
fullTrainLabels = np.load('train_labels1_orb.npy')

def leave_one_out_validate(fullTrainData, fullTrainLabels):
	accuracy = np.zeros(fullTrainLabels.shape[0])
	for i in range(fullTrainLabels.shape[0]):
		testData = fullTrainData[i]
		testLabels = fullTrainLabels[i]
		trainData = np.append(fullTrainData[:i], fullTrainData[i+1:], axis=0)
		trainLabels = np.append(fullTrainLabels[:i], fullTrainLabels[i+1:])

		#clf = svm.SVC(decision_function_shape='ovr')
		#clf.fit(trainData, trainLabels)
		lin_clf = svm.LinearSVC()
		lin_clf.fit(trainData, trainLabels)

		#predictions = clf.predict(testData)
		prediction = lin_clf.predict(testData.reshape(1,-1))
		score = lin_clf.decision_function(testData.reshape(1,-1))
		if prediction != testLabels:
			accuracy[i] = 0
		else:
			accuracy[i] = 1
			print score, prediction
	return np.mean(accuracy)
#print errors, accuracy
#print predictions
#print testLabels

#newTrainLabels = np.zeros(trainLabels.shape[0], "uint32")
#for i in range(trainLabels.shape[0]):
#	newTrainLabels[i] = trainLabels[i][0]
#trainLabels = newTrainLabels
#trainData = fullTrainData
#trainLabels = fullTrainLabels
#testData = fullTrainData
#testLabels = fullTrainLabels
#trainData = np.append(np.append(fullTrainData[:29], fullTrainData[30:59], axis=0), fullTrainData[60:89], axis=0)
#trainLabels = np.append(np.append(fullTrainLabels[:29], fullTrainLabels[30:59]), fullTrainLabels[60:89])
#testData = np.append(np.append(fullTrainData[29:30], fullTrainData[59:60], axis=0), fullTrainData[89:90], axis=0)
#testLabels = np.append(np.append(fullTrainLabels[29:30], fullTrainLabels[59:60]), fullTrainLabels[89:90])

#print trainData.shape, trainLabels.shape
#clf = svm.SVC(decision_function_shape='ovr')
#clf.fit(trainData, trainLabels)
#lin_clf = svm.LinearSVC()
#lin_clf.fit(trainData, trainLabels)

#predictions = clf.predict(testData)
#predictions = lin_clf.predict(testData)
#errors = 0
#for i in range(testLabels.shape[0]):
#	if predictions[i] != testLabels[i]:
#		errors += 1

#accuracy = 1 - float(errors)/testLabels.shape[0]
accuracy = leave_one_out_validate(fullTrainData, fullTrainLabels)
print accuracy

lin_clf = svm.LinearSVC()
lin_clf.fit(fullTrainData, fullTrainLabels)
with open('svm_lin_clf.pkl', 'wb') as output:
    pickle.dump(lin_clf, output, pickle.HIGHEST_PROTOCOL)