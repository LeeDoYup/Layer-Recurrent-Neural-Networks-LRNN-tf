import numpy as np 
import pdb

def confusion_matrix(test_out, y_test):

	test_size, class_size = np.shape(y_test)

	confusion_matrix_result = np.zeros([class_size,class_size])

	y_true_index = np.argmax(y_test,axis=1)
	y_out_index = np.argmax(test_out,axis=1)


	for i in range(test_size):
		ground_truth = y_true_index[i]
		model_out = y_out_index[i]
		confusion_matrix_result[ground_truth,model_out] += 1.0

	confusion_matrix_result /= test_size

	return confusion_matrix_result

def classification_accuracy(test_out, y_test):
	one_hot_result = np.argmax(test_out,axis=1)
	one_hot_true = np.argmax(y_test, axis=1)
	test_size = len(y_test)
	temp_count = 0
	for i in range(test_size):
		if one_hot_result[i] == one_hot_true[i]:
			temp_count +=1

	accuracy = float(temp_count)/test_size
	print accuracy
	return accuracy


def top3_classification_accuracy(test_out, y_test):
	x = np.argsort(test_out, axis=1)
	one_hot_true = np.argmax(y_test, axis=1)
	test_size = len(y_test)
	temp_count = 0
    
	for i in range(test_size):
		if x[i, one_hot_true[i]] > 6:
			temp_count +=1

	accuracy = float(temp_count)/test_size
	print accuracy
	return accuracy

def MCC(test_out, y_test):
	#MCC is a performance measure of binary classification
	TP, TN, FP, FN = 0,0,0,0
	for i in range(len(test_out)):
		if test_out[i] == 1 and y_test[i] ==1:
			TP +=1
		elif test_out[i] == 0 and y_test[i] ==0:
			TN +=1
		elif test_out[i] == 1 and y_test[i] ==0:
			FP +=1
		else:
			FN +=1
	return (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

