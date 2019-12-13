import numpy as np
a = [1,2,3,4,5]
b = []
for ind in a:
	b.append(ind)

myList = ['Arise', 'But', 'It', 'Juliet', 'Who', 'already', 'and', 'and', 'and', 
     'breaks', 'east', 'envious', 'fair', 'grief', 'is', 'is', 'is', 'kill', 'light', 
     'moon', 'pale', 'sick', 'soft', 'sun', 'sun', 'the', 'the', 'the', 
     'through', 'what', 'window', 'with', 'yonder']

auxiliaryList = []
for word in myList:
    if word not in auxiliaryList:
        auxiliaryList.append(word)
print (auxiliaryList)
# al = np.asarray(auxiliaryList)
# print(np.shape(al))
# al.reshape(-1,3)
# print(np.shape(al))

ab = np.array([1,2,3])
bc = np.array([2,3,4])
cd = np.array([4,5,6])
de = np.array([5,6,7])
# new_Arr1=np.stack((ab,bc), axis=0)
# new_Arr2=np.stack((cd,de), axis=0)
# print(new_Arr1)
# print(new_Arr2)
# new_Arr3 = new_Arr1.vstack(new_Arr2)
# print(new_Arr3)	

ini_array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]) 
  
# printing initial arrays 
print("initial array", str(ini_array1)) 
  
# Multiplying arrays 
result = ini_array1.flatten() 
  
# printing result 
print("New resulting array: ", result) 