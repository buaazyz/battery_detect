import os
import random
file_name = 'train.txt'

with open(file_name,'r') as f:
    train_list = f.readlines()
    f.close()
random.shuffle(train_list)

x = len(train_list)

with open('train_random.txt','w',newline='') as f:
    for i in range(x):
        # for j in range(x*i, x*(i+1)):
        f.write(train_list[i])
f.close()
print('train_random write finished')
