import os
import random
file_name = 'train.txt'

with open(file_name,'r') as f:
    train_list = f.readlines()
    f.close()
random.shuffle(train_list)

x = len(train_list) // 4
for i in range(4):
    with open('train_random_'+str(i+1)+'.txt','w',newline='') as f:
        for j in range(x*i, x*(i+1)):
            f.write(train_list[j])
    f.close()
    print('train_random_'+str(i+1)+' write finished')
