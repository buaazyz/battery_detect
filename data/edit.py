import os
# file_name = 'train'
file_name = 'test'
with open(file_name +'.txt','r') as f:
    list_old = f.readlines()
    f.close()
list_new = []
for item in list_old:
    tmp = item.split('_')
    tmp = tmp[1]
    tmp = tmp.split('.')
    tmp = tmp[0] + '\n'
    list_new.append(tmp)

with open(file_name + '_new.txt','w',newline='') as f:
    for name in list_new:
        f.write(name)
print('finish')