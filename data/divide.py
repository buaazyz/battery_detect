import random


train = list(range(5500))
testid = random.sample(train,1100)

# print(test)
for i in testid:
    train.remove(i)


# print(len(total))

trainitem = []
testitem =[]
with open('total.txt',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        item = []
        # print()
        if int(line[0]) in testid:
            # print(line[0])
            item.append(line[0])
            item.append(line[1])
            testitem.append(item)
        else:
            item.append(line[0])
            item.append(line[1])
            trainitem.append(item)


with open('test.txt','w',encoding='utf-8') as f:
    for it in testitem:
        f.write(it[0] + ' ')
        f.write(it[1])
        f.write('\n')

with open('train.txt','w',encoding='utf-8') as f:
    for it in trainitem:
        f.write(it[0] + ' ')
        f.write(it[1])
        f.write('\n')