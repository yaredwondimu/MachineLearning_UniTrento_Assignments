import string
import math
from random import randrange

listfile=[]
infile=open("heart.dat","r")
textfile=infile.readlines()
infile.close()
for linfo in textfile:
    clist=[]
    linfo=linfo.rstrip('\n')
    for data in linfo.split(","):
        clist.append(data)
    listfile.append(clist)

for ld in listfile:
    print ld
listclass1=[];
listclass2=[];

for data in listfile:
    if data[0]=='1':
        listclass1.append(data)
    else:
        listclass2.append(data)
listclass2=listclass2[1:]

#selecting randomly 21 example for train dataset(trainlist) from class2 (listclasse2) 
trainlist=[]
i=1
while i<=21:
    random_index = randrange(0,len(listclass2))
    trainlist.append(listclass2[random_index])
    listclass2.pop(random_index)
    i=i+1
    

#selecting randomly 21 example for test dataset(testlist) from class2 (listclasse2) 

testlist=[]
i=1
while i<=21:
    random_index = randrange(0,len(listclass2))
    #print foo[random_index]
    testlist.append(listclass2[random_index])
    listclass2.pop(random_index)
    i=i+1
    

#selecting randomly 79 example for train dataset(trainlist) from class1 (listclasse1) 

i=1
while i<=79:
    random_index = randrange(0,len(listclass1))
    #print foo[random_index]
    trainlist.append(listclass1[random_index])
    listclass1.pop(random_index)
    i=i+1
    

#selecting randomly 79 example for test dataset(testlist) from class1 (listclasse1) 

i=1
while i<=79:
    random_index = randrange(0,len(listclass1))
    #print foo[random_index]
    testlist.append(listclass1[random_index])
    listclass1.pop(random_index)
    i=i+1
    

filewrite = open("train.dat", "w")
filewrite.write("y,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22"+'\n')
for td in trainlist:
    filewrite.write(td[0]+','+td[1]+','+td[2]+','+td[3]+','+td[4]+','+td[5]+','+td[6]+','+td[7]+','+td[8]+','+td[9]+','+td[10]+','+td[11]+','+td[12]+','+td[13]+','+td[14]+','+td[15]+','+td[16]+','+td[17]+','+td[18]+','+td[19]+','+td[20]+','+td[21]+','+td[22]+'\n')
filewrite.close()

#for creating testlist file

filewrite = open("test.dat", "w")
filewrite.write("y,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22"+'\n')
for td in testlist:
    filewrite.write(td[0]+','+td[1]+','+td[2]+','+td[3]+','+td[4]+','+td[5]+','+td[6]+','+td[7]+','+td[8]+','+td[9]+','+td[10]+','+td[11]+','+td[12]+','+td[13]+','+td[14]+','+td[15]+','+td[16]+','+td[17]+','+td[18]+','+td[19]+','+td[20]+','+td[21]+','+td[22]+'\n')
filewrite.close()


    
