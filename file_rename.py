import os
from os import listdir
from os.path import isfile, join

oseba = 'Veno'
newName = oseba+'_00'
directory = 'C:\\Users\\venog\\Desktop\\Diploma\\facenet\\data\\images\\train_raw\\'+oseba

files = [f for f in listdir(directory) if isfile(join(directory, f))]

# print(files)


i = 0
for file in files:
    if i <= 9:
        newName = oseba+'_000'+str(i)
    elif i <= 99:
        newName = oseba+'_00'+str(i)
    else:
        newName = oseba+'_0'+str(i)

    print(newName)
    i += 1

    before =''+directory+'\\'+file
    after = ''+directory+'\\'+newName

    if before==after:
        continue

    os.rename(r''+directory+'\\'+file, r''+directory+'\\'+newName+'.JPG')

# print(files)