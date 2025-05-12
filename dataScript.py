import os
import cv2
import re
import numpy as np

#key = int(input('enter key 1 - delete .cat файлы, 2 - change shape of picture to (256x170x3)\n'))

key = 2


def absPath(s):
    base = './data/cats/'
    fStr = os.path.abspath(base)
    return os.path.join(fStr, s)

def relPath(s):
    return r'.//data//cats//' + s

match key:
    case 1:
        delete = [i for i in os.listdir('./data/cats') if re.search(r'\.cat$', i)]
        delete = list(map(absPath, delete))
        for file in delete:
            os.remove(file)
        remaining = [i for i in os.listdir('./data/cats') if re.search(r'\.cat$', i)]
        print(f"\nLeft .cat: {remaining}")

    case 2:
        files = os.listdir('./data/cats')
        filesPath = list(map(relPath, files))
        c = 0
        for i in filesPath:
            pic = cv2.imread(i)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            pic = cv2.resize(pic, (256, 170), interpolation=cv2.INTER_AREA)
            cv2.imwrite(i, pic)
            c += 1
            if c % 100 == 0:
                print(c)

    case _:
        print("Put the straitjacket back on")
