# This is the main program

import cv2
import os
import tensorflow as tf
import cv2 as cv
import re
from collections import Counter

font = cv2.FONT_HERSHEY_SIMPLEX
fileList = os.listdir("interface/signAlphbet/")
letter = [0]*26
vidCap = cv2.VideoCapture(0)
model = tf.keras.models.load_model("model/signlanguage_model_VGG16.h5")
correct = cv.imread("interface/signCorrect/check.png")
wrong = cv.imread("interface/signCorrect/cross.png")
delete = cv.imread("interface/deleteC.png")
delete = cv.resize(delete,(100,62))
space = cv.imread("interface/space.png")
space = cv.resize(space,(100,100))
correct = cv.resize(correct,(150,150))
wrong = cv.resize(wrong,(150,150))
i=0


for imageName in fileList:
    if not imageName.startswith("."):
        letter[ord(imageName[0])-65] = "interface/signAlphbet/" + imageName


def main():
    interface = cv2.imread("interface/interface_new.png")
    interface = cv2.resize(interface, (500, 800))
    cv2.imshow("interface", interface)
    cv2.setMouseCallback('interface', nextPage)

    cv2.waitKey()
    cv2.destroyAllWindows()

def prediction(roi):
    # roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    grayRoi = cv.resize(roi, (100, 100))

    input = grayRoi.reshape((1, 100, 100, 3))
    predictions = model.predict(input)
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
               "U", "V", "W", "X", "Y", "Z", "Space", "", "Delete"]
    for node_id in top_k:
        human_string = letters[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score



def nextPage (event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 30 <= x <= 496 and 240 <= y <= 470:
            #learn
            learn()
        elif 38<=x<=473 and 502<=y<=738:
            #learn
            write()
        elif 0<=x<=500 and 0<=y<=95:
            cv2.destroyWindow("inter")


def learn():
    i=0
    while i>=0 and i<len(letter):
            while True:
                name = letter[i]
                learn = cv2.imread("interface/learn.png")
                learn = cv2.resize(learn, (500, 800))
                alphbet = cv2.imread(name)
                (x, y, depth) = alphbet.shape
                learn[157:157 + x, 0:y] = alphbet
                letterNow = chr(letter.index(name)+65)

                ret, frame = vidCap.read()
                img2 = frame[:, ::-1, :]
                img2 = img2[0:290,0:500,:]
                learn[510:800,0:500] = img2

                #roi
                cv2.rectangle(learn, (350, 750), (150, 550), (255, 255, 255), 5)
                roi = learn[555:745, 155:345]
                roi = roi.astype('float32') / 255

                res,score = prediction(roi)
                print(res,score)
                if score >=0.7:
                    cv.putText(learn,res,(370,450),font, 3, (0, 0, 0), 20)
                    if res == letterNow and res!=" ":
                        learn[170:320,330:480] = correct
                    elif letterNow != res and res != " ":
                        learn[170:320, 330:480] = wrong

                cv2.imshow("inter",learn)
                x = cv2.waitKey(10)
                userChar = chr(x & 0xFF)
                if userChar == "d" or userChar == " ":
                    i +=1
                    break
                elif userChar == "a":
                    i -=1
                    break
                elif userChar =="q":
                    nextPage(cv2.EVENT_LBUTTONDOWN, 100, 90, None, None)
                    return

    nextPage(cv2.EVENT_LBUTTONDOWN, 100, 90, None, None)

def write():
    letterPre = ""
    letterNow = ""
    index = 0
    sentence = ""
    spaceIndex = 0
    letterCorrect = ""
    indexNextLine = 11
    while True:
        write = cv2.imread("interface/write.png")
        write = cv2.resize(write, (500, 800))

        ret, frame = vidCap.read()
        img2 = frame[:, ::-1, :]
        img2 = img2[0:290, 0:500, :]
        write[85:375, 0:500] = img2

        cv.rectangle(write,(150,120),(350,320),(255,255,255),5)
        roi = write[125:315,155:345]
        roi = roi.astype('float32') / 255

        res, score = prediction(roi)
        print(res,score)

        if res == "Space" and score < 0.7:
            score = 0.9
        elif res == "Delete" and score < 0.7:
            score = 0.9

        if score>0.6 and res !="Delete" and res != "Space":
            cv.putText(write, res, (410, 490), font, 3, (0, 0, 0), 20)
            letterNow = res
        elif score>0.6 and res =="Delete" and res !="Space":
            write[430:492, 385:485] = delete
            letterNow = res
        elif score>0.6 and res != "Delete" and res =="Space":
            write[410:510, 385:485] = space
            letterNow = res
        else:
            letterNow = ""

        if letterNow == letterPre :
            index += 1
        else:
            index =0

        x = cv2.waitKey(10)
        userChar = chr(x & 0xFF)
        if userChar == " ":
            write[410:510, 385:485] = space
            res = "Space"
            index = 11


        if index >=15 and res != "Delete" and res !=  "Space":
            sentence += res
            index = 0
        elif index >=10 and res != "Delete" and res =="Space":
            letterCorrect = sentence[spaceIndex:len(sentence)]
            if len(letterCorrect) != 0:
                letterCorrect = correction(letterCorrect.lower())
            sentence = sentence[0:spaceIndex]+letterCorrect.upper()
            sentence += " "
            index = 0
            spaceIndex = len(sentence)
        elif index >=10 and res == "Delete" and res !=  "Space":
            sentence = sentence[:-1]
            index = 0
            print(sentence)

        if len(sentence)  == indexNextLine:
            sentence +="\n"
            indexNextLine += 12

        # print(sentence)
        y0,dy = 450,40
        for i, line in enumerate(sentence.split('\n')):
            y = y0 + i * dy
            cv2.putText(write, line, (15, y), font, 1.3,(0,0,0),2)

        # print(sentence)

        # cv.putText(write,sentence,(15,445),font,1.3,(0,0,0),1)

        cv2.rectangle(write, (5, 410), (350, 760), (255, 255, 255), 5)
        cv2.rectangle(write, (375, 410), (495, 530), (255, 255, 255), 5)

        cv2.imshow("inter", write)
        letterPre = letterNow
        x = cv2.waitKey(10)
        userChar = chr(x & 0xFF)
        if userChar == "q":
            nextPage(cv2.EVENT_LBUTTONDOWN, 100, 90, None, None)
            return


    nextPage(cv2.EVENT_LBUTTONDOWN,100,90,None,None)



def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('autoCorrect/big.txt').read()))

def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N

def correction(word):
    return max(candidates(word), key=P)

def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

main()