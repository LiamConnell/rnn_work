import pandas as pd
from torch_input import run_net
import numpy as np





checkpoint ='cv/lm_lstm_epoch0.82_1.3566.t7'

def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def get_probs(row):
    answerprobs = []
    dumbanswerprobs = []
    for answer in row['answerA':'answerD']:
        charprobs = []
        dumbcharprobs = []
        primetest = ''.join((row['question'], ' '))
        primecontrol = ''
        for char in answer:
            testprob = run_net(checkpoint, primetest, char)
            controlprob = run_net(checkpoint, primecontrol, char)
            primetest += char
            primecontrol += char
            charprobs.append(testprob - controlprob)   #highest wins
            dumbcharprobs.append(testprob)
        prob = np.mean(charprobs)
        answerprobs.append(prob)
        dumbanswerprobs.append(np.mean(dumbcharprobs))
    print(row['id'])
    return answerprobs, dumbanswerprobs

def make_selection(ls):
    mx = max(ls)
    best = [i for i, j in enumerate(ls) if j == mx]
    if len(best) > 1:
        best = best[0]

    if best == [0]:
        return 'A'
    if best == [1]:
        return 'B'
    if best == [2]:
        return 'C'
    if best == [3]:
        return 'D'
    else:
        print('returning C becuase no max')
        return 'C'




data  = pd.read_csv('../input/training_set.tsv', '\t').tail(5)
#data  = pd.read_csv('../input/validation_set.tsv', '\t')

#stop = stopwords.words('english')

#data.question = data.question.apply(remove_punctuation)
#data.answerA = data.answerA.apply(remove_punctuation)
#data.answerB = data.answerB.apply(remove_punctuation)
#data.answerC = data.answerC.apply(remove_punctuation)
#data.answerD = data.answerD.apply(remove_punctuation)

data['smartprobs'], data['dumbprobs'] = data.apply(get_probs, axis = 1)


data['smartguess'] = data.smartprobs.apply(makeselection)
data['dumbguess'] = data.dumbprobs.apply(makeselection)

smartscore = sum(data['smartguess'] == data['correctAnswer'])/len(data)
dumbscore = sum(data['dumbguess'] == data['correctAnswer'])/len(data)

print('smart score:')
print(smartscore)
print('dumb score: ')
print(dumbscore)

data.to_csv('../output/justtryingoutRNN.csv', index = False)
