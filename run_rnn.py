import pandas as pd
from torch_input import run_net
import numpy as np





checkpoint ='cv/lm_lstm_epoch4.37_1.1368.t7'

def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def get_probs(row):
    print(row)
    print(row.index)
    answerprobs = []
    dumbanswerprobs = []
    lessdumbaprobs = []
    try:
        for answer in row['answerA':'answerD']:
            charprobs = []
            dumbcharprobs = []
            lessdumbcprobs = []
            primetest = ''.join((row['question'], ' '))
            primecontrol = ''
            for char in answer:
                testprob = run_net(checkpoint, primetest, char)
                controlprob = run_net(checkpoint, primecontrol, char)
                primetest += char
                primecontrol += char
                charprobs.append(testprob - controlprob)   #highest wins
                dumbcharprobs.append(testprob)
                lessdumbcprobs.append(controlprob/testprob)
            prob = np.mean(charprobs)
            answerprobs.append(prob)
            dumbanswerprobs.append(np.mean(dumbcharprobs))
            lessdumbaprobs.append(np.mean(lessdumbcprobs))
        print(row['id'])
        print(answerprobs)
        print(dumbanswerprobs)
        print(lessdumbaprobs)
        print('---------------------')
        d = {'smartprobs':answerprobs, 'dumbprobs':dumbanswerprobs,'lessdumbprobs':lessdumbaprobs}
        df = pd.DataFrame()
        df['smartprobs'] = pd.Series()
        df['smartprobs'] = df['smartprobs'].astype(object)
        df['dumbprobs'] = pd.Series()
        df['dumbprobs'] = df['dumbprobs'].astype(object)
        df['lessdumbprobs'] = pd.Series()
        df['lessdumbprobs'] = df['lessdumbprobs'].astype(object)
        df.set_value(0,'smartprobs',answerprobs)
        df.set_value(0, 'dumbprobs', dumbanswerprobs)
        df.set_value(0, 'lessdumbprobs', lessdumbaprobs)
    #    print(df)
        return answerprobs
    except
        return [0,0,0,0]

def makeselection(ls):
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




data  = pd.read_csv('../input/training_set.tsv', '\t').head(200)
#data  = pd.read_csv('../input/validation_set.tsv', '\t')

#stop = stopwords.words('english')

#data.question = data.question.apply(remove_punctuation)
#data.answerA = data.answerA.apply(remove_punctuation)
#data.answerB = data.answerB.apply(remove_punctuation)
#data.answerC = data.answerC.apply(remove_punctuation)
#data.answerD = data.answerD.apply(remove_punctuation)
data['smartprobs'] = data.apply(get_probs, axis = 1)
#data['probs'] = data.apply(get_probs, axis = 1)
#data['smartprobs'] =pd.Series( [data.probs.iloc[i][0] for i in range(len(data))])
#data['dumbprobs'] =pd.Series( [data.probs.iloc[i][1] for i in range(len(data))])
#data['lessdumbprobs']= pd.Series([data.probs.iloc[i][2] for i in range(len(data))])
#(data['smartprobs'], data['dumbprobs'], data['lessdumbprobs']) = data.apply(get_probs, axis = 1)
#data.merge(data.apply(get_probs, axis = 1), left_index=True, right_index=True)
#data = pd.concat([data, data.apply(get_probs, axis = 1)], axis =1)
#data['smartprobs'], data['dumbprobs'], data['lessdumbprobs'] = zip(data.apply(get_probs, axis = 1))
#print(data.smartprobs)
#print(data.probs)

data['smartguess'] = data.smartprobs.apply(makeselection)
#data['dumbguess'] = data.dumbprobs.apply(makeselection)
#data['lessdumbguess'] = data.lessdumbprobs.apply(makeselection)

smartscore = sum(data['smartguess'] == data['correctAnswer'])/len(data)
#dumbscore = sum(data['dumbguess'] == data['correctAnswer'])/len(data)
#lessdumbscore = sum(data['lessdumbguess'] == data['correctAnswer'])/len(data)

print('smart score:')
print(smartscore)
#print('dumb score: ')
#print(dumbscore)
#print('less dumb score: ')
#print(lessdumbscore)

data.to_csv('../output/justtryingoutRNN.csv', index = False)
