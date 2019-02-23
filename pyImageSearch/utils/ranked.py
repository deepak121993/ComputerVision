import numpy as np

def rank5_accuracy(preds,labels):
    rank1 =0
    rank5=0

    for (p,gt) in zip(pred,labels):
        #sort the probablities in decreasing order
        p = np.argsort(p)[::-1]
        ##check the top 5 predictions

        if gt in p[:5]:
            rank5 +=1

        if gt ==p[0]:
            rank1 += 1
        
        rank1 /=float(len(labels))
        rank5 /=float(len(labels))

        return (rank1,rank5)
        