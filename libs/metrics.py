def metric(pred, targ):
    TPs = 0
    FPs = 0
    TNs = 0
    FNs = 0

    for p, t in zip(pred, targ):
        if p == 1 and t == 1:
            TPs += 1
        elif p == 0 and t == 0:
            TNs += 1
        elif p == 0 and t == 1:
            FPs += 1
        else:
            FNs += 1

    return TPs, TNs, FPs, FNs


def get_metrics(TPs, TNs, FPs, FNs):
    accuracy = (TPs + TNs) / (TPs + TNs + FPs + FNs)
    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
    f1score = precision * recall / (precision + recall + 1e-16)
    return accuracy, precision, recall, f1score
