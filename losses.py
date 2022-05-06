def diceLoss(pred,target,smooth =1e-3):
    intersection = (pred *target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    dice = 2*(intersection)/(union+smooth)
    diceloss = 1-dice
    return diceloss.sum(),dice.sum()

