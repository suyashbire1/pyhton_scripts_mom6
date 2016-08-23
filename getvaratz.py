import numpy as np
def getvaratz(var,z,e):
    e = np.swapaxes(e,0,1).reshape((e.shape[1],-1))
    var1 = np.swapaxes(var,0,1).reshape((var.shape[1],-1))
    varatz = np.ma.zeros((z.size,var1.shape[1]))
    varatz[:] = np.NaN

    for i in range(z.size):
        t1 = e[0:-1,:] - z[i]
        t2 = e[1:,:] - z[i]
        tt = t1*t2
        indices = np.nonzero(tt<=0)
        varatz[i,indices[1]] = var1[indices[0],indices[1]]

    varatz = varatz.reshape((z.size,var.shape[0],var.shape[2],var.shape[3])).swapaxes(0,1)
    return varatz

def getTatz(zl,z,e):
    e = np.swapaxes(e,0,1).reshape((e.shape[1],-1))
    Tatz = np.zeros((z.size,e.shape[1]))
    Tatz[:] = np.NaN

    for i in range(z.size):
        t1 = e[0:-1,:] - z[i]
        t2 = e[1:,:] - z[i]
        tt = t1*t2
        indices = np.nonzero(tt<=0)
        Tatz[i,indices[1]] = zl[indices[0]]

    Tatz = Tatz.reshape((z.size,e.shape[0],e.shape[2],e.shape[3])).swapaxes(0,1)
    return Tatz
