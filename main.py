import warnings
from RUN import *
from Utils import utilities as ut

def Run(Xsource, Ysource,source_w, Xtarget, Ytarget):
    AUC=[]
    F=[]



    for i in range(0,20):
            model = RUN(Xsource, Ysource,source_w, Xtarget, Ytarget)
            model.fit()
            model.predict()
            AUC.append(model.AUC)
            F.append(model.F)


    print(np.mean(AUC),np.mean(F))
    print('======================================================================================')



if __name__ == '__main__':

    warnings.filterwarnings('ignore')


    datasets=sorted(["EQ", "JDT", "LC", "ML", "PDE"])
    for ds in datasets:
        Xtarget, Ltarget = ut.read_dataset("data/AEEEM/", dataset_name=ds) # data/AEEEM/ , data/PROMISE/ ,or data/SOFTLAB/
        Ltarget[Ltarget > 1] = 1
        Xtarget=Xtarget
        Xsource=np.zeros((1,Xtarget.shape[1]))
        Lsource=[0]
        source_w=[0]
        sn=0


        Xt_mean = Xtarget.mean(axis=0)

        dists=[]
        Xs_mean=0
        for dss in datasets:
            if dss!=ds:
                sn+=1
                X, L = ut.read_dataset("data/AEEEM/", dataset_name=dss)
                # snv=np.ones((L.shape[0]))*sn
                Xs_mean += X.mean(axis=0)

                Xsource= np.concatenate((Xsource, X), axis=0)

                Lsource= np.concatenate((Lsource, L), axis=0)




        Xsource=Xsource[1:,:]
        Lsource=Lsource[1:]
        source_w=source_w[1:]

        Lsource[Lsource > 1] = 1

        print(ds + 'Start!')

        Run(Xsource, Lsource,source_w, Xtarget, Ltarget)


    print('done!')




