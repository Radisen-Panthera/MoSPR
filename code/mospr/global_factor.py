import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import pathlib, sys
import h5py, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT=_p.ROOT
sys.path.insert(0,str(_p.CODE))
import cohorts
from spatial_proteome import pathways
CPOUT=_p.CPNN_REPO/"outputs"; RES=_p.RESULTS; DS=_p.DATA
GMT=_p.GENESETS/"h.all.v2025.1.Hs.symbols.gmt"
M=[("Max","AbMIL_ilse_max%ts",0),("Mean","AbMIL_ilse_mean%ts",0),("AbMIL","AbMIL%ts",0),
   ("HE2RNA","HE2RNA%ComparisonTrainerts",0),("AbReg","AbRegMIL%ts",0),
   ("tRNAformer","tRNAsformer%ComparisonTrainerts",0),("ILRA","ILRA%ts",0),
   ("S4MIL","S4Model_stop_sampling%ts",0),("MambaMIL","MambaMILvanira_stop_sampling%ts",0),
   ("SRMambaMIL","SRMambaMIL_stop_sampling%ts",0),("MOSBY","SumExpModel_MOSBY%ts",0),
   ("SEQUOIA VIS","SEQUOIA_VIS%ComparisonTrainerts",0),
   ("2DMamba","MambaMIL_2D_stop_sampling%Mamba2DTrainerts",0),
   ("CPNN","ProtoSum_1reg_mse_reg_1e3%DeconvExptsfine",1)]

def logcpm(X):
    X=np.clip(np.asarray(X,float),0,None); s=X.sum(1,keepdims=True)
    return np.log1p(np.divide(X,s,out=np.zeros_like(X),where=s>0)*1e4)
def zz(X):
    sd=X.std(0,keepdims=True); return (X-X.mean(0,keepdims=True))/np.where(sd>0,sd,1.0)
rows=[]
for c in ["BRCA","KIRC","LUAD"]:
    genes=pd.read_csv(pathlib.Path(cohorts.get(c)["processed"])/f"eval_genes_paper_{c}.txt",
                      header=None)[0].values
    idx=pathways.member_index(pathways.parse_gmt(GMT),genes,min_members=10)
    mem=[v for v in idx.values() if len(v)<=500]
    pair=DS/f"{c}-paper-digital_slide/sample_pair_feature_conch"
    for fold in range(4):
        z=np.load(RES/f"pred_cohort_{c}_fpsplit_tv_gene_fold{fold}.npz",allow_pickle=True)
        slides=[str(s) for s in z["slides"]]
        Y=logcpm(np.stack([h5py.File(pair/f"{s}.h5","r")["tpm"][:] for s in slides]))
        ZY=zz(Y); gY=ZY.mean(1)
        cand={"MoSPR (ours)":np.asarray(z["pred"],float)}
        for n,t,rate in M:
            p=CPOUT/str(fold)/"prediction"/f"{c}-paper-{t.replace('%',str(fold))}_pstvf.npy"
            if not p.exists(): continue
            d=np.load(p,allow_pickle=True).item()
            P=np.stack(d["preds"]).astype(float)
            if P.shape==Y.shape: cand[n]=logcpm(P) if rate else P
        for n,P in cand.items():
            ZP=zz(P); gP=ZP.mean(1)
            SY=np.stack([ZY[:,i].mean(1) for i in mem],1)
            SP=np.stack([ZP[:,i].mean(1) for i in mem],1)

            r2=np.median([np.corrcoef(SP[:,k],gP)[0,1]**2 for k in range(SP.shape[1])])
            rows.append({"cohort":c,"fold":fold,"model":n,
                         "gfac_scc":spearmanr(gY,gP).statistic,
                         "gfac_share_of_setscore_R2":r2,
                         "meanz_scc":np.nanmedian([spearmanr(SY[:,k],SP[:,k]).statistic
                                                   for k in range(SP.shape[1])])})
df=pd.DataFrame(rows)
o=df.groupby(["cohort","model"])[["gfac_scc","gfac_share_of_setscore_R2","meanz_scc"]].mean()
for c in ["BRCA","KIRC","LUAD"]:
    print(f"\n===== {c} ====="); print(o.loc[c].sort_values("meanz_scc",ascending=False).round(4).to_string())
OUTF=_p.TABLES/"tables/robustness/pathway_global_factor.csv"
df.round(5).to_csv(OUTF,index=False)
print("\nwritten:", OUTF)
