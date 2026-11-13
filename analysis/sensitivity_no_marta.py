# Sensitivity: rank stability (Q3) and Q4 recomputed without MARTA (failed to train on spread).
import pandas as pd
from itertools import combinations
from scipy.stats import kendalltau, spearmanr
pd.set_option('display.width',250)
m=pd.read_csv("metrics.csv"); l=pd.read_csv("learned.csv")
m=m[m.method!='marta']; l=l[l.method!='marta']
cells={}
for (t,s,o,sc),g in m.groupby(['task','source','op','scope']): cells[(t,f"{s}-{o}-{sc}")]=g.groupby('method').retained_auc.mean()
for (t,o,a,sc),g in l.groupby(['task','op','alpha','scope']): cells[(t,f"L-{o}-a{a}-{sc}")]=g.groupby('method').retained.mean()
rs=[]
for (k1,r1),(k2,r2) in combinations(cells.items(),2):
    if k1[0]!=k2[0]: continue
    c=r1.index.intersection(r2.index); tau,p=kendalltau(r1[c],r2[c])
    rs.append(dict(task=k1[0],cell_a=k1[1],cell_b=k2[1],kendall_tau=tau,p=p,n_methods=len(c)))
rs=pd.DataFrame(rs); rs.to_csv("rank_stability_no_marta.csv",index=False)
rs['pair']=[''.join(sorted(a[0]+b[0])) for a,b in zip(rs.cell_a,rs.cell_b)]
print(rs.groupby(['task','pair']).kendall_tau.agg(['mean','min','max','count']).round(2))
print(rs.groupby('task').kendall_tau.agg(['mean','median']).round(2))
# ranking table (1 = most retained) per cell
rk=pd.DataFrame({f"{t}|{c}":v.rank(ascending=False) for (t,c),v in cells.items()}).T
print(rk.to_string())
q=[]
rr=m[m.source=='R'].groupby(['task','method','op','scope']).retained_auc.mean()
for (t,o,a,sc),g in l.groupby(['task','op','alpha','scope']):
    x=rr.loc[t].xs((o,sc),level=('op','scope')); y=g.groupby('method').retained.mean()
    c=x.index.intersection(y.index); rho,p=spearmanr(x[c],y[c])
    q.append(dict(task=t,learned_cell=f"{o}-a{a}-{sc}",spearman=rho,p=p,n_methods=len(c)))
q=pd.DataFrame(q); q.to_csv("q4_correlation_no_marta.csv",index=False); print(q.round(2).to_string(index=False))
