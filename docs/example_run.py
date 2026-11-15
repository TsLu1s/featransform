import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
from featransform import Featransform
from featransform.configs.baseline import FTconfig
def make(seed):
    rng=np.random.default_rng(0); n=400
    X=pd.DataFrame({"customer_id":np.arange(1,n+1),"age":rng.integers(18,80,n).astype(float),"income":rng.normal(50,15,n).round(1),
     "city":rng.choice(["Lisbon","Porto","Faro"],n),"signup":pd.date_range("2024-01-01",periods=n,freq="D")})
    X.loc[[1,3,7,11,19,23,31],"income"]=np.nan; X.loc[[2,5,13],"city"]=None
    r2=np.random.default_rng(seed)
    y=pd.Series((X["age"]*0.5+0.1*X["income"].fillna(50)+r2.normal(0,5,n)>35).astype(int))
    return X,y
mode=sys.argv[1]
if mode=="scan":
    for s in range(12):
        X,y=make(s); ft=Featransform(FTconfig._get("standard",task_type=None)); out=ft.fit_transform(X,y)
        h=ft.get_optimization_history()
        print(s,[ (e.n_features,round(e.score,3)) for e in h], out.shape[1])
elif mode=="show":
    s=int(sys.argv[2]); X,y=make(s); ft=Featransform(FTconfig._get("standard",task_type=None)); out=ft.fit_transform(X,y)
    pd.set_option("display.width",250)
    print(X.head(4).to_string())
    d=ft._datetime_expander.transform(X.head(4).copy()); print(list(d.columns))
    d=ft._components["imputer"].transform(d); print(d[["customer_id","age","income","city"]].to_string())
    d=ft._components["encoder"].transform(d); print(d[["age","income","city"]].to_string())
    print(getattr(ft._components["encoder"],"__dict__",{}).keys())
    fe=ft._prepare_fe_input(d,False); print(fe.iloc[:, :4].round(2).to_string())
    for c in ["anomaly","clustering","dimensionality"]:
        f=ft._components[c].transform(fe).features; print(f.round(2).iloc[:, :5].to_string())
    for e in ft.get_optimization_history(): print(e.iteration, round(e.score,4), e.n_features)
    print(ft.get_summary()["optimization"]); print(list(out.columns))
    imp=ft.get_feature_importance(); print([(k,round(v,3)) for k,v in sorted(imp.items(),key=lambda t:-t[1])][:12])
    print(out.head(4).round(2).to_string())
if mode=="rows":
    X,y=make(4); ft=Featransform(FTconfig._get("standard",task_type=None)); out=ft.fit_transform(X,y)
    d=ft._datetime_expander.transform(X.copy()); d=ft._components["imputer"].transform(d); d=ft._components["encoder"].transform(d)
    fe=ft._prepare_fe_input(d,False)
    E=pd.concat([ft._components[c].transform(fe).features for c in ["anomaly","clustering","dimensionality"]],axis=1)
    print(E.iso_anomaly.sum(), E.lof_anomaly.sum(), E.kme_cluster.value_counts().to_dict())
    idx=E.index[(E.iso_anomaly==1)&(E.lof_anomaly==1)][:3].tolist(); idx2=E.index[E.iso_anomaly==1][:3].tolist()
    print(idx, idx2)
    rows=[0,1,2]+idx[:1]
    print(X.loc[rows].to_string()); print(d.loc[rows,["customer_id","age","income","city","signup_month","signup_month_cos"]].round(2).to_string())
    print(E.loc[rows,["iso_anomaly","lof_anomaly","kme_cluster","pca_0","pca_1","pca_9"]].round(2).to_string())
    print(out.loc[rows].round(2).to_string()); print(y.loc[rows].tolist())
    sel=ft._components["selector"]; df=sel._importance_df; print(df.head(16).round(3).to_string())
    print(np.linspace(0.5,1.0,10).round(3))
if mode=="proc":
    X,y=make(4); ft=Featransform(FTconfig._get("standard",task_type=None)); out=ft.fit_transform(X,y)
    d=ft._datetime_expander.transform(X.copy()); d=ft._components["imputer"].transform(d); d=ft._components["encoder"].transform(d)
    print(d.shape); print(d.loc[[0,1,2,41]].round(2).T.to_string())
    print(ft._components["encoder"]._code_maps)
    print(y.value_counts().to_dict())
