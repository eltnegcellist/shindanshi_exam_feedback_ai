import json, os, re, time, math
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from laya.onnx_agent import ONNXAgent
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.json"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# 24 semantic decision features: 6 per exam question.
CRITERIA = {
    "q1": {
        "q1_swot_valid": "SWOTのS・Wを内部環境、O・Tを外部環境として概ね適切に分類できているか。",
        "q1_case_specific": "一般論ではなく、A社の具体的な強み・弱み・機会・脅威を記述しているか。",
        "q1_balanced": "S・W・O・Tの4区分を偏りなくカバーしているか。",
        "q1_distinct": "同じ内容の言い換えを重複させず、異なる観点を複数示しているか。",
        "q1_clear": "各要素が簡潔で、何が強み・弱み・機会・脅威なのか明確に読めるか。",
        "q1_relevant": "新規の木製知育玩具事業を考えるうえで重要な経営環境要因を選べているか。",
    },
    "q2": {
        "q2_both_requirements": "取り組みだけの列挙ではなく、取り組みと工夫の両方を答えているか。",
        "q2_case_resources": "県・大学・子息の知見・既存の顧客接点など、A社固有の資源や関係性を施策に結び付けているか。",
        "q2_customer_loop": "顧客接点や実証・対話からニーズや気づきを得て、製品開発・改善につなげる因果関係が示されているか。",
        "q2_target_channel": "対象顧客と、EC・SNS・イベント・店舗等の接点や販路との対応関係が説明されているか。",
        "q2_effect": "施策を行う目的・効果まで説明しており、単なる施策名の羅列になっていないか。",
        "q2_coherent": "複数の施策が互いに矛盾せず、一つの事業開発・顧客獲得の流れとして理解できるか。",
    },
    "q3": {
        "q3_structure_explicit": "推奨する組織体制を具体的に示しているか。",
        "q3_business_fit": "知育玩具事業の市場変化の速さ、必要スキル、既存事業との違いなど事業特性を理由に結び付けているか。",
        "q3_authority_speed": "権限・責任の明確化や権限委譲と、意思決定の迅速化を因果的に結び付けているか。",
        "q3_hr_development": "次世代リーダーや後継者、専門人材の育成・確保という観点を組織体制に結び付けているか。",
        "q3_resource_coordination": "既存事業の技術・人材などの資源活用や部門間連携について考慮しているか。",
        "q3_reasoned": "組織形態の名称だけでなく、その体制を採る理由と期待効果を筋道立てて説明しているか。",
    },
    "q4": {
        "q4_redefine": "企業理念をどのように再定義するか、その新しい理念の内容を明示しているか。",
        "q4_future_value": "木材・地域資源、新しい価値、子ども・次世代などA社の将来方向を理念に意味的に結び付けているか。",
        "q4_internal": "社員に理念を浸透させる具体策を示しているか。",
        "q4_external": "顧客・大学・地域・職人など社外の関係者にも理念を伝え共有する観点があるか。",
        "q4_mechanism": "説明会・対話・研修・評価・日常活動への組込み等、理念浸透の具体的な仕組みを示しているか。",
        "q4_alignment": "理念浸透策を、共通目的・一体感・士気・継続的な行動などの効果につなげて説明しているか。",
    },
}

def split_questions(answer: str):
    # Works with full-width numerals used in the dataset.
    matches = list(re.finditer(r"第([１２３４])問（配点(?:20|30)点）", answer))
    out = {}
    mp = {"１":"q1","２":"q2","３":"q3","４":"q4"}
    for i,m in enumerate(matches):
        st = m.end()
        en = matches[i+1].start() if i+1 < len(matches) else len(answer)
        out[mp[m.group(1)]] = answer[st:en].strip()
    return out

def laya_questions(qmap):
    return {
        k: {"type":"noul", "instructions":v}
        for k,v in qmap.items()
    }

def metric_dict(y, pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    return {
        "mae": float(mean_absolute_error(y,pred)),
        "rmse": float(mean_squared_error(y,pred)**0.5),
        "pearson": float(pearsonr(y,pred).statistic),
        "spearman": float(spearmanr(y,pred).statistic),
        "pred_sd": float(np.std(pred,ddof=1)),
        "actual_sd": float(np.std(y,ddof=1)),
        "acc60": float(accuracy_score(y>=60,pred>=60)),
    }

def pair_order(y,pred,folds=None,gap=10):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    good=tot=0
    n=len(y)
    for i in range(n):
        for j in range(i+1,n):
            if folds is not None and folds[i] != folds[j]:
                continue
            if abs(y[i]-y[j]) < gap or y[i]==y[j]:
                continue
            tot += 1
            good += int((pred[i]-pred[j])*(y[i]-y[j]) > 0)
    return float(good/tot) if tot else float("nan"), int(tot)

def score_bins(y):
    # Five broad score strata for balanced folds.
    y=np.asarray(y)
    return np.digitize(y,[45,50,55,60,70])

def nested_ridge_oof(X,y,fold_splits):
    y=np.asarray(y,float)
    pred=np.zeros(len(y))
    chosen=[]
    alphas=[0.01,0.1,1,3,10,30,100,300]
    for fold,(tr,te) in enumerate(fold_splits):
        # small inner CV entirely inside outer training data
        inner=KFold(n_splits=5,shuffle=True,random_state=100+fold)
        best_a=None; best_mae=1e9
        for a in alphas:
            vals=[]
            for itr,iva in inner.split(tr):
                tri=tr[itr]; vai=tr[iva]
                m=Ridge(alpha=a)
                m.fit(X[tri],y[tri])
                vals.append(mean_absolute_error(y[vai],m.predict(X[vai])))
            v=float(np.mean(vals))
            if v<best_mae:
                best_mae=v; best_a=a
        m=Ridge(alpha=best_a)
        m.fit(X[tr],y[tr])
        pred[te]=m.predict(X[te])
        chosen.append(best_a)
    return pred, chosen

def ngram_oof(texts,y,fold_splits):
    y=np.asarray(y,float)
    pred=np.zeros(len(y))
    chosen=[]
    alphas=[0.1,1,3,10,30,100]
    texts=np.asarray(texts,dtype=object)
    for fold,(tr,te) in enumerate(fold_splits):
        # Fit vocabulary only on outer training.
        vec=TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=50000,sublinear_tf=True)
        Xtr=vec.fit_transform(texts[tr])
        Xte=vec.transform(texts[te])
        inner=KFold(n_splits=5,shuffle=True,random_state=200+fold)
        best_a=None; best_mae=1e9
        for a in alphas:
            vals=[]
            for itr,iva in inner.split(np.arange(len(tr))):
                m=Ridge(alpha=a)
                m.fit(Xtr[itr],y[tr][itr])
                vals.append(mean_absolute_error(y[tr][iva],m.predict(Xtr[iva])))
            v=float(np.mean(vals))
            if v<best_mae:
                best_mae=v; best_a=a
        m=Ridge(alpha=best_a)
        m.fit(Xtr,y[tr])
        pred[te]=m.predict(Xte)
        chosen.append(best_a)
    return pred, chosen

def main():
    data=json.loads(DATA.read_text())
    ids=[x["id"] for x in data]
    y=np.array([x["score"] for x in data],dtype=float)
    texts=[x["answer"] for x in data]

    model_dir=snapshot_download(
        "soyelmismo/laya-multilingual-onnx",
        allow_patterns=["model-int8.onnx","model.onnx","rl_agent_config.json","tokenizer.json","tokenizer/*"],
    )
    model_path=os.path.join(model_dir,"model-int8.onnx")
    if not os.path.exists(model_path):
        model_path=os.path.join(model_dir,"model.onnx")
    agent=ONNXAgent(model_dir,onnx_path=model_path)

    feature_names=[k for q in ("q1","q2","q3","q4") for k in CRITERIA[q]]
    X=np.full((len(data),len(feature_names)),np.nan,dtype=float)
    fidx={n:i for i,n in enumerate(feature_names)}
    raw_rows=[]
    t0=time.time()
    for i,row in enumerate(data):
        qs=split_questions(row["answer"])
        rec={"id":row["id"],"score":row["score"]}
        for qn in ("q1","q2","q3","q4"):
            state={
                "exam":"中小企業診断士2次筆記試験 令和7年度 事例I",
                "question":qn,
                "answer":qs.get(qn,""),
            }
            res=agent.system_one(state,laya_questions(CRITERIA[qn]),max_len=1024,head_max_len=256)
            for name,ans in res["answers"].items():
                v=float(ans["noul"])
                X[i,fidx[name]]=v
                rec[name]=v
        raw_rows.append(rec)
        if (i+1)%10==0:
            print(f"Laya features {i+1}/{len(data)} elapsed={time.time()-t0:.1f}s",flush=True)

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite semantic features")

    # Fixed, score-stratified five outer folds used identically by both models.
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260924)
    bins=score_bins(y)
    splits=list(skf.split(np.zeros(len(y)),bins))
    fold_id=np.full(len(y),-1,int)
    for f,(_,te) in enumerate(splits): fold_id[te]=f

    sem_pred,sem_alpha=nested_ridge_oof(X,y,splits)
    ng_pred,ng_alpha=ngram_oof(texts,y,splits)

    # A simple unweighted semantic index, useful to test whether the raw judgement vector
    # itself is ordered before any supervised regression.
    sem_index=X.mean(axis=1)

    sem_metrics=metric_dict(y,sem_pred)
    ng_metrics=metric_dict(y,ng_pred)
    idx_corr={
        "pearson":float(pearsonr(y,sem_index).statistic),
        "spearman":float(spearmanr(y,sem_index).statistic),
    }
    for name,pred,met in [("semantic_ridge",sem_pred,sem_metrics),("char_ngram",ng_pred,ng_metrics)]:
        a,n=pair_order(y,pred,folds=fold_id,gap=10)
        ag,ng=pair_order(y,pred,folds=None,gap=10)
        met["pair_order_10_same_fold"]=a
        met["pair_order_10_same_fold_n"]=n
        met["pair_order_10_global_oof"]=ag
        met["pair_order_10_global_oof_n"]=ng

    # Feature-level exploratory correlations and high/low separation.
    feat_stats=[]
    high=y>=65
    low=y<50
    for j,n in enumerate(feature_names):
        x=X[:,j]
        sd=np.std(x,ddof=1)
        pooled=math.sqrt(((high.sum()-1)*np.var(x[high],ddof=1)+(low.sum()-1)*np.var(x[low],ddof=1))/max(1,(high.sum()+low.sum()-2)))
        d=(float(x[high].mean()-x[low].mean())/pooled) if pooled>1e-12 else 0.0
        feat_stats.append({
            "feature":n,
            "pearson":float(pearsonr(y,x).statistic),
            "spearman":float(spearmanr(y,x).statistic),
            "mean_ge65":float(x[high].mean()),
            "mean_lt50":float(x[low].mean()),
            "cohen_d":float(d),
        })
    feat_stats.sort(key=lambda z:abs(z["pearson"]),reverse=True)

    # Twin-answer diagnostic from the previous keyword experiment.
    twin_pairs=[
        ("R07I-019","R07I-200"),("R07I-029","R07I-200"),
        ("R07I-022","R07I-194"),("R07I-022","R07I-189"),
        ("R07I-076","R07I-200"),("R07I-053","R07I-182"),
        ("R07I-059","R07I-177"),("R07I-041","R07I-163"),
    ]
    pos={k:i for i,k in enumerate(ids)}
    twins=[]
    for a,b in twin_pairs:
        if a in pos and b in pos:
            ia,ib=pos[a],pos[b]
            higher=a if y[ia]>y[ib] else b
            sem_higher=a if sem_pred[ia]>sem_pred[ib] else b
            ng_higher=a if ng_pred[ia]>ng_pred[ib] else b
            twins.append({
                "a":a,"b":b,"score_a":float(y[ia]),"score_b":float(y[ib]),
                "semantic_pred_a":float(sem_pred[ia]),"semantic_pred_b":float(sem_pred[ib]),
                "ngram_pred_a":float(ng_pred[ia]),"ngram_pred_b":float(ng_pred[ib]),
                "actual_higher":higher,
                "semantic_correct":sem_higher==higher,
                "ngram_correct":ng_higher==higher,
            })

    frame=pd.DataFrame(raw_rows)
    frame["fold"]=fold_id
    frame["semantic_index"]=sem_index
    frame["semantic_pred_oof"]=sem_pred
    frame["ngram_pred_oof"]=ng_pred
    frame.to_csv(OUT/"features_and_predictions.csv",index=False)

    result={
        "n":len(data),
        "model":"soyelmismo/laya-multilingual-onnx QInt8 / Laya multilingual",
        "features":feature_names,
        "semantic_metrics":sem_metrics,
        "ngram_metrics":ng_metrics,
        "raw_semantic_index_corr":idx_corr,
        "semantic_alphas":sem_alpha,
        "ngram_alphas":ng_alpha,
        "top_feature_stats":feat_stats,
        "twins":twins,
    }
    (OUT/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))

    lines=[
        "# Laya System One × 令和7年度事例I 採点予測実験",
        "",
        f"- 答案数: {len(data)}",
        "- 特徴: Laya Multilingualによる24個の意味判断確率（点数はLayaに見せない）",
        "- 評価: 得点帯で層化した5-fold OOF。各外側fold内でRidgeのalphaを内側CV選択。",
        "- 比較: 同一outer foldの文字2–5gram TF-IDF + Ridge",
        "",
        "## 結果",
        "",
        "| model | MAE | RMSE | Pearson | Spearman | 10点差順位正解率* | pred SD | 60点判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| System One semantic | {sem_metrics['mae']:.3f} | {sem_metrics['rmse']:.3f} | {sem_metrics['pearson']:.3f} | {sem_metrics['spearman']:.3f} | {sem_metrics['pair_order_10_same_fold']*100:.1f}% | {sem_metrics['pred_sd']:.2f} | {sem_metrics['acc60']*100:.1f}% |",
        f"| char n-gram | {ng_metrics['mae']:.3f} | {ng_metrics['rmse']:.3f} | {ng_metrics['pearson']:.3f} | {ng_metrics['spearman']:.3f} | {ng_metrics['pair_order_10_same_fold']*100:.1f}% | {ng_metrics['pred_sd']:.2f} | {ng_metrics['acc60']*100:.1f}% |",
        "",
        "*10点以上離れ、かつ同じouter foldに入った答案ペアのみ。異なる学習モデルの予測値を直接比較しないため。",
        "",
        f"教師なしの単純な24特徴平均（回帰前）と実得点の相関: Pearson {idx_corr['pearson']:.3f}, Spearman {idx_corr['spearman']:.3f}",
        "",
        "## 得点と関係が強かった意味特徴（探索的）",
        "",
        "| feature | Pearson | Spearman | >=65平均 | <50平均 | Cohen d |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for z in feat_stats[:10]:
        lines.append(f"| {z['feature']} | {z['pearson']:.3f} | {z['spearman']:.3f} | {z['mean_ge65']:.3f} | {z['mean_lt50']:.3f} | {z['cohen_d']:.2f} |")
    lines += ["","## 双子答案","",
              "| pair | actual | semantic OOF | ngram OOF | semantic正誤 | ngram正誤 |",
              "|---|---:|---:|---:|---|---|"]
    for z in twins:
        lines.append(f"| {z['a']} / {z['b']} | {z['score_a']:.0f}/{z['score_b']:.0f} | {z['semantic_pred_a']:.1f}/{z['semantic_pred_b']:.1f} | {z['ngram_pred_a']:.1f}/{z['ngram_pred_b']:.1f} | {'○' if z['semantic_correct'] else '×'} | {'○' if z['ngram_correct'] else '×'} |")
    lines += [
        "",
        "## 注意",
        "",
        "- 再現答案の総得点しかなく設問別の真の得点はない。",
        "- 24評価軸は探索的に設計したため、この200答案での結果は最終的な未知年度性能ではない。",
        "- Layaは診断士答案専用にfine-tuneしていないゼロショット判断モデルである。",
        "- 本命は、この実験で有望な軸が確認できた場合に診断士答案のペア比較/段階評価でfine-tuneすること。",
    ]
    (OUT/"report.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
