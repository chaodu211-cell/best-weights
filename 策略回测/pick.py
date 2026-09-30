# -*- coding: utf-8 -*-
"""从 _grid.csv 里挑策略。规则事先定好：

  1. 可行 = 目标窗口年化 ≥ 50% 且最大回撤不超过 30%。
  2. 余量分 = min((年化 − 50%) / 50%, (回撤 + 30%) / 30%)：两个约束里离得最近的那一个还剩多少余量，
     取余量分最高的前 N 个候选（只看最大年化会挑到恰好擦线的那一组）。
  3. 对每个候选，把每个参数单独挪到相邻取值（其余不动），全部重跑——"邻居"。
     稳健度 = 邻居里仍然可行的比例。最终选：邻居可行比例最高者，平手看邻居年化中位数。
  4. 选定之后才看样本外（2007-2016；三年策略另看 2016-09~2023-09），样本外不参与挑选。

    python3 策略回测/pick.py              # 默认网格（grid.py → _grid.csv → _picked.json）
    python3 策略回测/pick.py grid_semi    # 七只标的网格（→ _grid_semi.csv → _picked_semi.json）
"""
import importlib, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MOD = sys.argv[1] if len(sys.argv) > 1 else "grid"
GR = importlib.import_module(MOD)
SUF = "" if MOD == "grid" else "_" + MOD.split("_", 1)[-1]
GRID_CSV = getattr(GR, "OUT", os.path.join(HERE, "_grid.csv"))

TGT_CAGR, TGT_MDD = 0.50, -0.30
N_CAND = 40
KEYS = list(GR.SPACE)


def norm(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def row_cfg(r):
    c = {k: norm(r[k]) for k in KEYS}
    for k in ("red_hold", "blue_hold", "blue_k"):
        c[k] = int(c[k])
    for k in ("use_ad",):
        c[k] = bool(c[k])
    return c


def neighbors(c):
    out = []
    for k, vals in GR.SPACE.items():
        vals = list(vals)
        try:
            i = vals.index(c[k])
        except ValueError:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals) and not (k == "universe"):
                d = dict(c); d[k] = vals[j]
                if hasattr(GR, "normalize"):
                    d = GR.normalize(d)
                elif d["red_hold"] == 0:
                    d["red_mult"], d["use_ad"] = 0.0, False
                out.append((k, vals[j], d))
    return out


def score(df, w):
    return np.minimum((df[f"{w}_cagr"] - TGT_CAGR) / TGT_CAGR, (df[f"{w}_mdd"] - TGT_MDD) / -TGT_MDD)


def main():
    df = pd.read_csv(GRID_CSV)
    res = {}
    with Pool(8, initializer=GR.init) as pool:
        for w in ("10y", "3y"):
            ok = (df[f"{w}_cagr"] >= TGT_CAGR) & (df[f"{w}_mdd"] >= TGT_MDD)
            print(f"\n=== {w}：可行 {ok.sum():,} / {len(df):,} 组（{ok.mean():.1%}）===")
            d = df[ok].copy()
            d["score"] = score(d, w)
            cand = d.sort_values("score", ascending=False).head(N_CAND)
            best = None
            for _, r in cand.iterrows():
                c = row_cfg(r)
                nb = neighbors(c)
                rows = pool.map(GR.evaluate, [x[2] for x in nb])
                R = pd.DataFrame(rows)
                feas = ((R[f"{w}_cagr"] >= TGT_CAGR) & (R[f"{w}_mdd"] >= TGT_MDD)).mean()
                rec = dict(cfg=c, row=r.to_dict(), nb_feas=feas, nb_cagr_med=R[f"{w}_cagr"].median(),
                           nb_mdd_med=R[f"{w}_mdd"].median(), nb_worst_mdd=R[f"{w}_mdd"].min(),
                           nb=[(k, str(v), a, b) for (k, v, _), a, b in zip(nb, R[f"{w}_cagr"], R[f"{w}_mdd"])])
                key = (feas, rec["nb_cagr_med"])
                print(f"  score {r['score']:.3f}  {r[f'{w}_cagr']:.1%}/{r[f'{w}_mdd']:.1%}  邻居可行 {feas:.0%}"
                      f"  邻居中位 {rec['nb_cagr_med']:.1%}/{rec['nb_mdd_med']:.1%}  {c['universe']}")
                if best is None or key > (best["nb_feas"], best["nb_cagr_med"]):
                    best = rec
            res[w] = best
    json.dump(res, open(os.path.join(HERE, f"_picked{SUF}.json"), "w"), ensure_ascii=False, indent=1, default=str)
    for w, b in res.items():
        print(f"\n★ {w} 选定：{json.dumps(b['cfg'], ensure_ascii=False)}")
        r = b["row"]
        for k in GR.WIN:
            print(f"   {k:6s} 年化 {r[f'{k}_cagr']:7.1%}  最大回撤 {r[f'{k}_mdd']:7.1%}")


if __name__ == "__main__":
    main()
