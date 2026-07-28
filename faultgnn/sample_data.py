# -*- coding: utf-8 -*-
"""デモ用のFault Managementサンプルデータを生成する。
ドキュメント中の例（Motor×高温×摩耗が多発 / 装置B×E205×夜勤×高湿度が急増している未知パターン）
を模した相関構造をあえて仕込み、3手法の挙動が確認しやすいようにしている。
"""
import numpy as np
import pandas as pd


def build_sample_dataframe(n_main=380, n_rare_pattern=22, seed=7, base_date=None) -> pd.DataFrame:
    """デモ用データを生成する。

    base_date は日付列の基準日（この日から過去に向かってデータを配置する）。
    未指定なら実行日。固定日を渡すと再現可能なデータになる（テストで使用）。
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(base_date) if base_date is not None else pd.Timestamp.today().normalize()

    devices = ["装置A", "装置B", "装置C", "装置D"]
    parts = ["Motor", "Bearing", "Belt", "Sensor", "Valve"]
    symptoms = ["振動", "異音", "発熱", "漏れ", "誤動作"]
    errors = ["E101", "E102", "E103", "E205", "E310"]
    conditions = ["高温", "低温", "通常", "高湿度", "夜間"]
    processes = ["組立", "検査", "搬送", "梱包"]
    causes = ["摩耗", "軸ずれ", "潤滑不足", "異物混入", "センサ故障", "配線不良"]
    actions = {
        "摩耗": "Bearing交換・定期給油周期の見直し",
        "軸ずれ": "アライメント再調整・固定ボルト増し締め",
        "潤滑不足": "給油・グリスアップ実施",
        "異物混入": "フィルター清掃・防塵カバー追加",
        "センサ故障": "センサ交換・配線点検",
        "配線不良": "配線点検・コネクタ交換",
    }

    rows = []

    # --- 主パターン: 装置×部品×条件×Errorと原因に相関を持たせる ---
    for _ in range(n_main):
        device = rng.choice(devices, p=[0.35, 0.25, 0.25, 0.15])
        part = rng.choice(parts)
        condition = rng.choice(conditions, p=[0.30, 0.15, 0.35, 0.10, 0.10])

        if part == "Motor" and condition == "高温":
            symptom = rng.choice(["振動", "発熱"], p=[0.6, 0.4])
            error = rng.choice(["E102", "E101"], p=[0.7, 0.3])
            cause = rng.choice(["摩耗", "潤滑不足", "軸ずれ"], p=[0.6, 0.25, 0.15])
        elif part == "Bearing":
            symptom = rng.choice(["異音", "振動"], p=[0.6, 0.4])
            error = rng.choice(["E103", "E101"], p=[0.5, 0.5])
            cause = rng.choice(["摩耗", "軸ずれ"], p=[0.55, 0.45])
        elif part == "Sensor":
            symptom = "誤動作"
            error = rng.choice(["E310", "E103"], p=[0.7, 0.3])
            cause = "センサ故障"
        elif part == "Valve":
            symptom = "漏れ"
            error = rng.choice(["E101", "E310"], p=[0.6, 0.4])
            cause = rng.choice(["異物混入", "配線不良"], p=[0.7, 0.3])
        else:
            symptom = rng.choice(symptoms)
            error = rng.choice(errors)
            cause = rng.choice(causes)

        process = rng.choice(processes, p=[0.4, 0.25, 0.2, 0.15])
        rows.append(
            dict(
                装置=device, 部品=part, 症状=symptom, Error=error,
                条件=condition, 工程=process, 原因=cause, 対策=actions[cause],
                備考=f"{device}の{part}で{symptom}を確認。{condition}環境下で発生。",
            )
        )

    # --- レアパターン: 装置B×E205×夜間×高湿度が最近急増（未知パターン用の"埋め込み"クラスタ） ---
    for _ in range(n_rare_pattern):
        part = rng.choice(["Sensor", "Belt"])
        cause = rng.choice(["センサ故障", "異物混入"], p=[0.6, 0.4])
        rows.append(
            dict(
                装置="装置B", 部品=part, 症状=rng.choice(["誤動作", "発熱"]),
                Error="E205", 条件=rng.choice(["夜間", "高湿度"], p=[0.5, 0.5]),
                工程=rng.choice(["搬送", "検査"]), 原因=cause, 対策=actions[cause],
                備考="夜勤帯・高湿度条件下で発生。従来はあまり見られなかった組み合わせ。",
            )
        )

    df = pd.DataFrame(rows)

    # 日付列: 主パターンは過去1年に一様分布、レアパターンは直近60日に集中させ「急増」を再現
    dates = []
    for i in range(len(df)):
        if i < n_main:
            dates.append(base - pd.Timedelta(days=int(rng.integers(0, 365))))
        else:
            dates.append(base - pd.Timedelta(days=int(rng.integers(0, 60))))
    df["発生日"] = dates

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_sample_dataframe()
    df.to_csv("sample_fault_data.csv", index=False, encoding="utf-8-sig")
    print(df.head())
    print(len(df))
