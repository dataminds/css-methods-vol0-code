"""계산소통 공용 ABM 헬퍼 (Mesa 3.x).

학부생이 배관(scheduling·복제·집계·그림)에 시간을 뺏기지 않도록 최소 도구만 제공한다.
⭐ AI 보조로 코드를 짜더라도, 이 함수들이 '무엇을 하는지' 읽고 검증할 수 있어야 한다.
   (제출물은 학생이 읽고 검증·설명 가능해야 함 ; 책 부록 B)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 창 없이 파일로 저장 (노트북·서버 안전)
import matplotlib.pyplot as plt


def run(model, steps):
    """모형 하나를 steps 단계 돌리고 DataCollector의 시계열 DataFrame을 반환."""
    for _ in range(steps):
        model.step()
    return model.datacollector.get_model_vars_dataframe()


def run_seeds(model_cls, reporter, seeds=range(20), steps=40, **kwargs):
    """여러 seed로 돌려 '마지막 시점 reporter 값'의 분포를 모은다.

    단일 실행 결과에 속지 않도록, 확률적 모형은 반드시 다수 seed의 분포로 본다.
    reporter(model) -> 수치.
    """
    out = []
    for s in seeds:
        m = model_cls(seed=s, **kwargs)
        for _ in range(steps):
            m.step()
        out.append(reporter(m))
    return np.array(out, dtype=float)


def plot_series(df, path, title=""):
    """DataCollector DataFrame을 선그래프로 저장."""
    ax = df.plot(figsize=(6, 4))
    ax.set_xlabel("step")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def plot_hist(values, path, title=""):
    """다수 seed 분포 등을 히스토그램으로 저장."""
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins="auto", color="#4C78A8", edgecolor="white")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path
