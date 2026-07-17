# ===== 빈칸판(학생 배포용) =====
# TODO 블록을 채우고 raise 줄을 지우면, 완성본과 같은 서명이 나와야 합니다.
# 검증: 극단값 시험 + 씨앗 73 재실행 + (가능하면) 독립 재계산(새 셀·계산기). 검증 로그 한 문단!

"""3주 · 얇은 행위자: 경계신뢰(bounded confidence) 의견 동역학 (Mesa 3.x).

이론: 개인은 '신뢰 범위(eps)' 안의 상대와만 영향을 주고받아 평균 쪽으로 이동한다.
서명: eps가 작으면 여러 의견 군집, 크면 합의(단일 군집).
반증: eps를 바꿔도 군집 수가 그대로면 이 규칙이 아니다.

수업용 빈칸판: step()의 갱신 한 줄을 TODO로 비워 학생이 채운다(아래는 완성본).
실행: python w03_bounded_confidence.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import mesa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 상위 폴더(materials) abmlib 임포트
from abmlib import run, run_seeds, plot_series


def clusters(model):
    """소수점 첫째 자리로 반올림한 의견의 종류 수 = 대략의 군집 수."""
    return len({round(a.opinion, 1) for a in model.agents})


class Person(mesa.Agent):
    def __init__(self, model, opinion):
        super().__init__(model)
        self.opinion = opinion            # 상태: 의견 (0~1 연속)

    def step(self):
        eps, mu = self.model.eps, self.model.mu
        near = [a.opinion for a in self.model.agents
                if abs(a.opinion - self.opinion) < eps]   # 신뢰 범위 이웃
        if near:
            # =============================================================
            # TODO(학생): 갱신 규칙 한 줄 (책 3장 §3.2)
            # 힌트: 신뢰 범위 이웃 평균(np.mean(near)) 쪽으로, mu만큼 이동
            # 꼴: self.opinion += mu * ( ??? - self.opinion )
            raise NotImplementedError("TODO: 규칙을 채우세요")
            # =============================================================


class BoundedConfidence(mesa.Model):
    def __init__(self, n=80, eps=0.2, mu=0.5, seed=None):
        super().__init__(seed=seed)
        self.eps, self.mu = eps, mu
        for _ in range(n):
            Person(self, self.random.random())            # 초기 의견 무작위
        self.datacollector = mesa.DataCollector(
            model_reporters={"clusters": clusters})

    def step(self):
        self.agents.shuffle_do("step")                    # 무작위 순서로 각자 step
        self.datacollector.collect(self)


if __name__ == "__main__":
    print("[설계 점검] eps를 바꾸면 군집 수가 달라져야 한다 (서명)")
    for eps in (0.05, 0.15, 0.30):
        df = run(BoundedConfidence(eps=eps, seed=1), 40)
        print(f"  eps={eps:>4}: 최종 군집 ~ {int(df['clusters'].iloc[-1])}")
    print("[확률성] 단일 실행이 아니라 다수 seed 분포로 본다")
    dist = run_seeds(BoundedConfidence, clusters, seeds=range(20), steps=40, eps=0.1)
    print(f"  eps=0.1, 20 seeds: 평균 {dist.mean():.2f} 범위 {int(dist.min())}~{int(dist.max())}")
    plot_series(run(BoundedConfidence(eps=0.1, seed=1), 40),
                "w03_clusters.png", "bounded confidence (eps=0.1)")
    print("  그림 저장: w03_clusters.png")
