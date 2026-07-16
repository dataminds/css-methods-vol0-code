"""트랙 E · 의견 양극화(경계신뢰) — 기말 프로젝트 시작 코드 (self-contained, Mesa 3.x).

요지: 신뢰 범위(eps) 안 상대와만 영향을 주고받으면 집단이 여러 의견으로 갈린다.
ODD-lite: 목적=신뢰 범위가 합의/분열을 가르는가 / 행위자=개인 / 상태=의견(연속) /
  규칙: |의견차|<eps인 이웃 평균 쪽으로 mu만큼 이동 / 시간=매 스텝 전원 /
  출력=군집 수 / 한계=완전연결(네트워크 없음).
서명: eps 작으면 여러 군집, 크면 합의(단일 군집).

실행: python track_e_opinion_polarization.py
"""
from __future__ import annotations
import numpy as np
import mesa


class Person(mesa.Agent):
    def __init__(self, model, opinion):
        super().__init__(model)
        self.opinion = opinion

    def step(self):
        m = self.model
        near = [a.opinion for a in m.agents if abs(a.opinion - self.opinion) < m.eps]
        if near:
            self.opinion += m.mu * (np.mean(near) - self.opinion)


class BoundedConfidence(mesa.Model):
    def __init__(self, n=80, eps=0.2, mu=0.5, seed=None):
        super().__init__(seed=seed)
        self.eps, self.mu = eps, mu
        for _ in range(n):
            Person(self, self.random.random())


def clusters(model):
    return len({round(a.opinion, 1) for a in model.agents})


if __name__ == "__main__":
    print("[서명] eps가 작으면 여러 군집, 크면 합의")
    for eps in (0.05, 0.15, 0.30):
        m = BoundedConfidence(eps=eps, seed=1)
        for _ in range(40):
            m.agents.shuffle_do("step")
        print(f"  eps={eps:>4}: 최종 군집 ~ {clusters(m)}")

# 확장 아이디어(TODO): 완고한 극단주의자(update 안 함) / 2차원 의견 / 미디어 앵커(고정 노드) /
#   비대칭 신뢰(한쪽만 신뢰) / 네트워크 위 경계신뢰.
