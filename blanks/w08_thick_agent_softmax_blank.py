# ===== 빈칸판(학생 배포용) =====
# TODO 블록을 채우고 raise 줄을 지우면, 완성본과 같은 서명이 나와야 합니다.
# 검증: 극단값 시험 + 씨앗 73 재실행 + (가능하면) 손 검산. 검증 로그 한 문단!

"""8주 · 두꺼운 행위자: 가치·선택(softmax) + 갱신 = 선택적 노출 (Mesa 3.x).

얇은 행위자(수치+간단 규칙)와 달리, 두꺼운 행위자는 '가치를 계산하고 확률로 고르고
경험으로 상태를 바꾼다'. 여기서:
  가치   = 동의도 = 태도 x 메시지 입장 (같은 방향일수록 높음)
  선택   = softmax (온도 모수 tau: 낮으면 최고가치에 몰림 = 강한 편식)
  갱신   = 고른 메시지 쪽으로 태도 이동(선택적 노출 -> 강화)
서명: 태도가 극단으로 벌어진다(양극화). tau가 낮을수록 강함.
반증(절제): 갱신을 끄면(alpha=0) 양극화가 사라진다 -> 갱신 항이 서명을 떠받친다.

실행: python w08_thick_agent_softmax.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import mesa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 상위 폴더(materials) abmlib 임포트
from abmlib import run


def softmax(values, tau):
    v = np.asarray(values, dtype=float) / tau
    v -= v.max()                       # 수치 안정
    e = np.exp(v)
    return e / e.sum()


def polarization(model):
    """평균 |태도| = 중앙에서 얼마나 벌어졌나 (0 중앙 ~ 1 극단)."""
    return float(np.mean([abs(a.attitude) for a in model.agents]))


class Reader(mesa.Agent):
    def __init__(self, model, attitude):
        super().__init__(model)
        self.attitude = attitude       # 상태: 태도 -1..1

    def step(self):
        m = self.model
        value = [self.attitude * pos for pos in m.messages]        # 가치 = 동의도
        probs = softmax(value, m.tau)                              # 선택 규칙
        idx = m.random.choices(range(len(m.messages)), weights=probs)[0]
        # =============================================================
        # TODO(학생): 갱신 규칙 (책 8장 §8.3)
        # 힌트: 고른 메시지(m.messages[idx]) 쪽으로, 보폭 m.alpha 만큼
        # 꼴: self.attitude += m.alpha * ( ??? - self.attitude )
        # (alpha=0 이면 갱신 없음 = 절제 실험)
        raise NotImplementedError("TODO: 규칙을 채우세요")
        # =============================================================
        self.attitude = float(np.clip(self.attitude, -1.0, 1.0))


class SelectiveExposure(mesa.Model):
    def __init__(self, n=200, tau=0.3, alpha=0.1, n_msg=7, seed=None):
        super().__init__(seed=seed)
        self.tau, self.alpha = tau, alpha
        self.messages = list(np.linspace(-1, 1, n_msg))
        for _ in range(n):
            Reader(self, self.random.uniform(-0.3, 0.3))   # 초기 태도 = 중앙 근처 약한 성향
        self.datacollector = mesa.DataCollector(model_reporters={"polarization": polarization})

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    print("[서명] 선택적 노출 -> 태도 양극화. tau 낮을수록 강함")
    for tau in (0.5, 0.3, 0.15):
        df = run(SelectiveExposure(tau=tau, alpha=0.1, seed=1), 40)
        print(f"  tau={tau}: 양극화 {df['polarization'].iloc[0]:.2f} -> {df['polarization'].iloc[-1]:.2f}")
    print("[절제] 갱신을 끄면(alpha=0) 서명이 사라져야 한다")
    df0 = run(SelectiveExposure(tau=0.15, alpha=0.0, seed=1), 40)
    print(f"  alpha=0: 양극화 {df0['polarization'].iloc[0]:.2f} -> {df0['polarization'].iloc[-1]:.2f} (변화 없음 = 갱신 항이 원인)")
