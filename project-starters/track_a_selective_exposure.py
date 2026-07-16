"""트랙 A · 선택적 노출 — 기말 프로젝트 시작 코드 (self-contained, Mesa 3.x).

요지: 사람은 자기 태도에 맞는 메시지를 고르고, 그 노출이 태도를 강화한다.
ODD-lite: 목적=선택적 노출이 양극화를 낳는가 / 행위자=이용자 / 상태=태도 / 자극=메시지 입장 /
  규칙: 가치=동의도(태도x입장) → softmax 선택 → 태도를 고른 메시지 쪽으로 갱신 /
  시간=매 스텝 전원 1회 / 출력=평균 |태도|(양극화) / 한계=단일 태도 차원.
서명: 태도가 극단으로 벌어진다(tau 낮을수록 강). 절제: alpha=0이면 사라짐.

실행: python track_a_selective_exposure.py
"""
from __future__ import annotations
import numpy as np
import mesa


def softmax(values, tau):
    v = np.asarray(values, float) / tau
    v -= v.max()
    e = np.exp(v)
    return e / e.sum()


class Reader(mesa.Agent):
    def __init__(self, model, attitude):
        super().__init__(model)
        self.attitude = attitude

    def step(self):
        m = self.model
        value = [self.attitude * pos for pos in m.messages]      # 동의도
        idx = m.random.choices(range(len(m.messages)),
                               weights=softmax(value, m.tau))[0]
        self.attitude += m.alpha * (m.messages[idx] - self.attitude)  # 갱신
        self.attitude = float(np.clip(self.attitude, -1, 1))


class SelectiveExposure(mesa.Model):
    def __init__(self, n=200, tau=0.2, alpha=0.1, n_msg=7, seed=None):
        super().__init__(seed=seed)
        self.tau, self.alpha = tau, alpha
        self.messages = list(np.linspace(-1, 1, n_msg))
        for _ in range(n):
            Reader(self, self.random.uniform(-0.3, 0.3))


def polarization(model):
    return float(np.mean([abs(a.attitude) for a in model.agents]))


if __name__ == "__main__":
    m = SelectiveExposure(tau=0.2, alpha=0.1, seed=1)
    print(f"시작 양극화 {polarization(m):.2f}")
    for _ in range(40):
        m.agents.shuffle_do("step")
    print(f"끝 양극화   {polarization(m):.2f}  (오르면 선택적 노출 → 양극화)")

# 확장 아이디어(TODO): 자기 관련성(self-relevance) 차원 추가 / 진보·보수 두 진영 매체 /
#   완고한(update 안 하는) 소수 / 알고리즘 추천(가치에 노출확률 곱) / 2차원 태도.
