# ===== 빈칸판(학생 배포용) =====
# TODO 블록을 채우고 raise 줄을 지우면, 완성본과 같은 서명이 나와야 합니다.
# 검증: 극단값 시험 + 씨앗 73 재실행 + (가능하면) 손 검산. 검증 로그 한 문단!

"""6주 · 네트워크 위 확산·복합전염 (Centola) (Mesa 3.x + networkx).

이론: 행동·신념은 네트워크 위에서 퍼진다.
  단순 전염(simple): 채택 이웃 1명이면 채택.
  복합 전염(complex): k명 이상 채택 이웃이 있어야 채택(사회적 확증 필요).
서명: 복합 전염은 좁은세상의 '무작위 지름길'이 늘수록 오히려 확산이 무너진다.
      (긴 다리는 폭이 좁아 복합 전염을 못 나른다 ; Centola 2010)
반증: threshold를 올려도 rewire에 무관하게 똑같이 퍼진다.

seed는 인접 블록(0..n_seed)에서 시작한다 = 복합 전염이 나아갈 '넓은 front'.
네트워크는 networkx로 직접 다룬다(Mesa space 불필요). 갱신은 동시(synchronous).
실행: python w06_network_contagion.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import networkx as nx
import mesa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 상위 폴더(materials) abmlib 임포트
from abmlib import run


def adopted_frac(model):
    return float(np.mean([a.adopted for a in model.agents]))


class Node(mesa.Agent):
    def __init__(self, model, node_id, seeded):
        super().__init__(model)
        self.node_id = node_id
        self.adopted = seeded
        self._next = seeded

    def step(self):
        if self.adopted:
            self._next = True
            return
        m = self.model
        # =============================================================
        # TODO(학생): 채택 규칙 (책 6장 §6.3~6.4)
        # 1) 내 이웃(m.G.neighbors(self.node_id)) 중 채택자 수 n_ad 를 센다
        # 2) n_ad 가 문턱(m.threshold) 이상이고 m.random.random() < m.p 이면
        #    self._next = True  (단순 전염 = 문턱 1, 복합 전염 = 문턱 3)
        raise NotImplementedError("TODO: 규칙을 채우세요")
        # =============================================================


class NetworkContagion(mesa.Model):
    def __init__(self, n=400, k_deg=8, rewire=0.0, threshold=1, p=1.0,
                 n_seed=12, seed=None):
        super().__init__(seed=seed)
        self.threshold, self.p = threshold, p
        # 좁은세상 네트워크 (rewire=0 정규격자 ~ rewire=1 무작위)
        self.G = nx.watts_strogatz_graph(n, k_deg, rewire, seed=seed)
        self.node_agent = {}
        for node in self.G.nodes():
            self.node_agent[node] = Node(self, node, node < n_seed)  # 인접 블록 점화
        self.datacollector = mesa.DataCollector(model_reporters={"adopted": adopted_frac})

    def step(self):
        for a in self.agents:      # 1) 다음 상태 계산
            a.step()
        for a in self.agents:      # 2) 동시 반영
            a.adopted = a._next
        self.datacollector.collect(self)


def final_adopted(threshold, rewire, seed=1, steps=60):
    return run(NetworkContagion(threshold=threshold, rewire=rewire, seed=seed),
               steps)["adopted"].iloc[-1]


if __name__ == "__main__":
    print("[서명] 복합 전염은 무작위 지름길(rewire)이 늘수록 무너진다")
    print("  전염유형   rewire=0.0  rewire=0.2  rewire=0.5")
    for name, thr in (("단순(1)", 1), ("복합(3)", 3)):
        r0 = final_adopted(thr, 0.0)
        r2 = final_adopted(thr, 0.2)
        r5 = final_adopted(thr, 0.5)
        print(f"  {name}   {r0:8.2f}  {r2:8.2f}  {r5:8.2f}")
    print("  → 단순은 rewire에 무관. 복합은 격자(0.0)에선 퍼지다 지름길이 늘면 붕괴.")
