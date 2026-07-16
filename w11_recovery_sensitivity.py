"""11주 · 모형을 믿는 법: 모수 회복 + 민감도 (학부형).

핵심: 아는 값(lr, tau)으로 자료를 만든 뒤 '되찾을' 수 있어야, 실제 자료에서 얻은 추정도 믿는다.
  1) 학습 행위자(9주)로 참 (lr, tau)에서 선택열을 생성.
  2) 격자탐색으로 (lr, tau)를 회복 = 관측 선택의 로그우도를 최대화.
  3) 회복값 ≈ 참값이면 그 자료로는 모수가 식별된다.
  4) 민감도: 여러 seed에서 회복이 안정적인가.

Mesa 없이 numpy만으로 한 행위자를 다룬다(회복은 '적합' 절차).
실행: python w11_recovery_sensitivity.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 같은 폴더 abmlib 임포트 보장
from abmlib import plot_hist

TRUE_MEANS = (0.2, 0.5, 0.8)
K = len(TRUE_MEANS)


def softmax(values, tau):
    v = np.asarray(values, dtype=float) / tau
    v -= v.max()
    e = np.exp(v)
    return e / e.sum()


def simulate(lr, tau, seed, noise=0.2, T=250):
    """참 (lr, tau)에서 한 행위자의 선택·보상 열을 만든다."""
    rng = np.random.default_rng(seed)
    Q = np.zeros(K)
    choices, rewards = [], []
    for _ in range(T):
        p = softmax(Q, tau)
        c = int(rng.choice(K, p=p))
        r = TRUE_MEANS[c] + rng.normal(0, noise)
        Q[c] += lr * (r - Q[c])
        choices.append(c)
        rewards.append(r)
    return np.array(choices), np.array(rewards)


def loglik(lr, tau, choices, rewards):
    """같은 보상열로 신념 갱신을 재생하며 관측 선택의 로그우도 합."""
    Q = np.zeros(K)
    ll = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, tau)
        ll += np.log(p[c] + 1e-12)
        Q[c] += lr * (r - Q[c])
    return ll


def recover(choices, rewards, lrs, taus):
    """격자탐색: 로그우도를 최대로 하는 (lr, tau)."""
    best = (-np.inf, None, None)
    for lr in lrs:
        for tau in taus:
            ll = loglik(lr, tau, choices, rewards)
            if ll > best[0]:
                best = (ll, lr, tau)
    return best[1], best[2]


if __name__ == "__main__":
    true_lr, true_tau = 0.30, 0.20
    lrs = np.round(np.arange(0.05, 0.61, 0.05), 2)
    taus = np.round(np.arange(0.05, 0.51, 0.05), 2)

    print(f"[모수 회복] 참값 lr={true_lr}, tau={true_tau} — 되찾는가?")
    ch, rw = simulate(true_lr, true_tau, seed=1)
    rec_lr, rec_tau = recover(ch, rw, lrs, taus)
    print(f"  회복값 lr={rec_lr}, tau={rec_tau}  (참값 근처면 식별 가능)")

    print("[민감도] 20개 seed에서 회복 분포 — 안정적인가?")
    rec = [recover(*simulate(true_lr, true_tau, seed=s), lrs, taus) for s in range(20)]
    rls = np.array([r[0] for r in rec])
    rts = np.array([r[1] for r in rec])
    print(f"  lr 회복: 평균 {rls.mean():.2f} (참 {true_lr}) 범위 {rls.min()}~{rls.max()}")
    print(f"  tau 회복: 평균 {rts.mean():.2f} (참 {true_tau}) 범위 {rts.min()}~{rts.max()}")
    print("  → 되찾히면 추정을 믿을 근거. 넓게 흩어지면(비식별) 그 자료로는 못 믿는다.")
    plot_hist(rls, "w11_recover_lr.png", f"recovered lr (true={true_lr})")
    print("  그림 저장: w11_recover_lr.png")
