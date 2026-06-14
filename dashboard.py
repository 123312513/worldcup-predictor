import streamlit as st
import pandas as pd
import plotly.express as px
import math
import json
from typing import Dict

st.set_page_config(page_title="完整进球预测模型", layout="wide")
st.title("⚽ 竞彩进球数预测模型 v3.2（单文件完整版）")

# ==================== 模型核心函数 ====================

def league_scaler(home: str, away: str, league: str = None) -> Dict[str, float]:
    league = (league or "").lower()
    if any(x in league for x in ["德甲", "bundesliga", "荷甲", "eredivisie", "英超", "premier"]):
        return {"1-2球": 0.25, "2-3球": 0.35, "3-4球": 0.30, "其他": 0.10}
    elif any(x in league for x in ["西甲", "laliga", "意甲", "serie a"]):
        return {"1-2球": 0.30, "2-3球": 0.40, "3-4球": 0.20, "其他": 0.10}
    else:
        return {"1-2球": 0.45, "2-3球": 0.30, "3-4球": 0.15, "其他": 0.10}

def handicap_scaler(initial: float, final: float, trend: str) -> Dict[str, float]:
    adj = {"1-2球": 0.0, "2-3球": 0.0, "3-4球": 0.0, "其他": 0.0}
    if final > initial:
        if trend == "down":
            adj["3-4球"] += 0.10
            adj["2-3球"] -= 0.05
            adj["1-2球"] -= 0.05
        elif trend == "up":
            adj["3-4球"] -= 0.10
            adj["1-2球"] += 0.05
            adj["2-3球"] += 0.05
    elif final < initial:
        if trend == "down":
            adj["1-2球"] += 0.10
            adj["2-3球"] -= 0.05
            adj["3-4球"] -= 0.05
        elif trend == "up":
            adj["1-2球"] -= 0.10
            adj["3-4球"] += 0.05
            adj["2-3球"] += 0.05
    else:
        if trend == "down":
            adj["3-4球"] += 0.05
            adj["1-2球"] -= 0.05
        elif trend == "up":
            adj["1-2球"] += 0.05
            adj["3-4球"] -= 0.05
    for k in adj:
        adj[k] = max(-0.15, min(0.15, adj[k]))
    return adj

def poisson_scaler(lam: float) -> Dict[str, float]:
    def poisson(k): return math.exp(-lam) * (lam**k) / math.factorial(k) if lam > 0 else (1 if k==0 else 0)
    p = {str(i): poisson(i) for i in range(7)}
    p["7+"] = 1 - sum(p[str(i)] for i in range(7))
    return {
        "1-2球": p["1"] + p["2"],
        "2-3球": p["2"] + p["3"],
        "3-4球": p["3"] + p["4"],
        "其他": p["0"] + p["5"] + p["6"] + p["7+"]
    }

def odds_scaler(odds: Dict[str, float]) -> Dict[str, float]:
    implied = {}
    for k, v in odds.items():
        if v and v > 0:
            implied[k] = 1.0 / v
    interval = {
        "1-2球": implied.get("1", 0) + implied.get("2", 0),
        "2-3球": implied.get("2", 0) + implied.get("3", 0),
        "3-4球": implied.get("3", 0) + implied.get("4", 0),
        "其他": implied.get("0", 0) + implied.get("5", 0) + implied.get("6", 0) + implied.get("7+", 0),
    }
    total = sum(interval.values())
    if total > 0:
        interval = {k: v/total for k, v in interval.items()}
    return interval

def weighted_fusion(league, handicap_adj, poisson, odds, weights=None):
    if weights is None:
        weights = {"league": 0.25, "handicap": 0.25, "poisson": 0.30, "odds": 0.20}
    adjusted = {k: league[k] + handicap_adj.get(k, 0) for k in league}
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v/total for k, v in adjusted.items()}
    fusion = {}
    for k in adjusted:
        fusion[k] = (weights["league"] * adjusted[k] +
                     weights["handicap"] * handicap_adj.get(k, 0) +
                     weights["poisson"] * poisson.get(k, 0) +
                     weights["odds"] * odds.get(k, 0))
    total = sum(fusion.values())
    if total > 0:
        fusion = {k: v/total for k, v in fusion.items()}
    return fusion

def gamma_correction(base_lam, stage, win_must, goal_diff_urgent, already_qualified):
    stage_map = {"group1": 0.0, "group2": 0.0, "group3": -0.2, "knockout": -0.15}
    gamma = stage_map.get(stage, 0.0)
    if win_must: gamma += 0.15
    if goal_diff_urgent: gamma += 0.20
    if already_qualified: gamma -= 0.10
    return base_lam + gamma

def traffic_light(probs):
    p23 = probs.get("2-3球", 0)
    if p23 >= 0.45: return "green"
    elif p23 >= 0.35: return "yellow"
    else: return "red"

# ==================== 界面交互 ====================

st.sidebar.header("比赛参数")
home = st.sidebar.text_input("主队", "Germany")
away = st.sidebar.text_input("客队", "Curacao")
league = st.sidebar.text_input("联赛（可选）", "世界杯")
stage = st.sidebar.selectbox("阶段", ["group1", "group2", "group3", "knockout"])
goal_diff = st.sidebar.checkbox("需刷净胜球")
must_win = st.sidebar.checkbox("必须赢球")
initial_over = st.sidebar.number_input("初盘大小球", value=2.5, step=0.25)
final_over = st.sidebar.number_input("临场大小球", value=2.5, step=0.25)
water_trend = st.sidebar.selectbox("水位趋势", ["stable", "up", "down"])
xg_home = st.sidebar.number_input("主队xG/90", value=1.8, step=0.1)
xg_away = st.sidebar.number_input("客队xG/90", value=1.2, step=0.1)
odds_json = st.sidebar.text_area("赔率JSON", '{"2":3.3, "3":3.65}', help='示例: {"2":3.3, "3":3.65}')

if st.sidebar.button("开始预测"):
    base_lam = (xg_home + xg_away) / 2
    lam = gamma_correction(base_lam, stage, must_win, goal_diff, False)
    league_probs
