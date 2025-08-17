import itertools
import json
import csv
import multiprocessing
from enum import Enum
from collections import Counter
from math import factorial

class DiceRule(Enum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    CHOICE = 6
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    YACHT = 11

def calculate_score(dice, rule):
        if not dice: 
            return 0
        counts = {i: dice.count(i) for i in range(1, 7)}
        
        if rule <= 5: 
            return counts[rule + 1] * (rule + 1) * 1000
        if rule == DiceRule.CHOICE: 
            return sum(dice) * 1000
        if rule == DiceRule.FOUR_OF_A_KIND: 
            return sum(dice) * 1000 if any(c >= 4 for c in counts.values()) else 0
        if rule == DiceRule.FULL_HOUSE:
            vals = counts.values()
            return sum(dice) * 1000 if (3 in vals and 2 in vals) or 5 in vals else 0
        unique_dice_str = "".join(map(str, sorted(list(set(dice)))))
        if rule == DiceRule.SMALL_STRAIGHT: 
            return 15000 if "1234" in unique_dice_str or "2345" in unique_dice_str or "3456" in unique_dice_str else 0
        if rule == DiceRule.LARGE_STRAIGHT: 
            return 30000 if "12345" in unique_dice_str or "23456" in unique_dice_str else 0
        if rule == DiceRule.YACHT: 
            return 50000 if 5 in counts.values() else 0
        return 0

# -------------------------------------------------------------------
# 1. 초기 설정 및 데이터 로딩
# -------------------------------------------------------------------

# --- 상수 정의 ---
NUM_ROUNDS = 13
NUM_RULES = 12
UPPER_SCORE_LIMIT = 106 # 0~63점까지 상태로 관리
ALL_DICE_FACES = range(1, 7)

# --- 주사위 조합 및 확률 미리 계산 ---
ALL_5_DICE_COMPOSITIONS = list(set(
    tuple(sorted(c)) for c in itertools.product(ALL_DICE_FACES, repeat=5)
)) # 252가지 고유 조합

def multinomial_coefficient(n, k_counts):
    """다항 계수 계산: n! / (k1! * k2! * ...)"""
    denom = 1
    for k in k_counts:
        denom *= factorial(k)
    return factorial(n) // denom

DICE_PROBABILITIES = {}
for comp in ALL_5_DICE_COMPOSITIONS:
    counts = Counter(comp)
    num_cases = multinomial_coefficient(5, counts.values())
    # 각 주사위 조합이 나올 확률 = (조합의 경우의 수) * (1/6)^5
    DICE_PROBABILITIES[comp] = num_cases / (6**5)

print(f"주사위 5개 고유 조합 수: {len(ALL_5_DICE_COMPOSITIONS)}")
print(f"계산된 확률 총합: {sum(DICE_PROBABILITIES.values())}") # 1.0에 근접해야 함

# -------------------------------------------------------------------
# 2. 핵심 로직: EV 테이블 계산
# -------------------------------------------------------------------

# EV 테이블 초기화: EV[라운드][마스크][기본점수]
# 라운드는 0~13까지 (14개)
EV = [[[0.0 for _ in range(UPPER_SCORE_LIMIT)] for _ in range(1 << NUM_RULES)] for _ in range(NUM_ROUNDS + 1)]

def get_available_rules(mask):
    """비트마스크로부터 사용 가능한 규칙 리스트를 반환"""
    return [rule for rule in range(NUM_RULES) if not (mask & (1 << rule))]

def find_best_action_value(total_dices, current_mask, current_upper_score, next_round_ev_table):
    """
    주어진 주사위(10개)와 상태에서, "즉시점수 + 미래가치"가 최대가 되는 행동의 가치를 찾음
    """
    max_action_value = -1.0
    
    available_rules = get_available_rules(current_mask)
    
    for dice_to_use in set(itertools.combinations(total_dices, 5)):
        dice_to_use = tuple(sorted(dice_to_use))
        
        for rule in available_rules:
            immediate_score = calculate_score(dice_to_use, rule)
            
            # --- 다음 상태 결정 ---
            next_mask = current_mask | (1 << rule)
            next_upper_score = current_upper_score
            is_upper_rule = rule < 6 # ONE~SIX 규칙인지 확인
            
            if is_upper_rule:
                dice_sum = sum(d for d in dice_to_use if d == (rule + 1))
                next_upper_score += dice_sum
            
            # 보너스 점수 처리
            bonus_score = 0
            if current_upper_score < 63 or next_upper_score >= 63:
                bonus_score = 35000
            
            next_upper_score = min(next_upper_score, UPPER_SCORE_LIMIT - 1)
            
            future_value = next_round_ev_table[next_mask][next_upper_score]

            # --- 현재 행동의 총 가치 ---
            current_action_value = immediate_score + bonus_score + future_value
            
            if current_action_value > max_action_value:
                max_action_value = current_action_value
                
    return max_action_value if max_action_value != -1.0 else 0.0

# -------------------------------------------------------------------
# 3. 메인 루프: 라운드를 거꾸로 진행
# -------------------------------------------------------------------

def calculate_ev_for_state(args):
    """하나의 상태에 대한 계산을 수행하는 워커 함수"""
    r, mask, next_round_ev_table = args
    
    ev_for_this_state = [0.0] * UPPER_SCORE_LIMIT
    
    # 불가능한 상태 건너뛰기
    if (NUM_RULES - bin(mask).count('1')) != (NUM_ROUNDS - 1 - r):
        return mask, ev_for_this_state

    # upper_score에 대한 루프
    for upper_score in range(UPPER_SCORE_LIMIT):
        total_expected_value = 0.0
        
        for dice_a in ALL_5_DICE_COMPOSITIONS:
            prob_a = DICE_PROBABILITIES[dice_a]

            ev_a = find_best_action_value(dice_a, mask, upper_score, next_round_ev_table)
            total_expected_value += (prob_a * ev_a)
        
        ev_for_this_state[upper_score] = total_expected_value
        
    return mask, ev_for_this_state

def run_precomputation():
    """사전 계산 메인 함수"""
    # 라운드 12 (마지막에서 두 번째) 부터 0 (첫 번째) 까지 거꾸로 진행
    for r in range(NUM_ROUNDS - 2, -1, -1):
        print(f"Calculating EV for Round {r}...")
        
        # 처리해야 할 작업 목록 생성
        tasks = [(r, mask, EV[r+1]) for mask in range(1 << NUM_RULES)]
        
        # CPU 코어 수만큼 프로세스 풀 생성
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            
            # pool.map을 이용해 작업을 여러 코어에 분산
            # 각 task가 calculate_ev_for_mask 함수에 인자로 전달됨
            results = pool.map(calculate_ev_for_state, tasks)
            
            # 분산 처리된 결과를 다시 EV 테이블에 합치기
            for mask, ev_values in results:
                EV[r][mask] = ev_values

    # -------------------------------------------------------------------
    # 4. 결과 저장
    # -------------------------------------------------------------------
    print("Saving EV table to ev_table.json...")
    with open("ev_table.json", "w") as f:
        json.dump(EV, f)
    print("Pre-computation finished.")

if __name__ == "__main__":
    run_precomputation()