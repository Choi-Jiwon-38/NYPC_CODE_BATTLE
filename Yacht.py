from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import product, combinations
from collections import Counter
from math import factorial
from gzip import open
from pickle import load

import sys

DEBUG_MODE = False

# 가능한 주사위 규칙들을 나타내는 enum
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


# ==================================================================== #

LOW_UTILITY = 0.01        # 효율성이 이 이하이면 SACRIFICE
BID_DIFF_RATE = 0.02      # 경매 입찰 시 입찰 점수 보정 비율
NUM_RULES = 12
UPPER_SCORE_LIMIT = 64    # 0~63점까지 상태로 관리
FUTURE_RATE = 1.0

# ==================================================================== #

# 입찰 방법을 나타내는 데이터클래스
@dataclass
class Bid:
    group: str  # 입찰 그룹 ('A' 또는 'B')
    amount: int  # 입찰 금액


# 주사위 배치 방법을 나타내는 데이터클래스
@dataclass
class DicePut:
    rule: DiceRule  # 배치 규칙
    dice: List[int]  # 배치할 주사위 목록


# 게임 상태를 관리하는 클래스
class Game:
    def __init__(self):
        self.my_state = GameState() # 내 팀의 현재 상태
        self.opp_state = GameState() # 상대 팀의 현재 상태
        self.round = 0
        # 상대방 베팅 히스토리 (최근 10개만 유지)
        self.opp_bid_history = []
        # 기댓값 테이블 및 구조체
        self.ev_table = None
        self.ev_state = EvState()
        # 주사위 조합 및 확률 미리 계산
        self.ALL_5_DICE_COMPOSITIONS = list(set(
            tuple(sorted(c)) for c in product(range(1, 7), repeat=5)
        ))
        self.DICE_PROBABILITIES = {}
        for comp in self.ALL_5_DICE_COMPOSITIONS:
            counts = Counter(comp)
            num_cases = self.multinomial_coefficient(5, counts.values())
            # 각 주사위 조합이 나올 확률 = (조합의 경우의 수) * (1/6)^5
            self.DICE_PROBABILITIES[comp] = num_cases / (6 ** 5)

        # 테이블 초기화
        try:
            with open('data.bin', 'rb') as f:
                self.ev_table = load(f)
        except FileNotFoundError:
            print("open() failed.", file=sys.stderr)

    # ================================ [필수 구현] ================================

    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        group_a, group_b = dice_a + self.my_state.dice, dice_b + self.my_state.dice
        best_put_a, ev_a = self.calculate_ev(group_a, self.ev_state)
        best_put_b, ev_b = self.calculate_ev(group_b, self.ev_state)

        # 더 높은 효율을 가진 그룹에 입찰
        if best_put_a is not None and ev_a > ev_b:
            group = "A"
            score = ev_a
        elif best_put_b is not None and ev_b > ev_a:
            group = "B"
            score = ev_b
        else:
            # dice_a, dice_b의 우열이 없으면, 전체 다이스에서
            # 숫자들이 높은 그룹에 입찰
            importance_a = sum(group_a)
            importance_b = sum(group_b)
            group = "A" if importance_a >= importance_b else "B"
            score = 0

        score_diff = int(abs(ev_a - ev_b) * BID_DIFF_RATE)

        # TODO: 승기를 잡고있다는 기준을 현재 점수가 아닌 기대 점수에 따른 방향으로 변경해야함.
        # 상대방 히스토리 기반 베팅 금액 계산 
        # + 기대 점수의 차이 값으로 보정
        if self.opp_bid_history:
            # 상대방 베팅 히스토리 분석
            max_opp_bid = 1 if max(self.opp_bid_history) == 0 else max(self.opp_bid_history)
            sorted_bids = sorted(self.opp_bid_history, reverse=True)
            top_3_avg = 1 if sum(sorted_bids[1:4]) / min(3, len(sorted_bids)) == 0 else sum(sorted_bids[1:4]) / min(3, len(sorted_bids))
            avg_opp_bid = 1 if sum(self.opp_bid_history) / len(self.opp_bid_history) == 0 else sum(self.opp_bid_history) / len(self.opp_bid_history)

            # 점수에 따른 베팅 전략
            if score >= 50000:  # Yacht - 가장 공격적
                amount = int(max_opp_bid * 1.2)  # 최대 베팅의 1.2배
            elif score >= 30000:  # Large Straight - 매우 공격적
                amount = int(top_3_avg * 1.1)  # 상위 3개 평균의 1.1배
            elif score >= 15000:  # Small Straight - 공격적
                amount = int(top_3_avg * 1.05)  # 상위 3개 평균의 1.05배
            elif score >= 10000:  # 높은 기본 점수 - 적당히 공격적
                amount = int(avg_opp_bid * 1.1)  # 평균 베팅의 1.1배
            else:
                amount = int(avg_opp_bid * 0.5)  # 평균 베팅의 0.5배
            
            # 보정 값 추가
            amount += score_diff
        else:
            amount = 0
        
        if DEBUG_MODE: print(f"{self.ev_state.round+1}R, BID CACL END -> a: {ev_a}, b: {ev_b}, score_diff: {score_diff}, amount: {amount}", file=sys.stderr) # 디버깅용
        return Bid(group, amount)
    
    def calculate_put(self) -> DicePut:
        best_put = []
        dice_pool = self.my_state.dice
        
        if DEBUG_MODE: print(f"{self.ev_state.round+1}R, CALC START -> dice pool: {sorted(dice_pool)}, state: {self.ev_state.upper_score}", file=sys.stderr) # 디버깅용

        best_put, _ = self.calculate_ev(dice_pool, self.ev_state)
        self.ev_state = self.ev_state.get_next_state(best_put)

        if DEBUG_MODE: print(f"{self.ev_state.round+1}R, CALC START -> dice pool: {sorted(dice_pool)}, state: {self.ev_state.upper_score}", file=sys.stderr) # 디버깅용

        return best_put
    
    # ============================== [필수 구현 끝] ==============================

    def multinomial_coefficient(self, n, k_counts):
        """다항 계수 계산: n! / (k1! * k2! * ...)"""
        denom = 1
        for k in k_counts:
            denom *= factorial(k)
        return factorial(n) // denom
    
    def calculate_ev(self, dice_pool: List[int], state: 'EvState') -> Tuple[Optional[DicePut], float]:
        best_put, max_ev = None, -1.0
        next_round_ev_table = self.ev_table[state.round + 1]
        all_rules = list(DiceRule)

        unique_combination = {tuple(sorted(comb)) for comb in combinations(dice_pool, 5)}
        # 모든 규칙에 대해 점수를 계산
        for dice in unique_combination:
            for rule in all_rules:
                if self.my_state.rule_score[rule.value] is None:
                    current_put = DicePut(rule, dice)
                    score = self.my_state.calculate_score(current_put)
                    next_state = state.get_next_state(current_put)

                    if state.upper_score < 63 and next_state.upper_score >= 63:
                        bonus_score = 35000
                    else:
                        bonus_score = 0

                    # 미래 가치 계산
                    future_value = (next_round_ev_table[next_state.mask][next_state.upper_score] * 100000 * FUTURE_RATE)

                    current_ev = score + bonus_score + future_value
                    if DEBUG_MODE: print(f"{self.ev_state.round+1}R, rule: {rule}, dice: {sorted(dice)}, current_ev: {current_ev}", file=sys.stderr)
                    if current_ev > max_ev:
                        max_ev = current_ev
                        best_put = DicePut(rule, dice)
        
        if DEBUG_MODE: print(f"{self.ev_state.round+1}R, calculate_ev() END -> rule: {best_put.rule.name}, dice: {sorted(best_put.dice)}, max_ev: {max_ev}", file=sys.stderr)
        return best_put, max_ev


    def update_get(self, dice_a: List[int], dice_b: List[int], my_bid: Bid, opp_bid: Bid, my_group: str):
        self.round += 1

        # 기존 상태 업데이트
        if my_group == "A": 
            self.my_state.add_dice(dice_a)
            self.opp_state.add_dice(dice_b)
        else: 
            self.my_state.add_dice(dice_b)
            self.opp_state.add_dice(dice_a)

        # 베팅 결과 반영
        self.my_state.bid(my_bid.group == my_group, my_bid.amount)
        self.opp_state.bid(opp_bid.group == ('B' if my_group == 'A' else 'A'), opp_bid.amount)

        # 상대방 베팅 히스토리 업데이트 (최근 10개만 유지)
        self.opp_bid_history.append(opp_bid.amount)
        if len(self.opp_bid_history) > 10:
            self.opp_bid_history.pop(0)

    def update_put(self, put: DicePut): 
        self.my_state.use_dice(put)
    def update_set(self, put: DicePut): 
        self.opp_state.use_dice(put)

class EvState:
    def __init__(self, mask=0, upper_score=0, round=0, rule_scores=None):
        self.mask = mask
        self.upper_score = upper_score
        self.round = round

    def get_next_state(self, put: DicePut):
        """
        주어진 DicePut 행동을 수행했을 때의 '다음' GameState 객체를 반환합니다.
        """
        rule_value = put.rule.value
        
        # 다음 상태의 mask 계산
        next_mask = self.mask | (1 << rule_value)
        
        # 다음 상태의 upper_score 계산
        next_upper_score = self.upper_score
        if rule_value < 6:
            dice_sum = sum(d for d in put.dice if d == (rule_value + 1))
            next_upper_score += dice_sum
        
        next_upper_score = min(next_upper_score, UPPER_SCORE_LIMIT - 1)
            
        return EvState(
            mask=next_mask,
            upper_score=next_upper_score,
            round=self.round + 1,
        )

class GameState:
    def __init__(self):
        self.dice: List[int] = []
        self.rule_score: List[Optional[int]] = [None] * 12
        self.bid_score = 0

    def get_total_score(self) -> int:
        basic = sum(s for s in self.rule_score[0:6] if s is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(s for s in self.rule_score[6:12] if s is not None)
        return basic + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int): 
        self.bid_score += -amount if is_successful else amount
    def add_dice(self, new_dice: List[int]): 
        self.dice.extend(new_dice)
    def use_dice(self, put: DicePut):
        if put.rule is None: return
        assert self.rule_score[put.rule.value] is None
        for d in put.dice:
            if d in self.dice: self.dice.remove(d)
        self.rule_score[put.rule.value] = self.calculate_score(put)

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        rule, dice = put.rule, sorted(put.dice)
        if not dice: 
            return 0
        counts = {i: dice.count(i) for i in range(1, 7)}
        
        if rule.value <= 5: 
            return counts[rule.value + 1] * (rule.value + 1) * 1000
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

def main():
    game = Game()

    # 입찰 라운드에서 나온 주사위들
    dice_a, dice_b = [0] * 5, [0] * 5
    # 내가 마지막으로 한 입찰 정보
    my_bid = Bid("", 0)

    while True:
        try:
            line = input().strip()
            if not line:
                continue

            command, *args = line.split()

            if command == "READY":
                # 게임 시작
                print("OK")
                continue

            if command == "ROLL":
                # 주사위 굴리기 결과 받기
                str_a, str_b = args
                for i, c in enumerate(str_a):
                    dice_a[i] = int(c)  # 문자를 숫자로 변환
                for i, c in enumerate(str_b):
                    dice_b[i] = int(c)  # 문자를 숫자로 변환

                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}")
                continue

            if command == "GET":
                # 주사위 받기
                get_group, opp_group, opp_score = args
                opp_score = int(opp_score)
                game.update_get(
                    dice_a, dice_b, my_bid, Bid(opp_group, opp_score), get_group
                )
                continue

            if command == "SCORE":
                # 주사위 골라서 배치하기
                put = game.calculate_put()
                game.update_put(put)
                assert put.rule is not None
                print(f"PUT {put.rule.name} {''.join(map(str, put.dice))}")
                continue

            if command == "SET":
                # 상대의 주사위 배치
                rule, str_dice = args
                dice = [int(c) for c in str_dice]
                game.update_set(DicePut(DiceRule[rule], dice))
                continue

            if command == "FINISH":
                # 게임 종료
                break

            # 알 수 없는 명령어 처리
            print(f"Invalid command: {command}", file=sys.stderr)
            sys.exit(1)

        except EOFError:
            break


if __name__ == "__main__":
    main()