from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import combinations

import sys


# 가능한 주사위 규칙들을 나타내는 enum
class DiceRule(Enum):
    ONE, TWO, THREE, FOUR, FIVE, SIX, CHOICE, FOUR_OF_A_KIND, FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT, YACHT = range(12)

# 각 규칙의 평균 기대 점수 (전략의 핵심) - 개선됨
AVERAGE_SCORES = {
    DiceRule.ONE: 2000, DiceRule.TWO: 4000, DiceRule.THREE: 6000,
    DiceRule.FOUR: 8000, DiceRule.FIVE: 10000, DiceRule.SIX: 12000,
    DiceRule.CHOICE: 22000, DiceRule.FOUR_OF_A_KIND: 25000, DiceRule.FULL_HOUSE: 30000,
    DiceRule.SMALL_STRAIGHT: 15000, DiceRule.LARGE_STRAIGHT: 30000, DiceRule.YACHT: 50000,
}

# 점수를 버려야 할 때의 우선순위 (개선됨)
SACRIFICE_PRIORITY = [
    # 가장 먼저 버릴 것들 (낮은 점수 + 높은 확률)
    DiceRule.ONE,      # 1000점 (가장 낮음)
    DiceRule.TWO,      # 2000점
    DiceRule.THREE,    # 3000점
    DiceRule.FOUR,     # 4000점
    
    # 중간 우선순위 (보통 점수)
    DiceRule.FIVE,     # 5000점
    DiceRule.SIX,      # 6000점
    DiceRule.CHOICE,   # 모든 주사위 합 × 1000 (보통 높음)
    
    # 나중에 버릴 것들 (높은 점수 + 낮은 확률)
    DiceRule.SMALL_STRAIGHT,  # 15000점
    DiceRule.FOUR_OF_A_KIND,  # 4개 같음 × 1000
    DiceRule.FULL_HOUSE,      # 3개+2개 × 1000
    DiceRule.LARGE_STRAIGHT,  # 30000점
    DiceRule.YACHT,           # 50000점 (가장 마지막)
]

@dataclass
class Bid:
    group: str
    amount: int

@dataclass
class DicePut:
    rule: DiceRule
    dice: List[int]

class Game:
    def __init__(self):
        self.my_state = GameState()
        self.opp_state = GameState()
        self.round = 0
        # 상대방 베팅 히스토리 (최근 10개만 유지)
        self.opp_bid_history = []

    # ================================ [필수 구현] ================================
    
    def _calculate_bid_amount(self, utility_a: float, utility_b: float, score_diff: int) -> int:
        """상대방 히스토리 기반 베팅 금액 계산 (전략적 개선)"""
        
        # 기본 베팅 금액
        utility_diff = abs(utility_a - utility_b)
        base_amount = int(utility_diff * 3000)
        
        # 상대방 히스토리가 없으면 기본 금액 사용
        if not self.opp_bid_history:
            return min(base_amount, 2000)
        
        # 상대방의 최근 베팅 패턴 분석
        recent_bids = self.opp_bid_history[-5:]  # 최근 5개
        avg_bid = sum(recent_bids) / len(recent_bids)
        max_recent_bid = max(recent_bids)
        min_recent_bid = min(recent_bids)
        
        # 상대방 예상 베팅액 계산
        if len(recent_bids) >= 3:
            # 최근 트렌드 분석
            recent_trend = recent_bids[-1] - recent_bids[-3]  # 최근 3라운드 트렌드
            if recent_trend > 0:  # 베팅액이 증가하는 추세
                expected_bid = max_recent_bid + int(recent_trend * 0.5)
            else:  # 베팅액이 감소하거나 유지
                expected_bid = avg_bid
        else:
            expected_bid = avg_bid
        
        # 전략적 베팅액 결정
        if score_diff < 0:  # 지고 있을 때 - 더 적극적으로
            # 상대방 예상 베팅액보다 최소 10% 더 높게 베팅
            strategic_amount = int(expected_bid * 1.1)
            # 뒤처진 점수 고려
            max_bid = abs(score_diff) + 2000
            amount = min(strategic_amount, max_bid)
        else:  # 이기고 있을 때 - 안정적으로
            # 상대방 예상 베팅액과 비슷하게 베팅
            strategic_amount = int(expected_bid * 0.9)
            amount = min(strategic_amount, 3000)
        
        # 최소 베팅액 보장 (상대방이 우리를 이용하는 것 방지)
        min_bid = max(500, int(avg_bid * 0.8))
        amount = max(amount, min_bid)
        
        return max(0, int(amount))
    
    # [개선됨] 안정적으로 점수를 관리하는 '보수적인 입찰 전략'
    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        # 각 주사위 묶음의 최대 기대 효율(utility)을 계산
        _, _, utility_a = self._calculate_best_put_for_dice(dice_a, self.my_state)
        _, _, utility_b = self._calculate_best_put_for_dice(dice_b, self.my_state)

        # 더 높은 효율을 가진 그룹에 입찰
        group = "A" if utility_a > utility_b else "B"
        
        # 보수적인 입찰 금액 산정
        score_diff = self.my_state.get_total_score() - self.opp_state.get_total_score()

        # 만약 5000점 이상 이기고 있다면, 위험을 감수하지 않고 0을 베팅하여 리드를 지킴
        if score_diff > 5000:
            return Bid(group, 0)
        
        # 상대방 히스토리 기반 베팅 금액 계산
        amount = self._calculate_bid_amount(utility_a, utility_b, score_diff)
        
        return Bid(group, max(0, min(100000, int(amount))))

    def get_dynamic_sacrifice_priority(self) -> List[DiceRule]:
        """현재 상황에 따른 동적 우선순위 계산 (개선됨)"""
        priorities = []
        
        # 기본 점수 규칙들 (ONE~SIX)
        basic_rules = [DiceRule.ONE, DiceRule.TWO, DiceRule.THREE, 
                       DiceRule.FOUR, DiceRule.FIVE, DiceRule.SIX]
        
        # 현재 기본 점수 계산
        current_basic = sum(s for i, s in enumerate(self.my_state.rule_score) if s and i <= 5)
        
        # 보너스 점수(35000)를 고려한 우선순위 (강화됨)
        if current_basic < 63000:
            # 보너스를 얻을 수 있다면 기본 규칙을 절대 버리지 않음
            available_basic = [r for r in basic_rules if self.my_state.rule_score[r.value] is None]
            if available_basic:
                # 기본 규칙을 우선적으로 보존
                priorities.extend(available_basic)
            
            # 조합 규칙들 (높은 점수 순, 보너스 획득 전에는 신중하게)
            combination_rules = [
                DiceRule.SMALL_STRAIGHT, DiceRule.FOUR_OF_A_KIND, 
                DiceRule.FULL_HOUSE, DiceRule.LARGE_STRAIGHT, DiceRule.YACHT
            ]
            priorities.extend([r for r in combination_rules if self.my_state.rule_score[r.value] is None])
        else:
            # 보너스를 이미 얻었다면 기본 규칙을 먼저 버림
            priorities.extend([r for r in basic_rules if self.my_state.rule_score[r.value] is None])
            
            # 조합 규칙들 (높은 점수 순)
            combination_rules = [
                DiceRule.SMALL_STRAIGHT, DiceRule.FOUR_OF_A_KIND, 
                DiceRule.FULL_HOUSE, DiceRule.LARGE_STRAIGHT, DiceRule.YACHT
            ]
            priorities.extend([r for r in combination_rules if self.my_state.rule_score[r.value] is None])
        
        return priorities

    def calculate_put(self) -> DicePut:
        best_put = None
        max_utility = -1.0
        
        dice_pool = self.my_state.dice
        num_to_pick = 5 if len(dice_pool) >= 5 else len(dice_pool)
        if num_to_pick == 0:
             # 동적 우선순위 사용
             sacrifice_priority = self.get_dynamic_sacrifice_priority()
             rule_to_sacrifice = sacrifice_priority[0] if sacrifice_priority else SACRIFICE_PRIORITY[0]
             return DicePut(rule_to_sacrifice, [])

        for dice_combination in combinations(dice_pool, num_to_pick):
            dice_list = list(dice_combination)
            best_rule, _, utility = self._calculate_best_put_for_dice(dice_list, self.my_state)
            if utility > max_utility:
                max_utility = utility
                best_put = DicePut(best_rule, dice_list)

        if best_put is None or max_utility <= 0.01:
            # 동적 우선순위 사용
            sacrifice_priority = self.get_dynamic_sacrifice_priority()
            rule_to_sacrifice = sacrifice_priority[0] if sacrifice_priority else SACRIFICE_PRIORITY[0]
            dice_to_sacrifice = self.my_state.dice[:num_to_pick]
            return DicePut(rule_to_sacrifice, dice_to_sacrifice)
            
        return best_put
    # ============================== [필수 구현 끝] ==============================

    def _calculate_best_put_for_dice(self, dice: List[int], state: 'GameState') -> Tuple[Optional[DiceRule], int, float]:
        best_rule, best_score, max_utility = None, -1, -1.0

        for rule in DiceRule:
            if state.rule_score[rule.value] is None:
                score = GameState.calculate_score(DicePut(rule, dice))
                if score == 0 and rule not in [DiceRule.YACHT, DiceRule.LARGE_STRAIGHT]: continue
                
                # 높은 점수 규칙 보호 로직 (개선됨)
                if rule == DiceRule.YACHT and self.round <= 6:
                    # Yacht는 7라운드 이후에만 사용
                    continue
                if rule == DiceRule.LARGE_STRAIGHT and self.round <= 4:
                    # Large Straight는 5라운드 이후에만 사용
                    continue
                
                utility = score / AVERAGE_SCORES.get(rule, 1) if AVERAGE_SCORES.get(rule, 1) > 0 else 0

                if rule.value <= 5:
                    basic_score = sum(s for i, s in enumerate(state.rule_score) if s and i <= 5)
                    if basic_score < 63000 and basic_score + score >= 63000:
                         utility *= 2.5  # 보너스 점수 가중치 증가
                
                if utility > max_utility:
                    max_utility, best_rule, best_score = utility, rule, score
        
        return best_rule, best_score, max_utility

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
        my_bid_success = my_bid.group == my_group
        self.my_state.bid(my_bid_success, my_bid.amount)
        
        opp_group = 'B' if my_group == 'A' else 'A'
        opp_bid_success = opp_bid.group == opp_group
        self.opp_state.bid(opp_bid_success, opp_bid.amount)
        
        # 상대방 베팅 히스토리 업데이트 (최근 10개만 유지)
        self.opp_bid_history.append(opp_bid.amount)
        if len(self.opp_bid_history) > 10:
            self.opp_bid_history.pop(0)

    def update_put(self, put: DicePut): self.my_state.use_dice(put)
    def update_set(self, put: DicePut): self.opp_state.use_dice(put)

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

    def bid(self, is_successful: bool, amount: int): self.bid_score += -amount if is_successful else amount
    def add_dice(self, new_dice: List[int]): self.dice.extend(new_dice)
    def use_dice(self, put: DicePut):
        if put.rule is None: return
        assert self.rule_score[put.rule.value] is None
        for d in put.dice:
            if d in self.dice: self.dice.remove(d)
        self.rule_score[put.rule.value] = self.calculate_score(put)

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        rule, dice = put.rule, sorted(put.dice)
        if not dice: return 0
        counts = {i: dice.count(i) for i in range(1, 7)}
        
        if rule.value <= 5: return counts[rule.value + 1] * (rule.value + 1) * 1000
        if rule == DiceRule.CHOICE: return sum(dice) * 1000
        if rule == DiceRule.FOUR_OF_A_KIND: return sum(dice) * 1000 if any(c >= 4 for c in counts.values()) else 0
        if rule == DiceRule.FULL_HOUSE:
            vals = counts.values()
            return sum(dice) * 1000 if (3 in vals and 2 in vals) or 5 in vals else 0
        unique_dice_str = "".join(map(str, sorted(list(set(dice)))))
        if rule == DiceRule.SMALL_STRAIGHT: return 15000 if "1234" in unique_dice_str or "2345" in unique_dice_str or "3456" in unique_dice_str else 0
        if rule == DiceRule.LARGE_STRAIGHT: return 30000 if "12345" in unique_dice_str or "23456" in unique_dice_str else 0
        if rule == DiceRule.YACHT: return 50000 if 5 in counts.values() else 0
        return 0

def main():
    game = Game()
    dice_a, dice_b = [0] * 5, [0] * 5
    my_bid = Bid("", 0)
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line: continue
            command, *args = line.split()
            if command == "READY": print("OK"); sys.stdout.flush()
            elif command == "ROLL":
                dice_a, dice_b = [int(c) for c in args[0]], [int(c) for c in args[1]]
                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}"); sys.stdout.flush()
            elif command == "GET": game.update_get(dice_a, dice_b, my_bid, Bid(args[1], int(args[2])), args[0])
            elif command == "SCORE":
                put = game.calculate_put()
                game.update_put(put)
                print(f"PUT {put.rule.name} {''.join(map(str, sorted(put.dice)))}"); sys.stdout.flush()
            elif command == "SET": game.update_set(DicePut(DiceRule[args[0]], [int(c) for c in args[1]]))
            elif command == "FINISH": break
        except (EOFError, IndexError): break
if __name__ == "__main__":
    main()