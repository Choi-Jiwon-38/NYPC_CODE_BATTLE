from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import combinations

import sys


# 가능한 주사위 규칙들을 나타내는 enum
class DiceRule(Enum):
    ONE, TWO, THREE, FOUR, FIVE, SIX, CHOICE, FOUR_OF_A_KIND, FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT, YACHT = range(12)

# 각 규칙의 평균 기대 점수 (전략의 핵심)
AVERAGE_SCORES = {
    DiceRule.ONE: 2000, DiceRule.TWO: 5500, DiceRule.THREE: 8500,
    DiceRule.FOUR: 12000, DiceRule.FIVE: 15500, DiceRule.SIX: 19000,
    DiceRule.CHOICE: 22000, DiceRule.FOUR_OF_A_KIND: 13000, DiceRule.FULL_HOUSE: 22500,
    DiceRule.SMALL_STRAIGHT: 15000, DiceRule.LARGE_STRAIGHT: 30000, DiceRule.YACHT: 17000,
}

# 점수를 버려야 할 때의 우선순위
SACRIFICE_PRIORITY = [
    DiceRule.YACHT, DiceRule.LARGE_STRAIGHT, DiceRule.FOUR_OF_A_KIND, DiceRule.FULL_HOUSE,
    DiceRule.SMALL_STRAIGHT, DiceRule.ONE, DiceRule.TWO, DiceRule.THREE,
    DiceRule.FOUR, DiceRule.FIVE, DiceRule.SIX, DiceRule.CHOICE,
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

    # ================================ [필수 구현] ================================
    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        # NOTE: 점수 획득 로직을 우선 개선하기 위하여 임시로 A를 0원에 입찰하도록 고정
        return Bid('A', 0)

    # 효율성 기반 점수 획득 로직은 그대로 유지
    def calculate_put(self) -> DicePut:
        best_put = []
        max_utility = -1.0
        
        dice_pool = self.my_state.dice

        for dice_combination in combinations(dice_pool, 5):
            dice_list = list(dice_combination)
            best_rule, _, utility = self._calculate_best_put_for_dice(dice_list, self.my_state)
            if utility >= max_utility:
                max_utility = utility
                if utility == max_utility:
                    best_put = [DicePut(best_rule, dice_list)]
                else:
                    best_put.append(DicePut(best_rule, dice_list))

        if len(best_put) == 0 or max_utility <= 0.01:
            rule_to_sacrifice = next(r for r in SACRIFICE_PRIORITY if self.my_state.rule_score[r.value] is None)
            dice_to_sacrifice = self.my_state.dice[:5]
            return DicePut(rule_to_sacrifice, dice_to_sacrifice)
        
        if len(best_put) == 1:
            return best_put[0]
        
        # 여러 후보 중 중요도 합이 가장 낮은 것을 선택
        importance = self.get_importance_of_numbers()
        def importance_sum(dice_list):
            return sum(importance[val - 1] for val in dice_list)
        best_put.sort(key=lambda put: (importance_sum(put.dice), sum(put.dice)))
        return best_put[0]

    # ============================== [필수 구현 끝] ==============================

    def get_importance_of_numbers(self) -> List[int]:
        importance_of_numbers = [1, 1.2, 1.4, 1.6, 1.8, 2] # 높은 숫자일수록 기본적으로 중요도가 높음
        _rule_score = self.my_state.rule_score
        _dice_count = [self.my_state.dice.count(i) for i in range(6)]

        # 현재 보유 중인 숫자가 많은 경우에는 최대한 사용하지 않도록 함
        for num in range(6):
            importance_of_numbers[num] *= (1 + 0.1 * _dice_count[num])

        # 기본 점수 규칙(ONE ~ SIX)을 만족하지 못한 경우에는 해당 숫자의 중요도를 올림.
        for num in range(6):
            if _rule_score[num] is None:
                importance_of_numbers[num] * 1.2

    def _calculate_best_put_for_dice(self, dice: List[int], state: 'GameState') -> Tuple[Optional[DiceRule], int, float]:
        best_rule, best_score, max_utility = None, -1, -1.0

        for rule in DiceRule:
            if state.rule_score[rule.value] is None:
                score = GameState.calculate_score(DicePut(rule, dice))
                if score == 0 and rule not in [DiceRule.YACHT, DiceRule.LARGE_STRAIGHT]: continue
                utility = score / AVERAGE_SCORES.get(rule, 1) if AVERAGE_SCORES.get(rule, 1) > 0 else 0

                if rule.value <= 5:
                    basic_score = sum(s for i, s in enumerate(state.rule_score) if s and i <= 5)
                    if basic_score < 63000 and basic_score + score >= 63000:
                         utility *= 2.0
                
                if utility > max_utility:
                    max_utility, best_rule, best_score = utility, rule, score
        
        return best_rule, best_score, max_utility

    def update_get(self, dice_a: List[int], dice_b: List[int], my_bid: Bid, opp_bid: Bid, my_group: str):
        self.round += 1
        if my_group == "A": self.my_state.add_dice(dice_a); self.opp_state.add_dice(dice_b)
        else: self.my_state.add_dice(dice_b); self.opp_state.add_dice(dice_a)
        self.my_state.bid(my_bid.group == my_group, my_bid.amount)
        self.opp_state.bid(opp_bid.group == ('B' if my_group == 'A' else 'A'), opp_bid.amount)

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