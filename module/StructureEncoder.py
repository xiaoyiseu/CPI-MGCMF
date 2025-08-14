from datetime import datetime
import numpy as np

class StructureDataEncoder:
    def __init__(self):
        self.gender_map = {'男': 1, '女': 2}
        self.arrival_map = {'步入': 1, '扶走': 1, '轮椅': 2}
        self.age_map = {range(18, 66): 1, range(66, 80): 2, range(80, 200): 3, range(0, 18): 4}  # 年龄分组

    def check_empty(self, value):
        return value in {'空值', '', np.nan}

    def Gender(self, gender):
        return self.gender_map.get(gender, 0)

    def Age(self, birth):
        if self.check_empty(birth):
            return 0
        try:
            birth_d = datetime.strptime(birth, "%Y/%m/%d" if "/" in birth else "%Y-%m-%d").date()
            age = datetime.now().year - birth_d.year
            for age_range, category in self.age_map.items():
                if age in age_range:
                    return category
            return 0
        except ValueError:
            return 0

    def Arr_way(self, arr_way):
        # Check if arr_way is a valid string before using 'in'
        if isinstance(arr_way, str) and any(code in arr_way for code in ['120', '999']):
            return 3
        else:
            return self.arrival_map.get(arr_way, 0)

    def check_range(self, value, low, high):
        try:
            return low <= float(value) <= high
        except ValueError:
            return False

    def Temperature(self, temp):
        return 1 if not self.check_empty(temp) and self.check_range(temp, 36, 37) else 2 if temp else 0

    def Pulse(self, pulse):
        return 1 if not self.check_empty(pulse) and self.check_range(pulse, 60, 100) else 2 if pulse else 0

    def Respiration(self, resp):
        return 1 if not self.check_empty(resp) and self.check_range(resp, 12, 20) else 2 if resp else 0

    def BloodPressure(self, bp):
        if self.check_empty(bp) or '/' not in bp:
            return 0
        try:
            h, l = (float(x) if x.isdigit() else None for x in bp.split('/'))
            return self.bp_state(h, l)
        except ValueError:
            return 0

    def bp_state(self, h, l):
        # 使用字典映射简化逻辑
        thresholds = {
            (90, 140, 60, 89): 1,
            (140, 159, 90, 99): 2,
            (160, 179, 100, 109): 3,
            (180, 200, 110, 200): 4
        }
        if h and l:
            for (high_h, high_l, low_h, low_l), state in thresholds.items():
                if self.check_range(h, high_h, high_l) and self.check_range(l, low_h, low_l):
                    return state
        return 0

    def SpO2(self, spo2):
        return 1 if not self.check_empty(spo2) and self.check_range(spo2, 95, 100) else 2 if spo2 else 0
