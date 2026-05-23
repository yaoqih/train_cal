 - positive：64/64 全解
  - online：6/6 全解
  - truth：125/127，剩 2 个未解

  分布统计：

  - positive
      - 勾数：min 2 / p25 10 / median 15 / p75 79 / max 186 / mean 43.44
      - 时间(s)：min 0.01 / p25 0.07 / median 0.30 / p75 24.44 / max 68.79 / mean 14.45
  - truth
      - 勾数：min 6 / p25 38 / median 54 / p75 145 / max 323 / mean 92.50
      - 时间(s)：min 0.11 / p25 3.27 / median 20.18 / p75 28.95 / max 96.72 / mean 19.57
  - online
      - 勾数：min 11 / p25 40 / median 65 / p75 80 / max 161 / mean 74.00
      - 时间(s)：min 1.18 / p25 21.33 / median 23.98 / p75 26.61 / max 90.03 / mean 32.12

  未解分析：

  1. validation_2025_09_08_noon.json
      - 在标准批跑里是 timeout
      - 我单独用更大预算试跑：beam_width=12, timeout=180s
      - 结果：可解
      - 说明这不是口径问题，也不是明显不可解，而是当前批跑预算下的搜索耗时问题
  2. validation_2025_09_09_noon.json
      - 标准批跑：no solution within budget
      - 单独加预算后仍未解
      - 场景规模更大：99 车，20 个 大库:RANDOM，17 个 存4北
      - 部分解已做到 69 hooks，且 partial_is_valid=true
      - 但仍残留：
          - area_random_unfinished_count=3
          - work_position_unfinished_count=2
          - front_blocker_count=3
          - goal_track_blocker_count=1
          - loco_carry_count=4
      - 这说明它不是单纯时间不够，更像是当前搜索在“大库随机分配 + 存4北集中目标 + 前部清障/顺序债务”组合上缺少更强的建设性收敛策略
