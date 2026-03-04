# Key Studies Summary

## Objective 1: Delay analysis and key drivers
- Delay threshold used: **>10 min**
- Overall delay probability: **9.7%**
- On-time reliability: **90.3%**

### Highest-risk weekday/hour slots (Top 5)
- Wednesday 19:00 → delay probability 22.5% (n=2294)
- Wednesday 18:00 → delay probability 19.2% (n=2449)
- Wednesday 17:00 → delay probability 18.2% (n=2378)
- Monday 18:00 → delay probability 18.0% (n=2504)
- Wednesday 20:00 → delay probability 17.6% (n=1941)

### Highest-risk specific-train identifiers/proxies (Top 5)
- VIA @ Wiesbaden Hbf 04:00 → 100.0% (n=1)
- VIA @ Wiesbaden Hbf 03:00 → 100.0% (n=3)
- HLB @ Eltville 21:00 → 100.0% (n=1)
- HLB @ Wiesbaden Hbf 03:00 → 100.0% (n=2)
- IC @ Wiesbaden Hbf 00:00 → 100.0% (n=1)

### Key factors associated with higher delay risk (Top 10)
- rain_mm: uplift vs base = +9.75%
- precip_mm: uplift vs base = +9.75%
- temp_c: uplift vs base = +2.56%
- hour: uplift vs base = +1.99%
- wind_gusts_kmh: uplift vs base = +1.85%
- is_peak_hour: uplift vs base = +1.23%
- is_weekend: uplift vs base = -2.97%
- is_holiday: uplift vs base = -4.01%

## Objective 2: Decision support
- Use `study3_buffer_recommendations.csv` to pick recommended buffer minutes by weekday/hour.
- Use model inference (`06_smart_commute_tool.py`) for trip-level probability and alert label.