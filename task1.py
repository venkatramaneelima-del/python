
sales = [12000, 0, 8500, 15000, -200, 9800]

total = 0

for value in sales:

    if value <= 0:
        print(f"Skipped invalid value {value}")
        continue

    total += value
    print(f"Added {value} total {total}")

    if total > 30000:
        print(f"Target reached. Stopped at {total}")
        break
else:
    print("Target not reached")