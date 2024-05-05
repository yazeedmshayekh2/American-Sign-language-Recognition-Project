def add_padding(number, target_divisor):
    padding = (target_divisor - (number % target_divisor)) % target_divisor
    return number + padding

print(add_padding(24888, 543))