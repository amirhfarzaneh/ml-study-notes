"""
this is a sample code on how CE is calculated in LLMs.
"""

import math


def cross_entropy(logits, labels, ignore_index=-100):
    total, n = 0.0, 0
    for z, y in zip(logits, labels):
        if y == ignore_index:
            continue
        m = max(z)
        lse = m + math.log(sum(math.exp(a - m) for a in z))
        total += lse - z[y]
        n += 1
    return total / n


def main():
    logits = [
        [2.0, 1.0, 0.1],  # position 0, true token = 0
        [0.5, 2.5, 0.0],  # position 1, true token = 1
        [1.0, 1.0, 1.0],
    ]  # position 2, padding (label = -100)

    labels = [0, 1, -100]

    loss = cross_entropy(logits, labels)
    print(loss)


if __name__ == "__main__":
    main()
