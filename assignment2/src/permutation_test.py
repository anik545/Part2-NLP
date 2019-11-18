import numpy as np


R = 5000


def random_swap_mean_diff(arr1, arr2):
    r = np.random.randint(2, size=len(arr1))
    a1 = np.append(arr1[r == 1], arr2[r == 0])
    a2 = np.append(arr2[r == 1], arr2[r == 0])
    return abs(np.mean(a1) - np.mean(a2))


def permutation_test(systemAcorrects, systemBcorrects):
    if len(systemAcorrects) != len(systemBcorrects):
        return "Not the same length"

    a = np.array(systemAcorrects)
    b = np.array(systemBcorrects)

    original_diff = abs(np.mean(a) - np.mean(b))

    diffs = [random_swap_mean_diff(a, b) for x in range(R)]
    # TODO: should this be > or >= ???
    s = sum(1.0 if diff >= original_diff else 0.0 for diff in diffs)

    print(diffs)
    print(s)

    p = (s + 1.0) / (R + 1.0)

    return p


if __name__ == "__main__":
    p = permutation_test([True, True, True, True, True, True],
                         [False, False, False, False, False, False])
    print(p)
