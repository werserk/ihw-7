from fractions import Fraction

import numpy as np


def signature(Q: np.ndarray) -> tuple[int, int, int]:
    assert Q.shape[0] == Q.shape[1]
    for i in range(len(Q)):
        if (Q[i:, i:] == 0).all():
            break

        non_zero_x, non_zero_y = next(zip(*Q[i:, i:].nonzero()))
        non_zero_x += i
        non_zero_y += i

        Q[[non_zero_x, i]] = Q[[i, non_zero_x]]
        Q[:, [non_zero_y, i]] = Q[:, [i, non_zero_y]]

        for j in range(i + 1, len(Q)):
            Q[j] -= Q[i] * Q[j, i] / Q[i, i]
            Q[:, j] -= Q[:, i] * Q[i, j] / Q[i, i]
    diag = Q.diagonal()
    return (
        np.count_nonzero(diag > 0),
        np.count_nonzero(diag < 0),
        np.count_nonzero(diag == 0),
    )


given = np.vectorize(Fraction)(
    np.array(
        [[-7, 18, -25, -1], [18, 41, -5, 37], [-25, -5, 98, -19], [-1, -37, -19, 49]]
    )
)

integral_quadratic_form = np.array(
    [
        [Fraction(1 - 3 ** (k + m + 1) + 2 ** (k + m + 1), k + m + 1) for m in range(4)]
        for k in range(4)
    ]
)

print(f"{signature(given)=}, {signature(integral_quadratic_form)=}")
