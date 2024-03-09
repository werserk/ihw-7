from fractions import Fraction
import numpy as np


def canonical_form(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q_initial = Q
    Q = Q.copy()
    assert Q.shape[0] == Q.shape[1]

    C = np.eye(len(Q), dtype=Fraction)
    for i in range(len(Q)):
        if (Q[i:, i:] == 0).all():
            break

        non_zero_x, non_zero_y = next(zip(*Q[i:, i:].nonzero()))
        non_zero_x += i
        non_zero_y += i

        Q[[non_zero_x, i]] = Q[[i, non_zero_x]]
        Q[:, [non_zero_y, i]] = Q[:, [i, non_zero_y]]
        C[[non_zero_x, i]] = C[[i, non_zero_x]]

        for j in range(i + 1, len(Q)):
            C[j] -= C[i] * Q[j, i] / Q[i, i]
            Q[j] -= Q[i] * Q[j, i] / Q[i, i]
            Q[:, j] -= Q[:, i] * Q[i, j] / Q[i, i]
    assert (C @ Q_initial @ C.T == Q).all()
    return Q, C


def pretty_print_fraction_matrix(matrix: np.ndarray, label: str = "") -> None:
    print(label)
    for row in matrix:
        print("\t".join(map(str, row)))


mat = np.array([[9, -6, -27], [-6, 3, 16], [-27, 16, 81]])
mat = np.vectorize(Fraction)(mat)

diag, basis_change_mat = canonical_form(mat)

pretty_print_fraction_matrix(diag, label="Diagonal")
pretty_print_fraction_matrix(basis_change_mat, label="Basis change matrix")
