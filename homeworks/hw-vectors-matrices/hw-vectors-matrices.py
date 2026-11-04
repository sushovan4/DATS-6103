# %% [markdown]
# # DATS 6103 -- Homework: Vectors and Matrices
#
# Due Tuesday, September 8, before class.
#
# Work in this file, run it top to bottom in a fresh interpreter before you
# submit, and upload it with your prompt log and critique (see the assignment
# page). Every function below must run without editing the tests underneath it.
#
# **No NumPy.** Plain lists and lists of lists, all the way through. You met
# just enough NumPy in the lab to get the lab done; here you build the
# operations yourself, because the point of the last question is to say where
# plain Python runs out. Next week's session is the answer to that question.
#
# An AI assistant is allowed and encouraged. What is not optional is the
# disclosure, and the critique is graded alongside the code.

# %%
# ---------------------------------------------------------------------------
# Q1. The dot product, and the case it must refuse
# ---------------------------------------------------------------------------
# Return the dot product of two vectors of equal length. If the lengths differ,
# raise ValueError -- do not return a number, and do not let zip() quietly
# truncate the longer one. That silent truncation is the bug this question
# exists to prevent.

def dot(u, v):
    pass


assert dot([1, 2, 3], [4, 5, 6]) == 32
assert dot([1, 0], [0, 1]) == 0            # orthogonal
assert dot([3, 4], [3, 4]) == 25           # a vector with itself
assert dot([], []) == 0                    # the empty sum is 0, not an error

try:
    dot([1, 2, 3], [1, 2])
except ValueError:
    pass
else:
    raise AssertionError("Q1: mismatched lengths must raise ValueError")
print("Q1 ok")

# %%
# ---------------------------------------------------------------------------
# Q2. Length, and the angle you cannot always measure
# ---------------------------------------------------------------------------
# (a) norm(v) returns the length of v. Build it from dot() -- do not retype the
#     sum of squares.
#
# (b) cosine(u, v) returns cos(theta) between u and v. The zero vector has no
#     direction, so if either vector is zero, return None rather than dividing
#     by zero.

import math


def norm(v):
    pass


def cosine(u, v):
    pass


assert norm([3, 4]) == 5
assert norm([0, 0, 0]) == 0
assert abs(norm([1, 1]) - math.sqrt(2)) < 1e-12

assert abs(cosine([1, 0], [0, 1]) - 0.0) < 1e-12          # perpendicular
assert abs(cosine([1, 1], [2, 2]) - 1.0) < 1e-12          # same direction
assert abs(cosine([1, 1], [-1, -1]) + 1.0) < 1e-12        # opposite
assert abs(cosine([1, 0], [1, 1]) - math.sqrt(2) / 2) < 1e-12
assert cosine([0, 0], [1, 2]) is None                     # no direction
print("Q2 ok")

# %%
# ---------------------------------------------------------------------------
# Q3. Shape, transpose, and the matrix that is not a matrix
# ---------------------------------------------------------------------------
# A matrix is a list of rows, each row a list. shape(A) returns (rows, cols).
# A ragged list of lists is not a matrix: raise ValueError. An empty matrix []
# has shape (0, 0).
#
# transpose(A) returns a new matrix; it must not modify A.

def shape(A):
    pass


def transpose(A):
    pass


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
assert shape([[1], [2], [3]]) == (3, 1)
assert shape([]) == (0, 0)

try:
    shape([[1, 2], [3]])
except ValueError:
    pass
else:
    raise AssertionError("Q3: a ragged list of lists must raise ValueError")

A3 = [[1, 2, 3], [4, 5, 6]]
assert transpose(A3) == [[1, 4], [2, 5], [3, 6]]
assert A3 == [[1, 2, 3], [4, 5, 6]], "Q3: transpose must not modify its input"
assert transpose(transpose(A3)) == A3
assert shape(transpose(A3)) == (3, 2)
print("Q3 ok")

# %%
# ---------------------------------------------------------------------------
# Q4. Ax is a combination of the COLUMNS of A
# ---------------------------------------------------------------------------
# The reading gives two ways to read the same arithmetic: each output entry is
# a row of A dotted with x, or the whole output is x[0]*col0 + x[1]*col1 + ...
# Build the second one. It is the reading that comes back on Nov 3.
#
# (a) columns(A) returns the columns of A, each as a list.
# (b) matvec(A, x) returns A @ x, built as a linear combination of columns(A)
#     using the two helpers below. Raise ValueError if the shapes do not match.

def vadd(u, v):
    """Given. Adds two vectors of equal length."""
    return [a + b for a, b in zip(u, v)]


def scale(c, v):
    """Given. Multiplies a vector by a scalar."""
    return [c * a for a in v]


def columns(A):
    pass


def matvec(A, x):
    pass


assert columns([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

A4 = [[2, 0, 1], [1, 3, 0], [4, 1, 2]]
assert matvec(A4, [1, 1, 1]) == [3, 4, 7]
assert matvec(A4, [0, 0, 0]) == [0, 0, 0]
assert matvec([[1, 2], [3, 4]], [5, 6]) == [17, 39]

# A e_j is exactly the j-th column of A. This is the whole of "a matrix is
# determined by where it sends the basis", written as a test.
for j in range(3):
    e = [0, 0, 0]
    e[j] = 1
    assert matvec(A4, e) == columns(A4)[j], f"Q4: A e_{j} must be column {j}"

try:
    matvec(A4, [1, 2])
except ValueError:
    pass
else:
    raise AssertionError("Q4: a shape mismatch must raise ValueError")
print("Q4 ok")

# %%
# ---------------------------------------------------------------------------
# Q5. Matrix multiplication, and an identity worth trusting
# ---------------------------------------------------------------------------
# matmul(A, B) returns A @ B, raising ValueError when the inner dimensions
# disagree. Then the tests check (AB)^T = B^T A^T on matrices nobody picked to
# make it work -- including a non-square pair, where getting the order wrong
# does not even produce a matrix of the right shape.

def matmul(A, B):
    pass


assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
assert matmul([[1, 0], [0, 1]], [[5, 6], [7, 8]]) == [[5, 6], [7, 8]]
assert matmul([[1, 2, 3]], [[1], [2], [3]]) == [[14]]
assert shape(matmul([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]])) == (2, 2)

P = [[1, 2, 3], [4, 5, 6]]          # 2x3
Q = [[7, 8], [9, 10], [11, 12]]     # 3x2
assert transpose(matmul(P, Q)) == matmul(transpose(Q), transpose(P))

R = [[2, 0], [1, 3]]
S = [[1, 4], [5, 6]]
assert transpose(matmul(R, S)) == matmul(transpose(S), transpose(R))
assert matmul(R, S) != matmul(S, R), "Q5: these two were chosen not to commute"

try:
    matmul([[1, 2], [3, 4]], [[1, 2, 3]])
except ValueError:
    pass
else:
    raise AssertionError("Q5: inner dimensions must agree, or raise ValueError")
print("Q5 ok")

# %%
# ---------------------------------------------------------------------------
# Q6. Determinant and inverse, and the matrix that has neither
# ---------------------------------------------------------------------------
# (a) det2(A) for a 2x2, by the ad - bc rule.
# (b) det3(A) for a 3x3, by cofactor expansion along the top row -- call det2
#     three times rather than writing out six products.
# (c) inverse2(A) returns the inverse of a 2x2, or None when A is singular.
#     Returning None is the point: the caller has to handle it, which is the
#     behavior a silent nan or a crash both fail to give.

def det2(A):
    pass


def det3(A):
    pass


def inverse2(A):
    pass


assert det2([[1, 2], [3, 4]]) == -2
assert det2([[1, 2], [2, 4]]) == 0            # parallel columns
assert det3([[2, 0, 1], [1, 3, 0], [4, 1, 2]]) == 1
assert det3([[1, 2, 3], [0, 2, 1], [0, 0, 3]]) == 6     # triangular: 1*2*3
assert det3([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 0     # singular

inv = inverse2([[4, 7], [2, 6]])
assert all(abs(a - b) < 1e-12 for r, s in zip(inv, [[0.6, -0.7], [-0.2, 0.4]])
           for a, b in zip(r, s))
# The definition, tested rather than assumed:
assert all(abs(a - b) < 1e-12
           for r, s in zip(matmul(inverse2([[4, 7], [2, 6]]), [[4, 7], [2, 6]]),
                           [[1, 0], [0, 1]])
           for a, b in zip(r, s))
assert inverse2([[1, 2], [2, 4]]) is None     # singular: no inverse to return
print("Q6 ok")

# %%
# ---------------------------------------------------------------------------
# Q7. What the matrix can and cannot reach
# ---------------------------------------------------------------------------
# Ax = b has a solution exactly when b lies in the column space of A -- the set
# of all linear combinations of the columns.
#
# (a) For the matrix below, decide for each of the four b's whether Ax = b has
#     a solution. Answer by reasoning about the columns, then check yourself
#     with the code you already wrote.

A7 = [[1, 2],
      [2, 4]]

CANDIDATES = [[3, 6], [1, 0], [0, 0], [-2, -4]]

Q7_REACHABLE = [None, None, None, None]     # replace each with True or False

assert all(isinstance(t, bool) for t in Q7_REACHABLE), "Q7a: four booleans"

# (b) One sentence saying WHY the unreachable ones are unreachable. Name what
#     the columns of A7 have in common and what that does to the plane.

Q7_WHY = ""

assert len(Q7_WHY.split()) >= 15, "Q7b: one real sentence, not a fragment"

# (c) One sentence: where did plain Python run out in this assignment? Name the
#     specific function above that was most painful to write or would be worst
#     to scale to a 1000x1000 matrix, and say what went wrong -- speed, or the
#     amount of code, or how easy it was to get an index backwards. This is the
#     question next week's session answers.

Q7_PYTHON_LIMIT = ""

assert len(Q7_PYTHON_LIMIT.split()) >= 20, "Q7c: one real sentence, with a function named"
print("Q7 ok")

# %%
# ---------------------------------------------------------------------------
# AI disclosure -- required, and graded
# ---------------------------------------------------------------------------
# Fill both in. "I did not use one" is an acceptable and complete answer to the
# first, in which case write that and leave the second empty.

AI_PROMPT_LOG = """
"""

AI_CRITIQUE = """
What did the assistant get wrong, or would have got wrong if you had accepted
it unchanged? Be specific -- name the function and the input.
"""
