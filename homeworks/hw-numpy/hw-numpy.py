# %% [markdown]
# # DATS 6103 -- Homework: NumPy
#
# Due Tuesday, September 22, before class.
#
# Work in this file, run it top to bottom in a fresh interpreter before you
# submit, and upload it with your prompt log and critique (see the assignment
# page). Every function below must run without editing the tests underneath it.
#
# Last week you wrote the dot product, the matrix-vector product and matrix
# multiplication out of plain lists, and Q7(c) asked where plain Python ran
# out. This is the answer. Everything here is NumPy, and the last question asks
# what that bought and what it cost -- because it costs something.
#
# An AI assistant is allowed and encouraged. What is not optional is the
# disclosure, and the critique is graded alongside the code.

# %%
import math
import time

import numpy as np

rng = np.random.default_rng(6103)

# A month of readings from a sensor network: 30 days, 12 sensors, 4 channels
# (temperature, humidity, pressure, battery). Every question below works on
# this array, so run this cell first and leave it alone.
DAYS, SENSORS, CHANNELS = 30, 12, 4
T = rng.uniform(
    low=[15.0, 20.0, 980.0, 3.0],
    high=[35.0, 90.0, 1040.0, 4.2],
    size=(DAYS, SENSORS, CHANNELS),
)

# %%
# ---------------------------------------------------------------------------
# Q1. Building an axis, and moving one
# ---------------------------------------------------------------------------
# (a) stack_days(days) takes a LIST of 2-D arrays, each (SENSORS, CHANNELS),
#     and returns one 3-D array with the days along a new first axis.
#
# (b) channels_first(A) returns the same data with the channel axis moved to
#     the front: (DAYS, SENSORS, CHANNELS) becomes (CHANNELS, DAYS, SENSORS).
#     The values must still line up -- channel 1 must still be channel 1 --
#     and this rearranges how one buffer is read rather than building a new
#     one, so the result shares memory with A.

def stack_days(days):
    pass


def channels_first(A):
    pass


day_list = [rng.uniform(size=(SENSORS, CHANNELS)) for _ in range(5)]
S = stack_days(day_list)
assert S.shape == (5, SENSORS, CHANNELS)
assert np.array_equal(S[2], day_list[2]), "Q1: day 2 must still be day 2"

C = channels_first(T)
assert C.shape == (CHANNELS, DAYS, SENSORS)
assert np.array_equal(C[1], T[:, :, 1]), "Q1: channel 1 must still be channel 1"
assert np.shares_memory(C, T), "Q1: moving an axis rearranges the reading of one buffer"
print("Q1 ok")

# %%
# ---------------------------------------------------------------------------
# Q2. Broadcasting, by hand
# ---------------------------------------------------------------------------
# broadcast_shape(s1, s2) returns the shape the two would broadcast to, or
# raises ValueError if they do not. Implement the rule yourself -- right-align
# the shapes, pad the shorter one with 1s, and at each position the sizes must
# be equal or one of them must be 1.
#
# Do NOT call np.broadcast_shapes. The last block of tests compares your answer
# against it, which is only worth anything if you did not use it.

def broadcast_shape(s1, s2):
    pass


assert broadcast_shape((3,), (3,)) == (3,)
assert broadcast_shape((30, 12, 4), (4,)) == (30, 12, 4)
assert broadcast_shape((30, 12, 4), (12, 1)) == (30, 12, 4)
assert broadcast_shape((30, 1, 4), (12, 1)) == (30, 12, 4)
assert broadcast_shape((), (5, 5)) == (5, 5)
assert broadcast_shape((1,), ()) == (1,)

for bad in [((30, 12, 4), (12,)), ((3, 4), (4, 3)), ((2,), (3,))]:
    try:
        broadcast_shape(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Q2: {bad} must raise ValueError")

for s1, s2 in [((30, 12, 4), (4,)), ((30, 1, 4), (12, 1)), ((), (5, 5)),
               ((7, 1, 3), (1, 5, 1))]:
    assert broadcast_shape(s1, s2) == np.broadcast_shapes(s1, s2)
print("Q2 ok")

# %%
# ---------------------------------------------------------------------------
# Q3. A copy, and a view, on purpose
# ---------------------------------------------------------------------------
# (a) clip_channel(A, ch, lo, hi) returns a NEW array in which channel `ch` is
#     clipped to [lo, hi] and every other channel is untouched. A must not
#     change, and the result must not share memory with it. This is the same
#     obligation as cap_weights in the lab, and it is the one people lose marks
#     on: A[..., ch] = ... on the array you were handed edits the caller's data.
#
# (b) one_sensor(A, s) returns every reading from sensor s, shape
#     (DAYS, CHANNELS) -- and this one must be a VIEW. Writing into it should
#     reach the original.
#
# One question, both directions, because knowing which operations copy is the
# whole of this week.

def clip_channel(A, ch, lo, hi):
    pass


def one_sensor(A, s):
    pass


before = T.copy()
capped = clip_channel(T, 0, 18.0, 30.0)

assert capped.shape == T.shape
assert np.array_equal(T, before), "Q3: clip_channel must not modify its input"
assert not np.shares_memory(T, capped), "Q3: the result must be its own array"
assert capped[..., 0].max() <= 30.0 + 1e-12
assert capped[..., 0].min() >= 18.0 - 1e-12
assert (capped[..., 0] < 30.0).any(), "Q3: you replaced too much"
assert np.array_equal(capped[..., 1:], T[..., 1:]), "Q3: other channels untouched"
untouched = (T[..., 0] >= 18.0) & (T[..., 0] <= 30.0)
assert np.array_equal(capped[..., 0][untouched], T[..., 0][untouched])

v = one_sensor(T, 3)
assert v.shape == (DAYS, CHANNELS)
assert np.shares_memory(v, T), "Q3: one_sensor must return a VIEW, not a copy"
print("Q3 ok")

# %%
# ---------------------------------------------------------------------------
# Q4. Name the axis you are collapsing
# ---------------------------------------------------------------------------
# Each of these is one call. Say which axis disappears before you write it.
#
# (a) per_day(A)    -- one row per day, averaged over sensors    -> (DAYS, CHANNELS)
# (b) per_sensor(A) -- one row per sensor, averaged over days    -> (SENSORS, CHANNELS)
# (c) overall(A)    -- one number per channel                    -> (CHANNELS,)
# (d) hottest_day(A) -- the index of the day whose mean temperature (channel 0,
#     averaged over sensors) is highest. Return a plain int.
# (e) centre_by_day(A) -- every reading minus its own day's sensor-average, so
#     each day now averages to zero. The shape does not change, which is what
#     keepdims is for.

def per_day(A):
    pass


def per_sensor(A):
    pass


def overall(A):
    pass


def hottest_day(A):
    pass


def centre_by_day(A):
    pass


assert per_day(T).shape == (DAYS, CHANNELS)
assert per_sensor(T).shape == (SENSORS, CHANNELS)
assert overall(T).shape == (CHANNELS,)
# The design is balanced, so averaging the averages must give the grand mean.
assert np.allclose(per_day(T).mean(axis=0), overall(T))
assert np.allclose(per_sensor(T).mean(axis=0), overall(T))

d = hottest_day(T)
assert isinstance(d, int) and 0 <= d < DAYS
assert T[d, :, 0].mean() == T[:, :, 0].mean(axis=1).max()

Z = centre_by_day(T)
assert Z.shape == T.shape
assert np.allclose(Z.mean(axis=1), 0.0), "Q4: each day must now average to zero"
assert not np.shares_memory(Z, T)
print("Q4 ok")

# %%
# ---------------------------------------------------------------------------
# Q5. The same product, and the number you measure
# ---------------------------------------------------------------------------
# matmul_loop below is last week's answer, unchanged. Write matmul_np, which
# must agree with it and must raise ValueError when the inner dimensions
# disagree -- NumPy raises on its own, but the message is worth owning.
#
# Vectorized, which means no Python loop over the elements. Rewriting the
# triple loop with NumPy calls inside it is not the answer: the whole question
# is the gap between looping and not, and a rewrite that still loops closes it.
#
# Then run the cell. The timing is not a trick question: report what your
# machine actually does. The assert is a floor, not a target -- it only fires
# when the answer is still looping.

def matmul_loop(A, B):
    """Given. Last week's answer, in plain Python."""
    n, k = len(A), len(A[0])
    m = len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            out[i][j] = s
    return out


def matmul_np(A, B):
    pass


P = rng.uniform(size=(40, 25))
Q = rng.uniform(size=(25, 18))
assert matmul_np(P, Q).shape == (40, 18)
assert np.allclose(matmul_np(P, Q), np.array(matmul_loop(P.tolist(), Q.tolist())))

for bad in [(P, P), (np.ones((2, 3)), np.ones((2, 3)))]:
    try:
        matmul_np(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Q5: a shape mismatch must raise ValueError")

N = 120
Ab = rng.uniform(size=(N, N))
Bb = rng.uniform(size=(N, N))
t0 = time.perf_counter(); matmul_loop(Ab.tolist(), Bb.tolist()); slow = time.perf_counter() - t0
t0 = time.perf_counter(); matmul_np(Ab, Bb); fast = time.perf_counter() - t0
Q5_SPEEDUP = slow / fast

print(f"    {N}x{N}: loop {slow:.3f}s, numpy {fast:.5f}s, {Q5_SPEEDUP:.0f}x")
assert Q5_SPEEDUP >= 10, \
    "Q5: that is loop speed -- matmul_np is still looping in Python somewhere. "\
    "A vectorized one beats the loop by hundreds of times; report whatever you measure."
print("Q5 ok")

# %%
# ---------------------------------------------------------------------------
# Q6. Masks, and the `and` that is not
# ---------------------------------------------------------------------------
# (a) count_hot(A, limit) -- how many readings have channel 0 above limit.
#     A plain int.
# (b) flag_both(A, hot, dry) -- a boolean array, shape (DAYS, SENSORS), true
#     where channel 0 is above `hot` AND channel 1 is below `dry`. Python's
#     `and` cannot do this; the test below proves it by asking `and` to try.
# (c) safe_ratio(a, b) -- elementwise a / b, giving 0.0 wherever b is 0, and
#     without NumPy printing a divide-by-zero warning. Guard the input rather
#     than filtering the output.

def count_hot(A, limit):
    pass


def flag_both(A, hot, dry):
    pass


def safe_ratio(a, b):
    pass


assert count_hot(T, 30.0) == int((T[..., 0] > 30.0).sum())
assert count_hot(T, 100.0) == 0

m = flag_both(T, 30.0, 40.0)
assert m.shape == (DAYS, SENSORS)
assert m.dtype == bool
assert np.array_equal(m, (T[..., 0] > 30.0) & (T[..., 1] < 40.0))

try:
    bool((T[..., 0] > 30.0) and (T[..., 1] < 40.0))
except ValueError:
    pass
else:
    raise AssertionError("Q6: `and` on two arrays must raise -- that is the point")

r = safe_ratio([1.0, 2.0, 3.0], [2.0, 0.0, 6.0])
assert np.allclose(r, [0.5, 0.0, 0.5])
assert np.all(np.isfinite(safe_ratio(np.ones(5), np.zeros(5))))
print("Q6 ok")

# %%
# ---------------------------------------------------------------------------
# Q7. What is a view, and what did this cost
# ---------------------------------------------------------------------------
# (a) For each expression below, decide whether the result shares memory with
#     T. Decide first, from what the operation has to do; then check yourself.

EXPRESSIONS = [
    "T[0]",
    "T.reshape(-1, CHANNELS)",
    "T[T[..., 0] > 25.0]",
    "T.T",
]

Q7_SHARES = [None, None, None, None]     # replace each with True or False

truth = [
    np.shares_memory(T[0], T),
    np.shares_memory(T.reshape(-1, CHANNELS), T),
    np.shares_memory(T[T[..., 0] > 25.0], T),
    np.shares_memory(T.T, T),
]
assert all(isinstance(x, bool) for x in Q7_SHARES), "Q7a: four booleans"
assert Q7_SHARES == truth, f"Q7a: not right -- the truth is {truth}"

# (b) One sentence: what do the ones that share have in common, and what does
#     the odd one out have to do that the others do not?

Q7_WHY = ""

assert len(Q7_WHY.split()) >= 15, "Q7b: one real sentence, not a fragment"

# (c) Last week's Q7(c) asked where plain Python ran out. Now the other
#     direction: name one thing NumPy made EASIER TO GET WRONG than plain lists
#     did. Name the operation and the specific shapes or the specific line --
#     "broadcasting is confusing" is not an answer, and neither is "nothing".

Q7_NEW_DANGER = ""

assert len(Q7_NEW_DANGER.split()) >= 20, "Q7c: name the operation and the case"
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

print("\nAll questions passed.")
