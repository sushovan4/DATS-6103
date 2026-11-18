# %% [markdown]
# # DATS 6103 -- Homework: Decision Trees
#
# Due Tuesday, December 1, before class.
#
# Work in this file, run it top to bottom in a fresh interpreter before you
# submit, and upload it with your prompt log and critique (see the assignment
# page). Every function below must run without editing the tests underneath it.
#
# A tree tells you what it did, not what is true. Each question below builds
# one piece of the machinery by hand -- a split, a depth, a bag, an importance,
# a boosted sum -- and then asks what its output is entitled to say. The last
# question asks what trees make easier to get wrong than the linear models did.
#
# An AI assistant is allowed and encouraged. What is not optional is the
# disclosure, and the critique is graded alongside the code.

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Friedman's benchmark, simulated, so you know the truth:
#     y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + noise
# The first five columns carry all the signal. Columns 6 to 10 are pure noise:
# y does not depend on them at all. Leave this cell alone.
X_TRAIN, Y_TRAIN = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=6103)
X_TEST, Y_TEST = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=6104)
SIGNAL, NOISE = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
FOLDS = KFold(5, shuffle=True, random_state=6103)

# A yes/no version of the same data for Q1: is y above its median?
LABELS = (Y_TRAIN > np.median(Y_TRAIN)).astype(int)


def r2(y, yhat):
    """1 - SSE/SST, as in HW Regression. Given, so every question shares it."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def mse(y, yhat):
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)) ** 2))

# %%
# ---------------------------------------------------------------------------
# Q1. A split, by hand
# ---------------------------------------------------------------------------
# (a) gini(labels) returns 1 - sum of p_c^2 over the classes present.
#
# (b) split_score(x, labels, t) sends the rows with x <= t left and the rest
#     right, and returns the children's Gini weighted by their share of the
#     rows: n_left/n * gini(left) + n_right/n * gini(right).
#
# (c) best_split(x, labels) tries every threshold halfway between two
#     consecutive distinct sorted values of x and returns (t, score) for the
#     lowest score. If two thresholds tie, keep the smaller t.
#
# (d) best_feature(X, labels) runs best_split on every column and returns the
#     index of the column whose best split scores lowest.

def gini(labels):
    pass


def split_score(x, labels, t):
    pass


def best_split(x, labels):
    pass


def best_feature(X, labels):
    pass


def check_q1():
    assert abs(gini([0, 0, 1, 1]) - 0.5) < 1e-12, "Q1a: an even mix of two classes is 0.5"
    assert gini([1, 1, 1]) == 0, "Q1a: a pure node is 0"
    assert abs(split_score([1, 2, 3, 4], [0, 0, 1, 1], 2.5)) < 1e-12, "Q1b: a perfect split scores 0"
    assert abs(split_score([1, 2, 3, 4], [0, 1, 0, 1], 2.5) - 0.5) < 1e-12, "Q1b: weight each child by its share"

    x = X_TRAIN[:, 3]
    t, s = best_split(x, LABELS)
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x.reshape(-1, 1), LABELS)
    # scikit-learn stores features as 32-bit floats, so its midpoint differs in the 8th decimal.
    assert abs(t - stump.tree_.threshold[0]) < 1e-6, "Q1c: scikit-learn's stump splits this column at the same t"
    assert abs(s - split_score(x, LABELS, t)) < 1e-12, "Q1c: return the score of the threshold you return"

    j = best_feature(X_TRAIN, LABELS)
    full_stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_TRAIN, LABELS)
    assert j == full_stump.tree_.feature[0], "Q1d: the column a depth-1 tree would choose"
    print(f"Q1 ok   best column {j + 1} (x{j + 1}), threshold {best_split(X_TRAIN[:, j], LABELS)[0]:.3f}")


check_q1()

# %%
# ---------------------------------------------------------------------------
# Q2. How deep?
# ---------------------------------------------------------------------------
# (a) depth_table(depths) fits DecisionTreeRegressor(max_depth=d, random_state=0)
#     on the training data for each d, and returns a DataFrame indexed by depth
#     with the columns "train_r2" (score on the data it was fit on) and "cv_r2"
#     (mean cross_val_score on the training data with cv=FOLDS).
#
# (b) best_depth_and_test(table) returns (best_depth, test_r2): the depth with the
#     highest cv_r2, and that tree, refit on all the training data, scored on the
#     test set.

DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 25]   # 25 is deeper than the tree ever grows here


def tree_at(d):
    return DecisionTreeRegressor(max_depth=d, random_state=0)


def depth_table(depths):
    pass


def best_depth_and_test(table):
    pass


def check_q2():
    t = depth_table(DEPTHS)
    assert list(t.columns) == ["train_r2", "cv_r2"] and list(t.index) == DEPTHS, "Q2a: one row per depth, two columns"
    assert t["train_r2"].is_monotonic_increasing, "Q2a: a deeper tree always fits its own data better"
    assert abs(t.loc[25, "train_r2"] - 1) < 1e-9, "Q2a: grown out, it fits every training row"
    d, test = best_depth_and_test(t)
    assert d == 5, "Q2b: the depth cross-validation prefers"
    assert abs(test - 0.580) < 0.002, "Q2b: refit at that depth, scored on the test set"
    print(f"Q2 ok   best depth {d}, test R^2 {test:.3f}")
    print(t.round(3))


check_q2()

# %%
# ---------------------------------------------------------------------------
# Q3. Bagging, by hand
# ---------------------------------------------------------------------------
# (a) bag_predictions(X_new, B, seed) makes rng = np.random.default_rng(seed),
#     and B times: draws n row numbers with rng.integers(0, n, n), fits
#     DecisionTreeRegressor(random_state=0) on those rows of the training data,
#     and predicts X_new. It returns a B x len(X_new) array, one row per tree.
#
# (b) bag_report(B, seed) returns a dict with "single_r2" (one fully grown tree
#     fit on all the training data, scored on the test set), "bag_r2" (the mean
#     of the B trees' test predictions, scored with r2), and "spread_at_first"
#     (the standard deviation of the B trees' predictions for the first test row).

def bag_predictions(X_new, B, seed):
    pass


def bag_report(B, seed):
    pass


def check_q3():
    P = bag_predictions(X_TEST[:5], 3, 1)
    assert P.shape == (3, 5), "Q3a: one row per tree, one column per new point"
    rep = bag_report(200, 6103)
    assert abs(rep["single_r2"] - 0.487) < 0.002, "Q3b: one grown-out tree on the test set"
    assert abs(rep["bag_r2"] - 0.799) < 0.002, "Q3b: average the 200 trees' predictions, then score"
    assert abs(rep["spread_at_first"] - 1.841) < 0.002, "Q3b: the spread of the trees at the first test point"
    print("Q3 ok  ", {k: round(v, 3) for k, v in rep.items()})


check_q3()

# %%
# ---------------------------------------------------------------------------
# Q4. What importance measures
# ---------------------------------------------------------------------------
# FOREST is a random forest fit on the training data, given.
#
# (a) noise_share() returns the share of FOREST.feature_importances_ that goes
#     to the five NOISE columns.
#
# (b) permutation_importance(model, X, y, seed) makes
#     rng = np.random.default_rng(seed), computes the model's MSE on (X, y), and
#     then for each column j in order: copies X, shuffles column j with
#     rng.permutation, and records how much the MSE rises. It returns one number
#     per column. Do not import scikit-learn's version.
#
# (c) noise_share_permutation(imp) returns the share of the total of imp that
#     goes to the NOISE columns.

FOREST = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_TRAIN, Y_TRAIN)


def noise_share():
    pass


def permutation_importance(model, X, y, seed):
    pass


def noise_share_permutation(imp):
    pass


def check_q4():
    ns = noise_share()
    assert 0.05 < ns < 0.15, "Q4a: the share of feature_importances_ on the five noise columns"
    imp = permutation_importance(FOREST, X_TEST, Y_TEST, 6103)
    assert len(imp) == 10, "Q4b: one number per column"
    assert imp[3] == max(imp), "Q4b: on held-out data, x4 matters most"
    assert abs(imp[3] - 13.325) < 0.01, "Q4b: one rng, made once, shuffling the columns in order"
    assert all(abs(imp[j]) < 0.1 for j in NOISE), "Q4b: shuffling a noise column changes nothing"
    nsp = noise_share_permutation(imp)
    assert abs(nsp) < 0.01, "Q4c: the noise columns' share of permutation importance"
    print(f"Q4 ok   noise share: impurity {ns:.3f}, permutation {nsp:.4f}")
    print("        permutation importance:", np.round(imp, 2))


check_q4()

# %%
# ---------------------------------------------------------------------------
# Q5. Boosting, by hand
# ---------------------------------------------------------------------------
# (a) boost(n_trees, lr, depth) starts every prediction at the training mean,
#     and then n_trees times fits DecisionTreeRegressor(max_depth=depth,
#     random_state=0) to the current residuals (Y_TRAIN minus the current
#     predictions) and adds lr times that tree's predictions. It returns
#     (start, trees): the mean, and the list of fitted trees.
#
# (b) staged_mse(start, trees, lr, X, y) returns a list: the MSE on (X, y) after
#     1 tree, after 2, ... after all of them.
#
# (c) rate_table() runs boost with 300 trees for each (lr, depth) in COMBOS and
#     returns a DataFrame with one row per combination, in that order, and the
#     columns "lr", "depth", "train_mse" (after 300 trees), "best_stage" (the
#     number of trees with the lowest TEST mse), "best_test_mse" and "test_mse"
#     (after 300 trees).

COMBOS = [(0.1, 3), (0.1, 6), (1.0, 3), (1.0, 6)]


def boost(n_trees, lr, depth):
    pass


def staged_mse(start, trees, lr, X, y):
    pass


def rate_table():
    pass


def check_q5():
    start, trees = boost(3, 0.5, 2)
    assert abs(start - Y_TRAIN.mean()) < 1e-12 and len(trees) == 3, "Q5a: start at the mean, then one tree per round"
    s = staged_mse(start, trees, 0.5, X_TRAIN, Y_TRAIN)
    assert len(s) == 3 and s[0] > s[1] > s[2], "Q5b: each round lowers the training error"
    t = rate_table()
    assert list(t.columns) == ["lr", "depth", "train_mse", "best_stage", "best_test_mse", "test_mse"], "Q5c: the columns"
    assert t.loc[0, "best_stage"] == 300 and abs(t.loc[0, "test_mse"] - 2.815) < 0.01, "Q5c: slow and shallow is still improving at 300"
    assert t.loc[2, "best_stage"] == 9, "Q5c: a learning rate of 1 is best after 9 trees"
    assert t.loc[2, "test_mse"] > t.loc[2, "best_test_mse"], "Q5c: and only gets worse after that"
    print("Q5 ok")
    print(t.round(3))


check_q5()

# %%
# ---------------------------------------------------------------------------
# Q6. Say what it means
# ---------------------------------------------------------------------------
# (a) A colleague reads FOREST.feature_importances_, sees about 0.02 on column 7,
#     and writes: "x7 has a small but real effect on y." The data were built so
#     that y does not depend on x7 at all. Say what the 0.02 measures, and what
#     you would report instead, with your number from Q4.
#
# (b) In Q5, a learning rate of 1 did best after 9 trees and a learning rate of
#     0.1 was still improving after 300. Explain why, and say how you would
#     choose the number of trees without looking at the test set.
#
# (c) HW Regression asked what scikit-learn made easier to get wrong than pandas.
#     Now: name one thing a TREE, or an ensemble of them, makes easier to get
#     wrong than the linear models did. Name the case. "Overfitting" alone is not
#     an answer.

Q6_IMPORTANCE = ""

Q6_RATE = ""

Q6_NEW_DANGER = ""


def check_q6():
    assert len(Q6_IMPORTANCE.split()) >= 25, "Q6a: say what it measures, and give the number"
    assert len(Q6_RATE.split()) >= 25, "Q6b: say why, and how you would choose"
    assert len(Q6_NEW_DANGER.split()) >= 20, "Q6c: name the case"
    print("Q6 ok")


check_q6()

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
