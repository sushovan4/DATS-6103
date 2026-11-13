# %% [markdown]
# # DATS 6103 -- Homework: Regression
#
# Due Tuesday, November 10, before class.
#
# Work in this file, run it top to bottom in a fresh interpreter before you
# submit, and upload it with your prompt log and critique (see the assignment
# page). Every function below must run without editing the tests underneath it.
#
# The thread from the last two homeworks, one layer up. NumPy raised when you
# got a shape wrong. Pandas did not raise, and lined your data up on labels you
# had not checked. scikit-learn does not raise either, and what it hands back
# is a model that fit: a coefficient with a sign, a score with three decimals.
# Each question below is a case where the fit is correct, the number is real,
# and the sentence you would write about it is false. The last one asks you to
# name the case that would have caught you out.
#
# An AI assistant is allowed and encouraged. What is not optional is the
# disclosure, and the critique is graded alongside the code.

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Days at one Capital Bikeshare station. This is simulated, so you know the
# truth: a degree Celsius is worth 4 trips, whatever else is in the model. The
# station has TWO thermometers, one in Celsius and one in Fahrenheit, and they
# disagree only by a little sensor noise. Leave this cell alone.
def make_days(n, seed):
    rng = np.random.default_rng(seed)
    temp_c = rng.normal(15, 9, n)
    temp_f = temp_c * 9 / 5 + 32 + rng.normal(0, 0.4, n)
    humidity = rng.uniform(30, 95, n)                      # percent
    wind = rng.gamma(2, 7, n)                              # km/h
    weekend = (rng.random(n) < 2 / 7).astype(float)
    trips = (140 + 4 * temp_c - 0.6 * humidity - 1.2 * wind + 30 * weekend
             + rng.normal(0, 25, n))
    return pd.DataFrame({"temp_c": temp_c, "temp_f": temp_f, "humidity": humidity,
                         "wind": wind, "weekend": weekend, "trips": trips})


FEATURES = ["temp_c", "temp_f", "humidity", "wind", "weekend"]
DAYS = make_days(150, 6112)                  # what you fit on
NEW_DAYS = make_days(2000, 6200)             # days the model has not seen
# The same days, after the Fahrenheit sensor starts reading 10 degrees high.
BROKEN_DAYS = NEW_DAYS.assign(temp_f=NEW_DAYS["temp_f"] + 10)

# %%
# ---------------------------------------------------------------------------
# Q1. Least squares is a projection
# ---------------------------------------------------------------------------
# (a) design(df, columns) returns the design matrix as a float ndarray: a first
#     column of ones, then the named columns in the order given.
#
# (b) ols_qr(X, y) returns the least squares coefficients, using np.linalg.qr
#     and then np.linalg.solve on R w = Q^T y. Not lstsq, not pinv, not inv, not
#     LinearRegression: September's projection chapter is the method.
#
# (c) project(X, y) returns the fitted values as Q Q^T y, without computing any
#     coefficients. The asserts check the two things that make it a projection:
#     the residual is orthogonal to every column of X, and projecting twice
#     changes nothing.
#
# (d) r2(y, yhat) returns 1 - SSE/SST, where SST is measured around the mean of
#     y. Do not clip it: on new data it can be negative, and that is information.
#
# (e) r2_by_noise(train, test, ks) takes two tuples (X, y, Z), where Z holds
#     columns of pure noise. For each k in ks, append the first k columns of Z
#     to X, fit on train with ols_qr, and record R^2 on train and on test.
#     Return the two lists, (train_r2, test_r2).

def design(df, columns):
    pass


def ols_qr(X, y):
    pass


def project(X, y):
    pass


def r2(y, yhat):
    pass


def r2_by_noise(train, test, ks):
    pass


def check_q1():
    no_sensor = ["temp_c", "humidity", "wind", "weekend"]
    X = design(DAYS, no_sensor)
    y = DAYS["trips"].to_numpy()
    assert X.shape == (150, 5), "Q1a: 150 rows, a column of ones and four features"
    assert np.allclose(X[:, 0], 1), "Q1a: the first column is ones"

    w = ols_qr(X, y)
    sk = LinearRegression(fit_intercept=False).fit(X, y)
    assert np.allclose(w, sk.coef_), "Q1b: not the least squares solution"

    p = project(X, y)
    assert np.allclose(X.T @ (y - p), 0, atol=1e-6), "Q1c: the residual must be orthogonal to the columns"
    assert np.allclose(project(X, p), p), "Q1c: projecting twice must change nothing"
    assert abs(np.sum(y - p)) < 1e-6, "Q1c: with an intercept, the residuals sum to zero"

    assert abs(r2(y, X @ w) - sk.score(X, y)) < 1e-9, "Q1d: R^2 is 1 - SSE/SST"
    assert r2(np.array([1.0, 2.0, 3.0]), np.array([9.0, 9.0, 9.0])) < 0, "Q1d: do not clip"

    rng = np.random.default_rng(1)
    Xn = design(NEW_DAYS, no_sensor)
    yn = NEW_DAYS["trips"].to_numpy()
    ks = [0, 20, 60, 100, 145]
    tr, te = r2_by_noise((X, y, rng.normal(size=(150, 145))),
                         (Xn, yn, rng.normal(size=(2000, 145))), ks)
    assert all(b >= a for a, b in zip(tr, tr[1:])), "Q1e: a bigger subspace cannot project worse"
    assert abs(tr[-1] - 1) < 1e-9, "Q1e: 150 columns on 150 rows fit exactly"
    assert te[-1] < 0, "Q1e: ...and do worse on new days than predicting the mean"
    print("Q1 ok   R^2 on train:", np.round(tr, 3), " on new days:", np.round(te, 3))


check_q1()

# %%
# ---------------------------------------------------------------------------
# Q2. Two thermometers
# ---------------------------------------------------------------------------
# (a) ols_svd(X, y, tol=1e-10) returns the coefficients from the SVD,
#     w = V diag(1/s) U^T y, but treating any s_i below tol * s.max() as a
#     direction with no information: 1/s_i becomes 0 for those. Use
#     full_matrices=False.
#
# (b) Fit DAYS on the five FEATURES and read the two temperature coefficients.
#     warming_effect(w_c, w_f) returns the change in predicted trips when the day
#     warms by 1 degree Celsius, so that BOTH sensors move. The Fahrenheit sensor
#     moves 1.8 degrees.

def ols_svd(X, y, tol=1e-10):
    pass


def warming_effect(w_c, w_f):
    pass


def check_q2():
    X = design(DAYS, FEATURES)
    y = DAYS["trips"].to_numpy()
    w = ols_svd(X, y)
    assert np.allclose(w, ols_qr(X, y)), "Q2a: on a full-rank X this is ordinary least squares"

    # The same column twice. inv(X^T X) raises, or returns garbage, depending on
    # rounding. The SVD has a sensible answer, and so does scikit-learn, which
    # gives it to you without a word.
    t = DAYS["temp_c"].to_numpy()
    twin = np.column_stack([np.ones(150), t, t])
    wt = ols_svd(twin, y)
    assert abs(wt[1] - wt[2]) < 1e-8, "Q2a: identical columns get identical weights"
    assert np.allclose(twin @ wt, project(twin[:, :2], y)), "Q2a: and the predictions of one column"
    sk = LinearRegression(fit_intercept=False).fit(twin, y).coef_
    assert np.allclose(wt, sk), "Q2a: this is what LinearRegression silently returns"

    w_c, w_f = w[1], w[2]
    assert w_c < -5 and w_f > 5, "Q2b: fit on all five FEATURES, and read the two sensors"
    assert abs(warming_effect(w_c, w_f) - 4) < 0.5, "Q2b: both sensors move, by different amounts"
    print(f"Q2 ok   temp_c {w_c:+.1f}, temp_f {w_f:+.1f}, warming by 1 C {warming_effect(w_c, w_f):+.2f}")


check_q2()

# %%
# ---------------------------------------------------------------------------
# Q3. Ridge, through the SVD, with an intercept
# ---------------------------------------------------------------------------
# (a) shrink_factors(s, alpha) returns s^2 / (s^2 + alpha) for each singular
#     value.
#
# (b) effective_df(s, alpha) returns their sum: how many dimensions the fit is
#     really using.
#
# (c) ridge_fit(X, y, alpha) takes X WITHOUT a column of ones and returns
#     (w0, w). The reading's formula assumes centered data, and scikit-learn
#     does not penalize the intercept. So center X and y, apply the formula to
#     get w, and recover w0 from the means. Do not call Ridge.

def shrink_factors(s, alpha):
    pass


def effective_df(s, alpha):
    pass


def ridge_fit(X, y, alpha):
    pass


def check_q3():
    assert np.allclose(shrink_factors([10.0, 1.0, 0.1], 1.0), [100 / 101, 0.5, 0.01 / 1.01]), "Q3a"
    assert abs(effective_df([3.0, 2.0, 1.0], 0.0) - 3) < 1e-12, "Q3b: alpha 0 uses every dimension"
    assert effective_df([3.0, 2.0, 1.0], 10.0) < effective_df([3.0, 2.0, 1.0], 1.0), "Q3b: more penalty, fewer"

    X = DAYS[FEATURES].to_numpy()
    y = DAYS["trips"].to_numpy()
    for alpha in [0.1, 10.0, 1000.0]:
        w0, w = ridge_fit(X, y, alpha)
        sk = Ridge(alpha=alpha).fit(X, y)
        assert np.allclose(w, sk.coef_, atol=1e-6), f"Q3c: coefficients at alpha = {alpha}"
        assert abs(w0 - sk.intercept_) < 1e-6, f"Q3c: intercept at alpha = {alpha}"

    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    print("Q3 ok   singular values", s.round(1), " effective df at alpha 100:",
          round(effective_df(s, 100.0), 2))


check_q3()

# %%
# ---------------------------------------------------------------------------
# Q4. The penalty reads your units
# ---------------------------------------------------------------------------
# (a) as_fraction(df) returns a COPY with humidity as a fraction (0.3 to 0.95)
#     rather than a percent. df must be unchanged.
#
# (b) prediction_change(make_model, train, test) makes one model with
#     make_model(), fits it on train[FEATURES] and predicts test[FEATURES]; makes
#     a second, fits it on as_fraction(train)[FEATURES] and predicts
#     as_fraction(test)[FEATURES]. Both target train["trips"]. It returns the
#     largest absolute difference between the two sets of predictions. The data
#     are the same; only the way one column is written has changed.
#
# (c) scaled_ridge(alpha) returns a pipeline: StandardScaler, then Ridge.
#
# (d) rmse(y, yhat), and sensor_report(train, clean, broken) which fits
#     LinearRegression and scaled_ridge(10.0) on train[FEATURES] and returns a dict
#     with the RMSE on "trips" under the keys
#     "ols_clean", "ols_broken", "ridge_clean", "ridge_broken".

def as_fraction(df):
    pass


def prediction_change(make_model, train, test):
    pass


def scaled_ridge(alpha):
    pass


def rmse(y, yhat):
    pass


def sensor_report(train, clean, broken):
    pass


def check_q4():
    before = DAYS["humidity"].iloc[0]
    frac = as_fraction(DAYS)
    assert DAYS["humidity"].iloc[0] == before, "Q4a: DAYS must not be modified"
    assert frac["humidity"].max() <= 1 and frac["temp_c"].equals(DAYS["temp_c"]), "Q4a"

    ols = prediction_change(LinearRegression, DAYS, NEW_DAYS)
    bare = prediction_change(lambda: Ridge(alpha=100.0), DAYS, NEW_DAYS)
    pipe = prediction_change(lambda: scaled_ridge(100.0), DAYS, NEW_DAYS)
    assert ols < 1e-6, "Q4b: least squares does not care how a column is written"
    assert bare > 10, "Q4b: bare ridge does: the same day, a different prediction"
    assert pipe < 1e-6, "Q4c: standardizing first removes the choice"

    r = sensor_report(DAYS, NEW_DAYS, BROKEN_DAYS)
    assert abs(r["ols_clean"] - r["ridge_clean"]) < 1, "Q4d: on clean days the two tie"
    assert r["ols_broken"] > 2 * r["ridge_broken"], "Q4d: when the sensor breaks, they do not"
    print(f"Q4 ok   largest change: OLS {ols:.1e}, bare ridge {bare:.1f} trips, scaled {pipe:.1e}")
    print("        RMSE:", {k: round(v, 1) for k, v in r.items()})


check_q4()

# %%
# ---------------------------------------------------------------------------
# Q5. Cross-validation can be lied to
# ---------------------------------------------------------------------------
# NOISE_X is 60 people and 2,000 columns of random numbers. NOISE_Y is a coin
# flip, exactly half 1s. Nothing in NOISE_X says anything about NOISE_Y, so the
# honest accuracy of any model here is 0.5.
#
# (a) top_k(X, y, k) returns the column indices of the k columns with the largest
#     absolute correlation with y, best first. No loops over columns.
#
# (b) leaky_cv(X, y, k, folds) picks the top k columns using ALL the rows, and
#     then returns the mean cross_val_score of LogisticRegression() on those
#     columns, with cv=folds.
#
# (c) honest_cv(X, y, k, folds) loops over folds.split(X). In each fold it picks
#     the top k columns using the TRAINING rows only, fits LogisticRegression()
#     on them, and scores on the held-out rows. It returns the mean of the fold
#     scores.

_noise = np.random.default_rng(6103)
NOISE_X = _noise.normal(size=(60, 2000))
NOISE_Y = _noise.permutation(np.repeat([0, 1], 30))
FOLDS = KFold(5, shuffle=True, random_state=6103)


def top_k(X, y, k):
    pass


def leaky_cv(X, y, k, folds):
    pass


def honest_cv(X, y, k, folds):
    pass


def check_q5():
    A = np.column_stack([np.arange(6.0), [1.0, -1, 1, -1, 1, -1], [5.0, 4, 3, 2, 1, 0.5]])
    v = np.arange(6.0) * 2 + 1
    assert list(top_k(A, v, 2))[0] == 0 and 2 in list(top_k(A, v, 2)), "Q5a: |correlation|, best first"

    leaky = leaky_cv(NOISE_X, NOISE_Y, 20, FOLDS)
    honest = honest_cv(NOISE_X, NOISE_Y, 20, FOLDS)
    assert leaky > 0.8, "Q5b: select on ALL rows, then cross-validate"
    assert abs(honest - 25 / 60) < 0.02, "Q5c: select inside each fold, on its training rows"
    assert abs(honest_cv(NOISE_X, NOISE_Y, 10, FOLDS) - 26 / 60) < 0.02, "Q5c: on its TRAINING rows"
    print(f"Q5 ok   on pure noise: leaky {leaky:.3f}, honest {honest:.3f}, truth 0.500")


check_q5()

# %%
# ---------------------------------------------------------------------------
# Q6. Accuracy does not notice
# ---------------------------------------------------------------------------
# (a) binary_log_loss(y, p, eps=1e-15) is the loss from the reading, averaged
#     over the rows. Clip p to [eps, 1 - eps] first, so that a confident mistake
#     costs a lot rather than infinity. Do not import it.
#
# (b) poly_logit(degree, C) returns a pipeline: PolynomialFeatures(degree,
#     include_bias=False), StandardScaler, LogisticRegression(C=C, max_iter=20000).
#
# (c) score_by_C(train, test, degree, Cs) takes (X, y) tuples and returns a
#     DataFrame with one row per C, indexed by C, with the columns
#     "train_acc", "test_acc", "train_loss", "test_loss". Loss is binary_log_loss
#     on predict_proba's probability of class 1.

MOONS = make_moons(100, noise=0.3, random_state=6103)
MOONS_NEW = make_moons(2000, noise=0.3, random_state=6104)
CS = [0.01, 0.1, 1.0, 10.0, 100.0, 10000.0]


def binary_log_loss(y, p, eps=1e-15):
    pass


def poly_logit(degree, C):
    pass


def score_by_C(train, test, degree, Cs):
    pass


def check_q6():
    assert abs(binary_log_loss([1], [0.5]) - np.log(2)) < 1e-12, "Q6a: a coin flip costs ln 2"
    assert binary_log_loss([1, 0], [1.0, 0.0]) < 1e-12, "Q6a: right and certain costs nothing"
    assert 30 < binary_log_loss([1], [0.0]) < 40, "Q6a: clip, so wrong and certain is large, not infinite"

    t = score_by_C(MOONS, MOONS_NEW, 9, CS)
    assert list(t.columns) == ["train_acc", "test_acc", "train_loss", "test_loss"], "Q6c: the columns"
    assert t["train_loss"].is_monotonic_decreasing, "Q6c: less penalty always fits the training set better"
    assert t["test_loss"].idxmin() == 1.0, "Q6c: new days' loss bottoms out at C = 1"
    assert t["test_acc"].idxmax() == 100.0, "Q6c: accuracy prefers a different C"
    assert t.loc[10000.0, "test_loss"] > 3 * t.loc[1.0, "test_loss"], "Q6c: overconfidence is what it costs"
    print("Q6 ok")
    print(t.round(3))


check_q6()

# %%
# ---------------------------------------------------------------------------
# Q7. Choosing the degree and C, once
# ---------------------------------------------------------------------------
# (a) choose(train, degrees, Cs, folds) tries every pair, scoring poly_logit by
#     cross-validated log loss on train only: cross_val_score with cv=folds and
#     scoring="neg_log_loss", and flip the sign. It returns the best
#     (degree, C, cv_loss), lowest loss winning.
#
# (b) final_loss(train, test, degree, C) refits that one model on all of train
#     and returns binary_log_loss on test. This is the single time test is used.

CHOOSE = make_moons(300, noise=0.3, random_state=6103)
CHOOSE_NEW = make_moons(2000, noise=0.3, random_state=6104)


def choose(train, degrees, Cs, folds):
    pass


def final_loss(train, test, degree, C):
    pass


def check_q7():
    d, C, cv = choose(CHOOSE, [1, 2, 3, 5, 7, 9], [0.01, 1.0, 100.0, 10000.0], FOLDS)
    assert (d, C) == (3, 100.0), "Q7a: the pair with the lowest cross-validated log loss"
    assert 0.2 < cv < 0.35, "Q7a: report the loss, with its sign flipped"
    final = final_loss(CHOOSE, CHOOSE_NEW, d, C)
    assert abs(final - 0.218) < 0.01, "Q7b: fit on all of train, score on test"
    print(f"Q7 ok   degree {d}, C {C:g}: cross-validated loss {cv:.3f}, test loss {final:.3f}")


check_q7()

# %%
# ---------------------------------------------------------------------------
# Q8. Say what it means
# ---------------------------------------------------------------------------
# (a) A colleague reads your Q2 fit and writes: "Each extra degree Celsius costs
#     the station about 9 trips." The model predicts well on new days. Say why the
#     sentence is still false, and give the number from Q2 that you would report
#     instead.
#
# (b) Q5's leaky_cv never touched a held-out row while fitting a model. So where
#     did the held-out rows get in? Answer in terms of what the 0.85 is an estimate
#     of.
#
# (c) HW NumPy asked what NumPy made easier to get wrong than plain lists. HW
#     Pandas asked what pandas made easier to get wrong than NumPy. Now the next
#     one: name one thing SCIKIT-LEARN makes easier to get wrong than pandas did.
#     Name the call and the specific case. "Machine learning is hard" is not an
#     answer, and neither is "nothing".

Q8_SIGN = ""

Q8_LEAK = ""

Q8_NEW_DANGER = ""


def check_q8():
    assert len(Q8_SIGN.split()) >= 25, "Q8a: say why, and give the number"
    assert len(Q8_LEAK.split()) >= 25, "Q8b: say where they got in"
    assert len(Q8_NEW_DANGER.split()) >= 20, "Q8c: name the call and the case"
    print("Q8 ok")


check_q8()

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
