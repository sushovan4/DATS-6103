# %% [markdown]
# # DATS 6103 -- Homework: Pandas
#
# Due Tuesday, October 6, before class.
#
# Work in this file, run it top to bottom in a fresh interpreter before you
# submit, and upload it with your prompt log and critique (see the assignment
# page). Every function below must run without editing the tests underneath it.
#
# Q1 to Q4 are the September 22 session -- alignment, selection, axes, and what
# read_csv guesses. Q5 to Q8 are September 29 -- missing values, duplicates,
# merging, reshaping. You can do the first half now.
#
# The thread running through all of it: NumPy raised when you got a shape
# wrong. Pandas does not. It lines your data up on labels you did not check,
# skips values you did not know were missing, and returns a table of the
# wrong size without a word. Every question below is one of those, and the
# last one asks you to name the one that would have caught you out.
#
# An AI assistant is allowed and encouraged. What is not optional is the
# disclosure, and the critique is graded alongside the code.

# %%
import io

import numpy as np
import pandas as pd

# Four Capital Bikeshare stations, and a September of trips out of them. Every
# question below works on these two frames, so run this cell first and leave it
# alone. Look at both before you start: the station ids are strings with
# leading zeros, one trip names a station that is not in STATIONS, one trip id
# appears twice, and four trips have no duration.
STATIONS = pd.DataFrame({
    "station_id": ["00031", "00042", "00107", "00219"],
    "name": ["Eastern Market", "Union Station", "Columbia Heights", "Anacostia"],
    "ward": [6, 6, 1, 8],
})

TRIPS = pd.DataFrame({
    "trip_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1010, 1011],
    "station_id": ["00031", "00042", "00107", "00031", "00219", "00042",
                   "00107", "00031", "00999", "00042", "00042", "00219"],
    "rider": ["member", "casual", "member", "member", "casual", "member",
              "casual", "member", "casual", "casual", "casual", "member"],
    "minutes": [12.0, np.nan, 9.5, 14.0, np.nan, 7.0,
                np.nan, 21.0, 33.0, np.nan, 18.0, 11.5],
})

# %%
# ---------------------------------------------------------------------------
# Q1. Addition lines up on the index
# ---------------------------------------------------------------------------
# `aug` and `sep` below hold trip counts for overlapping but different sets of
# stations. Two stations ran in only one of the two months.
#
# (a) strict_total(a, b) returns their plain sum.
#
# (b) filled_total(a, b) returns the sum with a station absent from one month
#     counted as zero for that month. Neither function may modify a or b.
#
# (c) only_one(a, b) returns a SORTED LIST of the labels that appear in exactly
#     one of the two indexes.
#
# (a) and (b) are two different questions, and pandas will answer whichever one
# you typed without mentioning that the other exists.

def strict_total(a, b):
    pass


def filled_total(a, b):
    pass


def only_one(a, b):
    pass


aug = pd.Series([120, 90, 145], index=["00031", "00042", "00107"])
sep = pd.Series([110, 95, 130], index=["00042", "00107", "00219"])

st = strict_total(aug, sep)
assert len(st) == 4, "Q1a: the result has one entry per label in either index"
assert int(st.isna().sum()) == 2, "Q1a: a station in only one month cannot be added"

ft = filled_total(aug, sep)
assert len(ft) == 4 and int(ft.isna().sum()) == 0, "Q1b: nothing missing after fill_value"
assert float(ft["00031"]) == 120.0 and float(ft["00219"]) == 130.0, "Q1b: absent means zero"
assert float(ft["00042"]) == 200.0, "Q1b: a station in both months is still added"
assert list(aug.index) == ["00031", "00042", "00107"], "Q1: aug must not be modified"

assert only_one(aug, sep) == ["00031", "00219"], "Q1c: the labels in exactly one index"
print("Q1 ok")

# %%
# ---------------------------------------------------------------------------
# Q2. A label, a position, and two kinds of boolean
# ---------------------------------------------------------------------------
# (a) by_label(s, k) returns the entry LABELLED k; by_position(s, k) returns
#     the entry in POSITION k. `docks` has an integer index that does not match
#     its positions, so the two answers differ. Plain s[k] is not an answer to
#     either: with an integer index pandas reads it as a label and with any
#     other index as a position, so the same expression means two things
#     depending on data you may not have looked at.
#
# (b) busy_rows(df, threshold) returns the ROWS whose "trips" value is above
#     the threshold -- a boolean Series, one entry per row.
#
# (c) masked(df, threshold) returns df[df > threshold], the boolean DataFrame.
#     Write the one-liner, then read the asserts: this is not row selection,
#     and it is the mistake that follows from expecting NumPy's mask.

def by_label(s, k):
    pass


def by_position(s, k):
    pass


def busy_rows(df, threshold):
    pass


def masked(df, threshold):
    pass


docks = pd.Series([15, 19, 11, 23], index=[3, 1, 4, 2])
assert int(by_label(docks, 2)) == 23, "Q2a: loc takes the label 2"
assert int(by_position(docks, 2)) == 11, "Q2a: iloc takes the third entry"

counts = pd.DataFrame({"trips": [120, 90, 145, 60], "docks": [15, 19, 11, 23]},
                      index=["00031", "00042", "00107", "00219"])

b = busy_rows(counts, 100)
assert b.shape == (2, 2), "Q2b: a boolean Series selects rows"
assert list(b.index) == ["00031", "00107"], "Q2b: the two stations over 100 trips"
assert b["trips"].dtype == counts["trips"].dtype, "Q2b: selecting rows keeps the dtype"

m = masked(counts, 100)
assert m.shape == counts.shape, "Q2c: a boolean DataFrame keeps every row and column"
assert int(m.isna().sum().sum()) == 6, "Q2c: every value that failed the test became NaN"
assert m["docks"].dtype == np.dtype("float64"), "Q2c: NaN forced the integers to floats"
print("Q2 ok")

# %%
# ---------------------------------------------------------------------------
# Q3. Which axis did you mean, and what got skipped
# ---------------------------------------------------------------------------
# `ride` is average trip length, three stations by two weeks, with one gap.
#
# (a) per_station(df) returns one number per STATION.
# (b) per_week(df) returns one number per WEEK.
# (c) per_week_strict(df) returns one number per week, but a week with any gap
#     in it must come back as NaN rather than as an average of the rest.
#
# The axis argument names the axis you are REMOVING. Getting it backwards
# raises nothing: you asked for the average of three stations and got the
# average of one station's two weeks, and both are numbers.

def per_station(df):
    pass


def per_week(df):
    pass


def per_week_strict(df):
    pass


ride = pd.DataFrame({"week1": [12.0, 8.0, np.nan], "week2": [15.0, 9.0, 21.0]},
                    index=["00031", "00042", "00107"])

ps = per_station(ride)
assert list(ps.index) == list(ride.index), "Q3a: one number per station"
assert float(ps["00031"]) == 13.5 and float(ps["00107"]) == 21.0, "Q3a: the mean across the weeks"

pw = per_week(ride)
assert list(pw.index) == ["week1", "week2"], "Q3b: one number per week"
assert float(pw["week1"]) == 10.0, "Q3b: the default skips the gap and averages the two present"

pws = per_week_strict(ride)
assert bool(np.isnan(pws["week1"])), "Q3c: skipna=False makes the gap propagate"
assert float(pws["week2"]) == 15.0, "Q3c: a column with no gap is unaffected"
print("Q3 ok")

# %%
# ---------------------------------------------------------------------------
# Q4. What read_csv decided without asking
# ---------------------------------------------------------------------------
# RAW is four columns of a station file: an id with leading zeros, a date, a
# count written with a thousands separator, and a note whose missing values are
# spelled "N/A (unknown)".
#
# (a) load_careless(text) reads it with no arguments at all. Do not fix
#     anything here -- the asserts under it record what plain read_csv did, and
#     the point is that none of it raised.
#
# (b) load_careful(text) reads the same text so that the id keeps its zeros,
#     the date is a date, the count is a number, and the sentinel is missing.
#     Four keyword arguments, one per problem.
#
# (c) round_trip(df) writes df to CSV and reads it straight back, with no
#     Unnamed: 0 column in the result. Use io.StringIO rather than a file.

RAW = (
    "station_id,opened,trips,note\n"
    '00031,2010-09-20,"1,204",N/A (unknown)\n'
    '00042,2011-03-14,"986",clean\n'
    '00107,2010-10-05,"1,530",N/A (unknown)\n'
)


def load_careless(text):
    pass


def load_careful(text):
    pass


def round_trip(df):
    pass


bad = load_careless(RAW)
assert bad["station_id"].iloc[0] == 31, "Q4a: read_csv turned the id into a number"
assert bad["trips"].dtype == object, "Q4a: '1,204' is text, so the whole column is text"

good = load_careful(RAW)
assert good["station_id"].iloc[0] == "00031", "Q4b: the leading zeros survive"
assert pd.api.types.is_numeric_dtype(good["trips"]), "Q4b: thousands= makes it a number"
assert int(good["trips"].sum()) == 3720, "Q4b: 1204 + 986 + 1530"
assert pd.api.types.is_datetime64_any_dtype(good["opened"]), "Q4b: a date, not a string"
assert int(good["note"].isna().sum()) == 2, "Q4b: the sentinel is missingness, not data"

back = round_trip(STATIONS)
assert list(back.columns) == list(STATIONS.columns), "Q4c: no Unnamed: 0 souvenir"
assert back.shape == STATIONS.shape, "Q4c: same table out as in"
print("Q4 ok")

# %%
# ---------------------------------------------------------------------------
# Q5. Two ways to deal with a gap, and what each one costs
# ---------------------------------------------------------------------------
# Four of the twelve trips have no duration, and they are not a random four.
#
# (a) complete_cases(df) returns the rows whose "minutes" is present.
# (b) mean_filled(df) returns a COPY with the gaps in "minutes" filled by that
#     column's mean. df itself must not change.
# (c) casual_share(df) returns the fraction of rows whose rider is "casual".
#
# Then read the last four asserts. They are the question: one of these two
# treatments changes who is in your sample, the other changes how spread out
# your data looks, and neither prints a warning. Q9(d) asks you to pick one and
# defend it.

def complete_cases(df):
    pass


def mean_filled(df):
    pass


def casual_share(df):
    pass


kept = complete_cases(TRIPS)
filled = mean_filled(TRIPS)

assert len(kept) == 8, "Q5a: four trips have no duration"
assert len(filled) == len(TRIPS), "Q5b: filling keeps every row"
assert int(filled["minutes"].isna().sum()) == 0, "Q5b: nothing missing afterwards"
assert not TRIPS["minutes"].isna().sum() == 0, "Q5b: and TRIPS itself is untouched"

assert abs(casual_share(TRIPS) - 0.5) < 1e-12, "Q5c: half the trips are casual"
assert casual_share(kept) < casual_share(TRIPS), \
    "Q5c: dropping the rows changed who is in the sample"

assert abs(filled["minutes"].mean() - kept["minutes"].mean()) < 1e-9, \
    "Q5d: filling with the mean leaves the mean alone"
assert filled["minutes"].std() < kept["minutes"].std(), \
    "Q5d: and shrinks the spread, which is the part nobody notices"
print("Q5 ok")

# %%
# ---------------------------------------------------------------------------
# Q6. The row logged twice, and the category nobody coded
# ---------------------------------------------------------------------------
# (a) dedupe_trips(df) returns the trips with no trip_id appearing twice. Trip
#     1010 was logged once with no duration and again, corrected, with 18.0.
#     Keep the corrected one. Which occurrence you keep is an argument, not a
#     default.
#
# (b) encode(s, mapping) returns s with each value replaced by its code.
#
# (c) ...and REFUSES rather than returns when the mapping has no code for some
#     value in s: raise ValueError naming the values it could not code. The
#     plain .map() answer returns NaN for them, which is a missing value you
#     manufactured yourself and will later fill in, average over, or drop.

def dedupe_trips(df):
    pass


def encode(s, mapping):
    pass


deduped = dedupe_trips(TRIPS)
assert len(deduped) == 11, "Q6a: trip 1010 was logged twice"
assert int(deduped["trip_id"].duplicated().sum()) == 0, "Q6a: no id survives twice"
assert float(deduped.loc[deduped["trip_id"] == 1010, "minutes"].iloc[0]) == 18.0, \
    "Q6a: keep='last' keeps the corrected row, not the first one"

codes = encode(TRIPS["rider"], {"member": 0, "casual": 1})
assert list(codes) == [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0], "Q6b: member 0, casual 1"
assert pd.api.types.is_integer_dtype(codes), "Q6b: every row mapped, so the codes are integers"

try:
    encode(pd.Series(["member", "casual", "day pass"]), {"member": 0, "casual": 1})
except ValueError as e:
    assert "day pass" in str(e), "Q6c: say which category has no code"
else:
    raise AssertionError("Q6c: a forgotten category must raise, not return NaN")
print("Q6 ok")

# %%
# ---------------------------------------------------------------------------
# Q7. Count the rows before and after
# ---------------------------------------------------------------------------
# (a) merge_checked(left, right, on) joins right onto left, keeping every row
#     of left, and raises ValueError if the result does not have exactly as
#     many rows as left did. Put both counts in the message.
#
# (b) unmatched(merged, col) returns how many rows of the merge found nothing
#     on the right, counted through `col`.
#
# A merge is the one operation in this homework that can make your dataset
# BIGGER, and it does it silently: `doubled` below has station 00042 listed
# twice, which is a data error nobody put there on purpose, and every trip from
# that station then appears twice in the join and twice in every total you
# compute afterwards.

def merge_checked(left, right, on):
    pass


def unmatched(merged, col):
    pass


joined = merge_checked(TRIPS, STATIONS, "station_id")
assert len(joined) == len(TRIPS), "Q7a: a left join on a unique key keeps the rows"
assert "ward" in joined.columns, "Q7a: the right-hand columns came along"
assert unmatched(joined, "ward") == 1, "Q7b: one trip names a station that is not in STATIONS"

doubled = pd.concat([STATIONS, STATIONS.iloc[[1]]], ignore_index=True)
try:
    merge_checked(TRIPS, doubled, "station_id")
except ValueError as e:
    assert "16" in str(e), "Q7c: say what the count became -- four trips from 00042, so 16"
else:
    raise AssertionError("Q7c: a key matching twice multiplies rows and must be caught")
print("Q7 ok")

# %%
# ---------------------------------------------------------------------------
# Q8. One variable per column
# ---------------------------------------------------------------------------
# WIDE has two variables hiding in each column name: "jan_trips" is a month and
# a measure glued together.
#
# (a) tidy(wide) returns a long frame with one row per station-month and one
#     column per measure: station_id, month, docks, trips. Melt, split the
#     variable name, and put it back. Melting alone is not enough -- it leaves
#     you a column holding "jan_trips", which is the same problem in a new
#     shape.
#
# (b) total_by(long) returns trips by station and month from LONG, which has
#     station 00031 logged twice in January. The TOTAL, 120, not the average.
#     pivot_table aggregates with the mean unless you tell it otherwise, so if
#     you reached for it because pivot complained about duplicates, you have
#     just averaged something without deciding to.

WIDE = pd.DataFrame({
    "station_id": ["00031", "00042"],
    "jan_trips": [120, 90],
    "jan_docks": [15, 19],
    "feb_trips": [135, 88],
    "feb_docks": [15, 21],
})

LONG = pd.DataFrame({
    "station_id": ["00031", "00031", "00042"],
    "month": ["jan", "jan", "jan"],
    "trips": [60, 60, 90],
})


def tidy(wide):
    pass


def total_by(long):
    pass


t = tidy(WIDE)
assert t.shape == (4, 4), "Q8a: two stations by two months, with two measures"
assert set(t.columns) == {"station_id", "month", "docks", "trips"}, \
    "Q8a: month and measure were one column name glued together"
row = t[(t["station_id"] == "00031") & (t["month"] == "feb")]
assert float(row["trips"].iloc[0]) == 135.0, "Q8a: the values have to land in the right cell"

tb = total_by(LONG)
assert float(tb.loc["00031", "jan"]) == 120.0, \
    "Q8b: pivot_table defaults to the mean -- 60, not 120 -- unless you say aggfunc"
print("Q8 ok")

# %%
# ---------------------------------------------------------------------------
# Q9. Prose
# ---------------------------------------------------------------------------
# (a) For each expression below, decide whether the result has a different
#     number of rows from TRIPS. Decide first, from what the operation has to
#     do; then check yourself.

EXPRESSIONS = [
    'TRIPS.dropna(subset=["minutes"])',
    'TRIPS.drop_duplicates(subset="trip_id", keep="last")',
    'TRIPS.merge(doubled, on="station_id", how="left")',
    'TRIPS.sort_values("minutes")',
]

Q9_CHANGES_ROWS = [None, None, None, None]     # replace each with True or False

truth = [
    len(TRIPS.dropna(subset=["minutes"])) != len(TRIPS),
    len(TRIPS.drop_duplicates(subset="trip_id", keep="last")) != len(TRIPS),
    len(TRIPS.merge(doubled, on="station_id", how="left")) != len(TRIPS),
    len(TRIPS.sort_values("minutes")) != len(TRIPS),
]
assert all(isinstance(x, bool) for x in Q9_CHANGES_ROWS), "Q9a: four booleans"
assert Q9_CHANGES_ROWS == truth, f"Q9a: not right -- the truth is {truth}"

# (b) One sentence: what do the ones that change the row count have in common,
#     and what does the odd one out not do that the others do?

Q9_WHY = ""

# (c) Last time, Q7(c) asked what NumPy made easier to get wrong than plain
#     lists did. Now the next one: name one thing PANDAS makes easier to get
#     wrong than NumPy did. Name the operation and the specific case -- "pandas
#     is confusing" is not an answer, and neither is "nothing".

Q9_NEW_DANGER = ""

# (d) Q5 gave you two treatments for the missing durations. Pick one for THIS
#     data and defend it in a sentence or two, with a number from Q5 in it. A
#     third option -- keep the rows and leave the gaps -- is also defensible.

Q9_DECISION = ""

assert len(Q9_WHY.split()) >= 15, "Q9b: one real sentence, not a fragment"
assert len(Q9_NEW_DANGER.split()) >= 20, "Q9c: name the operation and the case"
assert len(Q9_DECISION.split()) >= 20, "Q9d: name the choice and what it costs"
print("Q9 ok")

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
