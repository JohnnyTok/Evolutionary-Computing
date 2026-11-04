import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TV Scheduling with Genetic Algorithm", layout="wide")
st.title("📺 TV Program Scheduling using a Genetic Algorithm")
st.caption(
    "Find the program for each hourly slot (06:00–23:00) that maximises total "
    "audience rating. Set CO_R and MUT_R for each trial, run it, then compare."
)

CSV_PATH = "program_ratings_modified.csv"

# Assignment limits. The brief gives a default MUT_R of 0.2 but a range of
# 0.01-0.05. Lecturer's answer: Trial 1 uses the default (CO_R 0.8, MUT_R 0.2);
# Trials 2 and 3 must stay inside the stated range (MUT_R 0.01-0.05).
CO_MIN, CO_MAX = 0.0, 0.95
MUT_MIN = 0.01
TRIAL_CFG = {
    1: dict(co=0.8, mut=0.2, mut_max=0.20),   # default trial
    2: dict(co=0.5, mut=0.05, mut_max=0.05),  # in-range trial
    3: dict(co=0.95, mut=0.01, mut_max=0.05), # in-range trial
}

MAX_REPEATS = 2      # a program may air at most twice per day (as in the tutorial)
PENALTY = 1.0        # fitness penalty per airing above MAX_REPEATS


@st.cache_data
def load_ratings(source):
    df = pd.read_csv(source).set_index("Type of Program")
    df.columns = [int(c.split()[-1]) for c in df.columns]  # "Hour 6" -> 6
    return df


with st.sidebar:
    st.header("Data & GA settings")
    uploaded = st.file_uploader("Ratings CSV (optional)", type="csv")
    ratings_df = load_ratings(uploaded if uploaded else CSV_PATH)
    POP_SIZE = st.number_input("Population size", 10, 500, 50, step=10)
    GENERATIONS = st.number_input("Generations", 10, 1000, 100, step=10)
    ELITISM = st.number_input("Elitism (best kept each generation)", 0, 10, 2)
    SEED = st.number_input("Random seed (0 = random)", 0, 10_000, 0)

programs = list(ratings_df.index)
slots = list(ratings_df.columns)
n_slots = len(slots)
RATING = {p: list(ratings_df.loc[p]) for p in programs}

with st.expander("View ratings data (modified)"):
    st.dataframe(ratings_df, width="stretch")


# ----------------------------------------------------------------------------
# Genetic algorithm: a chromosome is a list of one program per time slot.
# ----------------------------------------------------------------------------
def fitness(schedule):
    total = sum(RATING[p][i] for i, p in enumerate(schedule))
    excess = sum(max(0, schedule.count(p) - MAX_REPEATS) for p in set(schedule))
    return total - PENALTY * excess


def total_rating(schedule):
    return sum(RATING[p][i] for i, p in enumerate(schedule))


def random_schedule():
    pool = programs * MAX_REPEATS
    random.shuffle(pool)
    return pool[:n_slots]


def crossover(a, b):
    point = random.randint(1, n_slots - 1)
    return a[:point] + b[point:], b[:point] + a[point:]


def mutate(s):
    s = s[:]
    s[random.randrange(n_slots)] = random.choice(programs)
    return s


def tournament(pop, fits, k=3):
    idx = random.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fits[i])]


def run_ga(co_r, mut_r):
    if SEED:
        random.seed(int(SEED))
    pop = [random_schedule() for _ in range(int(POP_SIZE))]
    history = []
    for _ in range(int(GENERATIONS)):
        fits = [fitness(s) for s in pop]
        ranked = sorted(zip(fits, pop), key=lambda t: t[0], reverse=True)
        history.append(ranked[0][0])
        new_pop = [s[:] for _, s in ranked[: int(ELITISM)]]
        while len(new_pop) < int(POP_SIZE):
            p1, p2 = tournament(pop, fits), tournament(pop, fits)
            if random.random() < co_r:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            if random.random() < mut_r:
                c1 = mutate(c1)
            if random.random() < mut_r:
                c2 = mutate(c2)
            new_pop.extend([c1, c2])
        pop = new_pop[: int(POP_SIZE)]
    fits = [fitness(s) for s in pop]
    best = pop[max(range(len(pop)), key=lambda i: fits[i])]
    history.append(max(fits))
    return best, max(fits), history


def schedule_table(schedule):
    return pd.DataFrame(
        {
            "Time Slot": [f"{h:02d}:00" for h in slots],
            "Program": schedule,
            "Rating": [RATING[p][i] for i, p in enumerate(schedule)],
        }
    )


# ----------------------------------------------------------------------------
# Trials
# ----------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = {}

tabs = st.tabs(["Trial 1", "Trial 2", "Trial 3", "Comparison"])

for t, tab in zip([1, 2, 3], tabs[:3]):
    with tab:
        c1, c2 = st.columns(2)
        co_r = c1.slider("CO_R (Crossover Rate)", CO_MIN, CO_MAX, TRIAL_CFG[t]["co"], 0.05, key=f"co{t}")
        mut_r = c2.slider("MUT_R (Mutation Rate)", MUT_MIN, TRIAL_CFG[t]["mut_max"], TRIAL_CFG[t]["mut"], 0.01, key=f"mut{t}")
        if t == 1:
            st.caption("Trial 1 uses the default values (CO_R 0.8, MUT_R 0.2).")
        else:
            st.caption("MUT_R is limited to 0.01–0.05 for this trial.")
        if st.button(f"Run Trial {t}", key=f"run{t}", type="primary"):
            with st.spinner("Evolving schedule..."):
                best, score, hist = run_ga(co_r, mut_r)
            st.session_state.results[t] = dict(co=co_r, mut=mut_r, best=best, score=score, hist=hist)

        res = st.session_state.results.get(t)
        if res:
            st.success(
                f"CO_R = {res['co']:.2f}, MUT_R = {res['mut']:.2f}  →  "
                f"Total rating = {total_rating(res['best']):.2f}"
            )
            left, right = st.columns([3, 2])
            left.dataframe(schedule_table(res["best"]), width="stretch", hide_index=True)
            right.line_chart(pd.Series(res["hist"], name="Best fitness"))

with tabs[3]:
    done = st.session_state.results
    if not done:
        st.info("Run at least one trial to see the comparison.")
    else:
        summary = pd.DataFrame(
            [
                {
                    "Trial": t,
                    "CO_R": r["co"],
                    "MUT_R": r["mut"],
                    "Total rating": round(total_rating(r["best"]), 2),
                    "Generation best first reached": r["hist"].index(max(r["hist"])),
                }
                for t, r in sorted(done.items())
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)
        st.line_chart(pd.DataFrame({f"Trial {t}": r["hist"] for t, r in done.items()}))
        best_t = max(done, key=lambda k: done[k]["score"])
        st.write(f"**Best trial so far:** Trial {best_t}")
