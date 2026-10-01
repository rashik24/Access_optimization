#!/usr/bin/env python3
"""
M1_single_period_spatial.py

Standalone SINGLE-PERIOD SPATIAL food-access right-sizing model.

This is the spatial baseline for later comparison with the temporal models.

IMPORTANT
---------
This model does NOT use opening hours, weeks, day/time windows, or temporal
neighbor compatibility in optimization.

Decision variables
------------------
S[A] : monthly allocation to agency A
y[A] : 1 if agency A is retained

Accessibility
-------------
x_j = sum_A (S[A] / ED[A]) * exp(-beta * d_jA)

Equity
------
Lower absolute mean semideviation (AMSD) across connected tracts.

Protection
----------
1. Every baseline-connected tract must retain at least one feasible agency.
2. A removed agency must have at least one retained agency within the
   static substitution radius.

The script solves:
  - maximum mean access
  - minimum AMSD
  - theta tradeoffs for theta in THETA_VALUES

No PreserveTotalAccess constraint is used.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

import run_utils as ru                                   # NEW
# ============================================================
# 0. USER CONFIGURATION
# ============================================================

HOME = Path("/home/rsiddiq2/FBCENC_Temporal_Access/New_run")
DATA = HOME / "data"

AGENCY_FILE = DATA / "FBCENC_opencage_clean.csv"
ODM_EXISTING_FILE = DATA / "ODM FBCENC 2.csv"
TRACT_INFO_FILE = DATA / "MMG NC Tract .xlsx"
RUCA_FILE = DATA / "GeoID RUCA.csv"
CAPACITY_FILE = DATA / "capacity_stats_fbcenc.csv"
GEO_FILE = DATA / "GEOID_GEO_CODE.xlsx"
AGENCY_DISTANCE_FILE = DATA / "FBCENC_agency_pairwise_distances.csv"

# Branch / spatial settings
FILT = ["R","N","G","S","W","D"]
YEAR = 2022
URBAN_THRESHOLD = 20
RURAL_THRESHOLD = 30
BETA = 0.10
NEIGHBOR_RADIUS_MILES = 15.0

FORCE_REBUILD_AGENCY_DISTANCES = False

# Current intended formulation
USE_PJ_ACCESS = False
USE_EXTRA_IMPEDANCE_ED = False

S_MIN_MONTHLY = 10.0
THETA_VALUES = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8,0.9,1.0]

EPS = 1e-9

OUTPUT_FILE = HOME / "M1_single_period_spatial.xlsx"


def haversine_miles(lat1, lon1, lat2, lon2):
    R_earth = 3958.8

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    aa = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * math.atan2(math.sqrt(aa), math.sqrt(1.0 - aa))
    return R_earth * c


def safe_name(x):
    """
    Short Gurobi-safe-ish label.
    """
    return (
        str(x)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("-", "_")
    )


def precompute_agency_distance_file(
    agency_file,
    distance_file,
    force_rebuild=False
):
    """
    Compute the complete agency-to-agency great-circle distance table ONCE.

    Output CSV columns
    ------------------
    Agency_A
    Agency_B
    Distance_Miles

    Only one unordered pair is stored (A,B), not both (A,B) and (B,A).

    If distance_file already exists and force_rebuild=False,
    nothing is recomputed.
    """

    distance_path = Path(distance_file)

    if distance_path.exists() and not force_rebuild:
        print(
            f"\nUsing existing precomputed agency-distance file:\n"
            f"{distance_path}"
        )
        return str(distance_path)

    print("\nPrecomputing agency-to-agency distances...")

    ag = pd.read_csv(agency_file).copy()

    ag["Name"] = ag["Name"].astype(str).str.strip()
    ag["Latitude"] = pd.to_numeric(
        ag["Latitude"],
        errors="coerce"
    )
    ag["Longitude"] = pd.to_numeric(
        ag["Longitude"],
        errors="coerce"
    )

    ag = (
        ag
        .dropna(
            subset=["Name", "Latitude", "Longitude"]
        )
        .drop_duplicates(subset=["Name"])
        .reset_index(drop=True)
    )

    names = ag["Name"].tolist()

    lat = ag["Latitude"].to_numpy(dtype=float)
    lon = ag["Longitude"].to_numpy(dtype=float)

    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            d = haversine_miles(
                lat[i],
                lon[i],
                lat[j],
                lon[j]
            )

            rows.append({
                "Agency_A": names[i],
                "Agency_B": names[j],
                "Distance_Miles": float(d)
            })

    distance_df = pd.DataFrame(
        rows,
        columns=[
            "Agency_A",
            "Agency_B",
            "Distance_Miles"
        ]
    )

    distance_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    distance_df.to_csv(
        distance_path,
        index=False
    )

    print(
        f"Saved {len(distance_df):,} agency pairs to:\n"
        f"{distance_path}"
    )

    return str(distance_path)


def load_agency_distance(
    distance_file,
    agencies=None,
    max_distance=None
):
    """
    Load PRECOMPUTED pairwise distances and create the symmetric
    dictionary used by the optimization.

    Parameters
    ----------
    distance_file : str
        Long-form CSV created by precompute_agency_distance_file().

    agencies : iterable, optional
        If provided, keep only pairs for agencies that are actually
        in the optimization model.

    max_distance : float, optional
        If provided, load only pairs within this distance.

        This is useful here because temporal-neighbor protection only
        needs neighbors within NEIGHBOR_RADIUS_MILES.

    Returns
    -------
    dist_agency : dict
        dist_agency[A][B] = distance in miles
    """

    df = pd.read_csv(distance_file).copy()

    required = {
        "Agency_A",
        "Agency_B",
        "Distance_Miles"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Precomputed distance file is missing columns: "
            f"{sorted(missing)}"
        )

    df["Agency_A"] = (
        df["Agency_A"]
        .astype(str)
        .str.strip()
    )

    df["Agency_B"] = (
        df["Agency_B"]
        .astype(str)
        .str.strip()
    )

    df["Distance_Miles"] = pd.to_numeric(
        df["Distance_Miles"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Agency_A",
            "Agency_B",
            "Distance_Miles"
        ]
    )

    if agencies is not None:
        Aset = set(agencies)

        df = df[
            df["Agency_A"].isin(Aset)
            & df["Agency_B"].isin(Aset)
        ].copy()

        dist_agency = {
            A: {}
            for A in Aset
        }

    else:
        all_agencies = set(df["Agency_A"]) | set(df["Agency_B"])

        dist_agency = {
            A: {}
            for A in all_agencies
        }

    if max_distance is not None:
        df = df[
            df["Distance_Miles"] <= float(max_distance)
        ].copy()

    # Build symmetric dictionary from one-row-per-unordered-pair CSV.
    for row in df.itertuples(index=False):

        A = row.Agency_A
        B = row.Agency_B
        d = float(row.Distance_Miles)

        dist_agency.setdefault(A, {})[B] = d
        dist_agency.setdefault(B, {})[A] = d

    print(
        "\nLoaded precomputed agency distances:"
        f"\n  agencies in dictionary : {len(dist_agency):,}"
        f"\n  stored directed links  : "
        f"{sum(len(v) for v in dist_agency.values()):,}"
    )

    if max_distance is not None:
        print(
            f"  loaded only <= {max_distance:g} miles"
        )

    return dist_agency


def prepare_spatial_data():
    """
    Reproduces the main existing-agency preprocessing from the
    single-period / prior temporal notebook.

    Returns the model-ready dictionaries.
    """

    Agency_GeoID_raw = pd.read_csv(ODM_EXISTING_FILE)

    geo_info = pd.read_excel(TRACT_INFO_FILE)
    geo_info = geo_info[geo_info["year"] == YEAR].copy()

    geo_map = pd.read_csv(RUCA_FILE)

    FBCENC_supply = pd.read_csv(AGENCY_FILE)

    capacity = pd.read_csv(CAPACITY_FILE)

    GEO = pd.read_excel(GEO_FILE)

    # --------------------------------------------------------
    # Branch filter
    # --------------------------------------------------------

    FBCENC_supply["Name"] = (
        FBCENC_supply["Name"].astype(str).str.strip()
    )

    FBCENC_supply["Warehouse_Code"] = (
        FBCENC_supply["Parent Agency No."]
        .astype(str)
        .str[0]
    )

    branch_agencies = FBCENC_supply[
        FBCENC_supply["Warehouse_Code"].isin(FILT)
    ].copy()

    branch_counties = (
        branch_agencies["FBC County Code"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.replace(r"\s+|COUNTY", "", regex=True)
        .unique()
    )
    # --------------------------------------------------------
    # GEOID cleaning
    # --------------------------------------------------------

    geo_map["GEOID_x"] = (
        geo_map["GEOID_x"].astype(str).str.split(".").str[0].str.zfill(11)
    )

    # Normalize county names: uppercase, no spaces, no "COUNTY"
    geo_map["County_x"] = (
        geo_map["County_x"].astype(str).str.upper()
        .str.replace(r"\s+|COUNTY", "", regex=True)
    )
    # RUCA stores "New Hanover" truncated to "New": repair it by FIPS code
    geo_map.loc[geo_map["GEOID_x"].str.startswith("37129"),
                "County_x"] = "NEWHANOVER"
    geo_map_branch = geo_map[
        geo_map["County_x"].isin(branch_counties)
    ].copy()

    branch_geoids = set(
        geo_map_branch["GEOID_x"].unique().tolist()
    )

    Agency_GeoID = Agency_GeoID_raw.copy()

    Agency_GeoID["Name"] = (
        Agency_GeoID["Name"].astype(str).str.strip()
    )

    Agency_GeoID["GEOID"] = (
        Agency_GeoID["GEOID"].astype(str).str.zfill(11)
    )

    geo_info["tractid"] = (
        geo_info["tractid"].astype(str).str.zfill(11)
    )

    # --------------------------------------------------------
    # Restrict tract-agency links to branch service area
    # --------------------------------------------------------

    Agency_GeoID = Agency_GeoID[
        Agency_GeoID["GEOID"].isin(branch_geoids)
    ].copy()

    Agency_GeoID = Agency_GeoID.merge(
        geo_info[["tractid", "number_food_insecure"]],
        left_on="GEOID",
        right_on="tractid",
        how="left"
    )

    Agency_GeoID = Agency_GeoID.merge(
        geo_map_branch,
        left_on="GEOID",
        right_on="GEOID_x",
        how="left"
    )

    Agency_GeoID["TravelTime_Threshold"] = np.where(
        Agency_GeoID["Urban"] == 1,
        URBAN_THRESHOLD,
        RURAL_THRESHOLD
    )

    Agency_GeoID["Total_TravelTime"] = pd.to_numeric(
        Agency_GeoID["Total_TravelTime"],
        errors="coerce"
    )

    Agency_GeoID = Agency_GeoID.dropna(
        subset=["Total_TravelTime"]
    ).copy()

    Agency_GeoID = Agency_GeoID[
        Agency_GeoID["Total_TravelTime"]
        <= Agency_GeoID["TravelTime_Threshold"]
    ].copy()

    # --------------------------------------------------------
    # Initial agency weighted demand used only to align agencies
    # --------------------------------------------------------

    Agency_GeoID["exp_weight"] = np.exp(
        -BETA * Agency_GeoID["Total_TravelTime"]
    )

    Agency_GeoID["weighted_demand"] = (
        Agency_GeoID["exp_weight"]
        * Agency_GeoID["number_food_insecure"]
    )

    agency_demand = (
        Agency_GeoID
        .groupby("Name")["weighted_demand"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------------
    # Existing agencies / monthly supply / monthly capacity
    # --------------------------------------------------------

    agency_supply = branch_agencies[
        ["Name", "Parent", "Avg_Monthly_Supply"]
    ].copy()

    agency_supply = agency_supply.merge(
        capacity[["Parent", "Mean", "P90"]],
        on="Parent",
        how="left"
    )

    agency_supply = agency_supply.merge(
        agency_demand,
        on="Name",
        how="left"
    )

    agency_supply = (
        agency_supply
        .dropna(subset=["weighted_demand", "Mean"])
        .drop_duplicates(subset=["Name"])
        .copy()
    )

    agency_supply["Mean"] = pd.to_numeric(
        agency_supply["Mean"], errors="coerce"
    )

    agency_supply["P90"] = pd.to_numeric(
        agency_supply["P90"], errors="coerce"
    )

    Aexisting = agency_supply["Name"].tolist()

    S_old = {
        A: float(v)
        for A, v in zip(
            agency_supply["Name"],
            agency_supply["Mean"]
        )
        if pd.notna(v)
    }

    C_old_max = {
        A: float(v) if pd.notna(v) else 1e12
        for A, v in zip(
            agency_supply["Name"],
            agency_supply["P90"]
        )
    }

    # Keep only aligned agencies
    Aexisting = [
        A for A in Aexisting
        if A in S_old and A in C_old_max
    ]

    Aset = set(Aexisting)

    # --------------------------------------------------------
    # Tract-agency travel times / feasible pairs
    # --------------------------------------------------------

    d_old = {}
    feasible_old_pairs = set()

    for r in Agency_GeoID.itertuples(index=False):
        j = str(r.GEOID)
        A = str(r.Name).strip()

        if A not in Aset:
            continue

        travel_time = float(r.Total_TravelTime)

        d_old.setdefault(j, {})
        d_old[j][A] = travel_time
        feasible_old_pairs.add((j, A))

    # --------------------------------------------------------
    # Tract population
    # --------------------------------------------------------

    GEO["GEOID"] = GEO["GEOID"].astype(str).str.zfill(11)

    GEO = GEO.merge(
        geo_info[["tractid", "number_food_insecure"]],
        left_on="GEOID",
        right_on="tractid",
        how="left"
    )

    GEO_branch = GEO[
        GEO["GEOID"].isin(branch_geoids)
    ].copy()

    Pj_population = dict(zip(
        GEO_branch["GEOID"],
        GEO_branch["number_food_insecure"]
    ))

    valid_population = [
        float(v)
        for v in Pj_population.values()
        if pd.notna(v)
    ]

    if not valid_population:
        raise ValueError(
            "No valid tract food-insecure population values."
        )

    mean_population = float(np.mean(valid_population))

    Pj_population = {
        str(j): (
            mean_population
            if pd.isna(v)
            else float(v)
        )
        for j, v in Pj_population.items()
    }

    # ED should still use actual tract population.
    Pj_ED = Pj_population.copy()

    # Access can be population weighted or uniform.
    if USE_PJ_ACCESS:
        Pj_access = Pj_population.copy()
    else:
        Pj_access = {
            j: 1.0
            for j in Pj_population
        }

    # --------------------------------------------------------
    # Connected tracts
    # --------------------------------------------------------

    tracts_with_old = {
        str(j)
        for j, A in feasible_old_pairs
    }

    I_all = list(Pj_population.keys())

    I = [
        str(j)
        for j in I_all
        if str(j) in tracts_with_old
    ]

    Iset = set(I)

    feasible_old_pairs = {
        (str(j), A)
        for j, A in feasible_old_pairs
        if str(j) in Iset and A in Aset
    }

    pairs_by_j = {}

    for j, A in feasible_old_pairs:
        pairs_by_j.setdefault(j, []).append(A)

    # --------------------------------------------------------
    # Access impedance exp(-beta*d)
    # --------------------------------------------------------

    exp_imp = {}

    for j in I:
        exp_imp[j] = {}

        for A in pairs_by_j.get(j, []):
            dist = d_old.get(j, {}).get(A, np.nan)

            if not np.isnan(dist):
                exp_imp[j][A] = float(
                    np.exp(-BETA * dist)
                )

    # --------------------------------------------------------
    # Competition weights G_ja
    #
    # Keep prior structure:
    # exp(-beta*d_ja) * sqrt(S_old[A])
    # --------------------------------------------------------

    G = {}

    for j in I:
        weights = {}
        denom = 0.0

        for A in pairs_by_j.get(j, []):
            if A not in S_old:
                continue

            tij = d_old[j][A]

            weight = (
                np.exp(-BETA * tij)
                * np.sqrt(max(S_old[A], 0.0))
            )

            weights[A] = float(weight)
            denom += float(weight)

        if denom > EPS:
            G[j] = {
                A: weight / denom
                for A, weight in weights.items()
            }
        else:
            G[j] = {}

    # --------------------------------------------------------
    # Monthly expected demand ED_A
    #
    # If USE_EXTRA_IMPEDANCE_ED=False:
    #   ED_A = sum_j Pj * G_ja
    #
    # If True:
    #   ED_A = sum_j Pj * G_ja * exp(-beta*d_ja)
    # --------------------------------------------------------

    ED = {}

    for j in I:
        pj = float(Pj_ED.get(j, 0.0))

        if pj <= 0:
            continue

        for A, gij in G.get(j, {}).items():
            term = pj * float(gij)

            if USE_EXTRA_IMPEDANCE_ED:
                term *= exp_imp[j][A]

            ED[A] = ED.get(A, 0.0) + term

    ED = {
        A: max(float(v), EPS)
        for A, v in ED.items()
    }

    # Remove agencies with no usable ED.
    # This prevents S/ED from being undefined.
    Aexisting = [
        A for A in Aexisting
        if A in ED
    ]

    Aset = set(Aexisting)

    feasible_old_pairs = {
        (j, A)
        for j, A in feasible_old_pairs
        if A in Aset
    }

    pairs_by_j = {}

    for j, A in feasible_old_pairs:
        pairs_by_j.setdefault(j, []).append(A)

    # Keep only tracts with at least one aligned agency.
    I = [
        j for j in I
        if len(pairs_by_j.get(j, [])) > 0
    ]

    print("\n=== SPATIAL MODEL DATA ===")
    print("Branches:", FILT)
    print("Existing agencies:", len(Aexisting))
    print("Connected tracts:", len(I))
    print("Feasible tract-agency pairs:",
          len(feasible_old_pairs))
    print("Monthly baseline supply:",
          f"{sum(S_old[A] for A in Aexisting):,.2f}")

    return {
        "Aexisting": Aexisting,
        "I": I,
        "S_old": S_old,
        "C_old_max": C_old_max,
        "d_old": d_old,
        "feasible_old_pairs": feasible_old_pairs,
        "pairs_by_j": pairs_by_j,
        "exp_imp": exp_imp,
        "G": G,
        "ED": ED,
        "Pj_ED": Pj_ED,
        "Pj_access": Pj_access,
        "Agency_GeoID": Agency_GeoID,
        "agency_supply": agency_supply,
        "geo_info": geo_info,
        "geo_map_branch": geo_map_branch,
    }


# ============================================================
# 5. STATIC SPATIAL NEIGHBORS
# ============================================================

def build_static_neighbors(Aexisting, dist_agency, radius_miles):
    """
    N_static[A] = agencies B within radius_miles of A.

    This is purely spatial. Opening hours are not used.
    """
    Aset = set(Aexisting)

    N_static = {}

    for A in Aexisting:
        N_static[A] = sorted([
            B
            for B, d in dist_agency.get(A, {}).items()
            if (
                B in Aset
                and B != A
                and float(d) <= float(radius_miles)
            )
        ])

    return N_static


# ============================================================
# 6. SINGLE-PERIOD SPATIAL MODEL
# ============================================================

def solve_single_period_spatial_model(
    Aexisting,
    I,
    S_old,
    C_old_max,
    ED,
    pairs_by_j,
    exp_imp,
    N_static,
    theta=0.5,
    objective_mode="max_mean",
    X_zenith=None,
    X_nadir=None,
    AMSD_zenith=None,
    AMSD_nadir=None,
    log_to_console=False,
    s_min_monthly=S_MIN_MONTHLY,
):
    """
    Pure single-period spatial optimization.

    No temporal data enter this model.
    """

    m = Model("single_period_spatial")
    m.Params.LogToConsole = 1 if log_to_console else 0
    m.Params.FeasibilityTol = 1e-8
    m.Params.OptimalityTol = 1e-8
    m.Params.Threads = 0
    m.Params.MIPGap = 0.005

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    S = m.addVars(
        Aexisting,
        lb=0.0,
        name="S_monthly"
    )

    y = m.addVars(
        Aexisting,
        vtype=GRB.BINARY,
        name="retain"
    )

    total_supply = float(
        sum(S_old[A] for A in Aexisting)
    )

    # --------------------------------------------------------
    # Supply / capacity
    # --------------------------------------------------------

    m.addConstr(
        quicksum(S[A] for A in Aexisting)
        == total_supply,
        name="MonthlySupplyBalance",
    )

    m.addConstrs(
        (
            S[A] <= float(C_old_max[A]) * y[A]
            for A in Aexisting
        ),
        name="MonthlyCapacity",
    )

    m.addConstrs(
        (
            S[A] >= float(s_min_monthly) * y[A]
            for A in Aexisting
        ),
        name="MinimumRetainedSupply",
    )

    # --------------------------------------------------------
    # Tract accessibility
    # --------------------------------------------------------

    access = {}

    for j in I:
        access[j] = quicksum(
            (
                S[A] / max(float(ED[A]), EPS)
            )
            * float(exp_imp[j][A])
            for A in pairs_by_j.get(j, [])
            if (
                A in ED
                and A in exp_imp.get(j, {})
            )
        )

    nI = len(I)

    avg_access = (
        1.0 / nI
    ) * quicksum(
        access[j]
        for j in I
    )
        # NEW: mean access may not fall below baseline
    base_mean = float(np.mean([
        sum(
            S_old[A] / max(float(ED[A]), EPS) * float(exp_imp[j][A])
            for A in pairs_by_j.get(j, [])
            if A in ED and A in exp_imp.get(j, {})
        )
        for j in I
    ]))
    m.addConstr(
        avg_access >= base_mean * (1 - 1e-9),
        name="PreserveTotalAccess",
    )
    # --------------------------------------------------------
    # Lower AMSD
    # --------------------------------------------------------

    v = m.addVars(
        I,
        lb=0.0,
        name="lower_dev"
    )

    for j in I:
        m.addConstr(
            v[j] >= avg_access - access[j],
            name=f"LowerDev_{j}",
        )

    AMSD = (
        1.0 / nI
    ) * quicksum(
        v[j]
        for j in I
    )

    # --------------------------------------------------------
    # Static tract coverage
    # --------------------------------------------------------

    for j in I:
        available = [
            A
            for A in pairs_by_j.get(j, [])
            if A in Aexisting
        ]

        if available:
            m.addConstr(
                quicksum(
                    y[A]
                    for A in available
                ) >= 1,
                name=f"TractCoverage_{j}",
            )

    # --------------------------------------------------------
    # Static spatial neighbor protection
    #
    # 1 - y_A <= sum_{B in N_A} y_B
    #
    # If A has no spatial substitute inside the radius,
    # A is retained.
    # --------------------------------------------------------

    for A in Aexisting:

        neighbors = N_static.get(A, [])

        if neighbors:
            m.addConstr(
                1 - y[A]
                <= quicksum(
                    y[B]
                    for B in neighbors
                ),
                name=f"StaticNeighbor_{safe_name(A)}",
            )
        else:
            m.addConstr(
                y[A] == 1,
                name=f"UniqueSpatialAgency_{safe_name(A)}",
            )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    if objective_mode == "max_mean":

        m.setObjective(
            avg_access,
            GRB.MAXIMIZE
        )

    elif objective_mode == "min_amsd":

        m.setObjective(
            AMSD,
            GRB.MINIMIZE
        )

    elif objective_mode == "tradeoff":

        required = [
            X_zenith,
            X_nadir,
            AMSD_zenith,
            AMSD_nadir,
        ]

        if any(x is None for x in required):
            raise ValueError(
                "Tradeoff requires payoff-table values."
            )

        X_range = max(
            float(X_zenith) - float(X_nadir),
            EPS
        )

        D_range = max(
            float(AMSD_nadir) - float(AMSD_zenith),
            EPS
        )

        X_score = (
            avg_access - float(X_nadir)
        ) / X_range

        D_score = (
            float(AMSD_nadir) - AMSD
        ) / D_range

        m.setObjective(
            float(theta) * X_score
            + (1.0 - float(theta)) * D_score,
            GRB.MAXIMIZE,
        )

    else:
        raise ValueError(
            f"Unknown objective_mode={objective_mode}"
        )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            "Single-period spatial model not optimal. "
            f"Gurobi status={m.Status}"
        )

    # --------------------------------------------------------
    # Extract solution
    # --------------------------------------------------------

    S_new = {
        A: float(S[A].X)
        for A in Aexisting
    }

    retained = {
        A: int(round(y[A].X))
        for A in Aexisting
    }

    access_j = {
        j: float(access[j].getValue())
        for j in I
    }

    vals = np.array(
        [access_j[j] for j in I],
        dtype=float
    )

    mu_literal = float(vals.mean())

    amsd_literal = float(
        np.mean(
            np.maximum(
                mu_literal - vals,
                0.0
            )
        )
    )

    agency_df = pd.DataFrame([
        {
            "Agency": A,
            "Retained": retained[A],
            "S_old_monthly": float(S_old[A]),
            "S_new_monthly": float(S_new[A]),
            "Capacity_monthly": float(C_old_max[A]),
            "Monthly_ED": float(ED[A]),
            "Static_Neighbor_Count":
                len(N_static.get(A, [])),
        }
        for A in Aexisting
    ])

    tract_df = pd.DataFrame([
        {
            "GEOID": j,
            "Access": access_j[j],
        }
        for j in I
    ])

    metrics = {
        "objective_mode": objective_mode,
        "theta": float(theta),
        "overall_mean_access":
            float(avg_access.getValue()),
        "overall_AMSD":
            float(AMSD.getValue()),
        "overall_mean_access_literal":
            mu_literal,
        "overall_AMSD_literal":
            amsd_literal,
        "total_supply":
            total_supply,
        "total_allocated":
            float(sum(S_new.values())),
        "retained_agencies":
            int(sum(retained.values())),
        "removed_agencies":
            int(
                len(Aexisting)
                - sum(retained.values())
            ),
    }
    metrics.update(ru.solver_stats(m))                   # NEW
    return {
        "model": m,
        "metrics": metrics,
        "S_monthly": S_new,
        "retained": retained,
        "access_j": access_j,
        "agency_df": agency_df,
        "tract_df": tract_df,
    }


# ============================================================
# 7. PAYOFF TABLE + THETA SWEEP
# ============================================================

def run_spatial_analysis(
    spatial,
    N_static,
    theta_values=THETA_VALUES,
    log_to_console=False,
):

    common = dict(
        Aexisting=spatial["Aexisting"],
        I=spatial["I"],
        S_old=spatial["S_old"],
        C_old_max=spatial["C_old_max"],
        ED=spatial["ED"],
        pairs_by_j=spatial["pairs_by_j"],
        exp_imp=spatial["exp_imp"],
        N_static=N_static,
        log_to_console=log_to_console,
        s_min_monthly=S_MIN_MONTHLY,
    )

    print("\n========================================")
    print("M1 SPATIAL: MAXIMIZE MEAN ACCESS")
    print("========================================")

    sol_mean = solve_single_period_spatial_model(
        **common,
        objective_mode="max_mean",
    )

    X_zenith = (
        sol_mean["metrics"]["overall_mean_access"]
    )

    AMSD_nadir = (
        sol_mean["metrics"]["overall_AMSD"]
    )

    print("Mean-access zenith :", X_zenith)
    print("AMSD at mean zenith:", AMSD_nadir)

    print("\n========================================")
    print("M1 SPATIAL: MINIMIZE AMSD")
    print("========================================")

    sol_amsd = solve_single_period_spatial_model(
        **common,
        objective_mode="min_amsd",
    )

    AMSD_zenith = (
        sol_amsd["metrics"]["overall_AMSD"]
    )

    X_nadir = (
        sol_amsd["metrics"]["overall_mean_access"]
    )

    print("AMSD zenith (minimum):", AMSD_zenith)
    print("Mean at AMSD zenith  :", X_nadir)

    payoff = {
        "X_zenith": X_zenith,
        "X_nadir": X_nadir,
        "AMSD_zenith": AMSD_zenith,
        "AMSD_nadir": AMSD_nadir,
    }

    theta_solutions = {}
    summary_rows = []

    for theta_i in theta_values:

        print("\n========================================")
        print(
            f"M1 SPATIAL TRADEOFF theta={theta_i}"
        )
        print("========================================")

        sol = solve_single_period_spatial_model(
            **common,
            theta=float(theta_i),
            objective_mode="tradeoff",
            X_zenith=X_zenith,
            X_nadir=X_nadir,
            AMSD_zenith=AMSD_zenith,
            AMSD_nadir=AMSD_nadir,
        )

        theta_solutions[float(theta_i)] = sol
        ru.save_checkpoint(RUN_DIR, f"theta_{theta_i:.2f}", sol)   # NEW
        metrics = sol["metrics"]

        access_score = (
            metrics["overall_mean_access"]
            - X_nadir
        ) / max(
            X_zenith - X_nadir,
            EPS
        )

        equity_score = (
            AMSD_nadir
            - metrics["overall_AMSD"]
        ) / max(
            AMSD_nadir - AMSD_zenith,
            EPS
        )

        weighted_score = (
            float(theta_i) * access_score
            + (1.0 - float(theta_i))
            * equity_score
        )

        summary_rows.append({
            "Model": "M1_SPATIAL",
            "Theta": float(theta_i),
            "Overall_Mean_Access":
                metrics["overall_mean_access"],
            "Overall_AMSD":
                metrics["overall_AMSD"],
            "Access_Score":
                access_score,
            "Equity_Score":
                equity_score,
            "Weighted_Score":
                weighted_score,
            "Retained_Agencies":
                metrics["retained_agencies"],
            "Removed_Agencies":
                metrics["removed_agencies"],
            "Total_Supply":
                metrics["total_supply"],
            "Total_Allocated":
                metrics["total_allocated"],
        })

    theta_summary = pd.DataFrame(
        summary_rows
    )

    print("\n========================================")
    print("M1 SPATIAL THETA SWEEP SUMMARY")
    print("========================================")
    print(
        theta_summary.to_string(index=False)
    )

    return {
        "mean_solution": sol_mean,
        "amsd_solution": sol_amsd,
        "theta_solutions": theta_solutions,
        "theta_summary": theta_summary,
        "payoff": payoff,
    }


# ============================================================
# 8. VALIDATION
# ============================================================

def validate_solution(
    spatial,
    solution,
    N_static,
    tol=1e-6,
):
    """
    Independent post-solve checks.
    """

    Aexisting = spatial["Aexisting"]
    I = spatial["I"]
    pairs_by_j = spatial["pairs_by_j"]
    C_old_max = spatial["C_old_max"]

    S = solution["S_monthly"]
    y = solution["retained"]

    violations = []

    total_supply = sum(
        spatial["S_old"][A]
        for A in Aexisting
    )

    total_alloc = sum(
        S[A]
        for A in Aexisting
    )

    if abs(total_alloc - total_supply) > tol * max(1.0, total_supply):
        violations.append("supply_balance")

    for A in Aexisting:

        if S[A] < -tol:
            violations.append(
                f"negative_supply::{A}"
            )

        if S[A] > C_old_max[A] * y[A] + tol:
            violations.append(
                f"capacity::{A}"
            )

        if S[A] + tol < S_MIN_MONTHLY * y[A]:
            violations.append(
                f"minimum_supply::{A}"
            )

        if y[A] == 0:

            if S[A] > tol:
                violations.append(
                    f"removed_positive_supply::{A}"
                )

            neighbors = N_static.get(A, [])

            if not any(
                y.get(B, 0) == 1
                for B in neighbors
            ):
                violations.append(
                    f"neighbor::{A}"
                )

    for j in I:

        available = pairs_by_j.get(j, [])

        if available and not any(
            y.get(A, 0) == 1
            for A in available
        ):
            violations.append(
                f"tract_coverage::{j}"
            )

    metrics = solution["metrics"]

    if abs(
        metrics["overall_mean_access"]
        - metrics["overall_mean_access_literal"]
    ) > 1e-6:
        violations.append(
            "mean_access_reconstruction"
        )

    if abs(
        metrics["overall_AMSD"]
        - metrics["overall_AMSD_literal"]
    ) > 1e-6:
        violations.append(
            "AMSD_reconstruction"
        )

    return violations


# ============================================================
# 9. EXPORT
# ============================================================

def export_results(
    results,
    spatial,
    N_static,
    output_file,
):

    payoff_df = pd.DataFrame([
        results["payoff"]
    ])

    theta_summary = (
        results["theta_summary"].copy()
    )

    neighbor_rows = []

    for A in spatial["Aexisting"]:
        for B in N_static.get(A, []):
            neighbor_rows.append({
                "Agency": A,
                "Neighbor": B,
            })

    neighbor_df = pd.DataFrame(
        neighbor_rows
    )

    validation_rows = []

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        payoff_df.to_excel(
            writer,
            sheet_name="Payoff_Table",
            index=False,
        )

        theta_summary.to_excel(
            writer,
            sheet_name="Theta_Summary",
            index=False,
        )

        for theta_i, sol in (
            results["theta_solutions"].items()
        ):

            tag = f"{theta_i:.1f}"

            pd.DataFrame([
                sol["metrics"]
            ]).to_excel(
                writer,
                sheet_name=f"Metrics_t{tag}",
                index=False,
            )

            sol["agency_df"].to_excel(
                writer,
                sheet_name=f"Agency_t{tag}",
                index=False,
            )

            sol["tract_df"].to_excel(
                writer,
                sheet_name=f"Tract_t{tag}",
                index=False,
            )

            violations = validate_solution(
                spatial=spatial,
                solution=sol,
                N_static=N_static,
            )

            validation_rows.append({
                "Theta": theta_i,
                "Passed": len(violations) == 0,
                "Violation_Count":
                    len(violations),
                "Violations":
                    " | ".join(violations),
            })

        neighbor_df.to_excel(
            writer,
            sheet_name="Static_Neighbors",
            index=False,
        )
                # NEW: baseline spatial access (current supply, all agencies open)
        base = {
            j: sum(
                spatial["S_old"][A] / max(spatial["ED"][A], EPS)
                * spatial["exp_imp"][j][A]
                for A in spatial["pairs_by_j"].get(j, [])
                if A in spatial["ED"] and A in spatial["exp_imp"].get(j, {})
            )
            for j in spatial["I"]
        }
        vals = np.array(list(base.values()))
        mu = vals.mean()
        pd.DataFrame({"GEOID": list(base), "Baseline_Access": vals}).to_excel(
            writer, sheet_name="Baseline_Tract", index=False)
        pd.DataFrame([{
            "Mean_Access": mu,
            "AMSD": np.mean(np.maximum(mu - vals, 0.0)),
            "Agencies": len(spatial["Aexisting"]),
            "Tracts": len(spatial["I"]),
        }]).to_excel(writer, sheet_name="Baseline_Summary", index=False)

        pd.DataFrame(
            validation_rows
        ).to_excel(
            writer,
            sheet_name="Validation",
            index=False,
        )

    print(
        f"\nSaved M1 spatial results to:\n"
        f"{output_file}"
    )


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    RUN_DIR = ru.make_run_dir(HOME / "runs", "M1")       # NEW
    ru.snapshot(RUN_DIR, __file__, globals(), [          # NEW
        AGENCY_FILE, ODM_EXISTING_FILE, TRACT_INFO_FILE, RUCA_FILE,
        CAPACITY_FILE, GEO_FILE, AGENCY_DISTANCE_FILE,
    ])

    # Spatial data only.
    spatial = prepare_spatial_data()

    # Build/load agency-to-agency spatial distances.
    precompute_agency_distance_file(
        agency_file=AGENCY_FILE,
        distance_file=AGENCY_DISTANCE_FILE,
        force_rebuild=
            FORCE_REBUILD_AGENCY_DISTANCES,
    )

    dist_agency = load_agency_distance(
        distance_file=AGENCY_DISTANCE_FILE,
        agencies=spatial["Aexisting"],
        max_distance=NEIGHBOR_RADIUS_MILES,
    )

    N_static = build_static_neighbors(
        Aexisting=spatial["Aexisting"],
        dist_agency=dist_agency,
        radius_miles=NEIGHBOR_RADIUS_MILES,
    )

    print("\n=== STATIC NEIGHBOR SUMMARY ===")
    counts = [
        len(N_static.get(A, []))
        for A in spatial["Aexisting"]
    ]

    print(
        "Agencies with zero spatial neighbors:",
        sum(c == 0 for c in counts)
    )
    print(
        "Mean static neighbors:",
        float(np.mean(counts))
    )

    results = run_spatial_analysis(
        spatial=spatial,
        N_static=N_static,
        theta_values=THETA_VALUES,
        log_to_console=False,
    )

    export_results(
        results=results,
        spatial=spatial,
        N_static=N_static,
        output_file=OUTPUT_FILE,
    )

    ru.save_checkpoint(RUN_DIR, "all_results", results)  # NEW
    ru.archive(OUTPUT_FILE, RUN_DIR)                     # NEW

