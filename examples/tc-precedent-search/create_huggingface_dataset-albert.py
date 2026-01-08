import pandas as pd

df = pd.read_csv(
    "/Users/ag2435/sci_llm/src/sci-llm/examples/Tc/supercon-arxiv.csv", dtype=str
)

df = df[df["property_name"].isin(["tc", "tcn"]) & ~df["file"].isna()]
# import pdb; pdb.set_trace()

df["tc_tcn"] = df.apply(
    lambda row: (
        row["property_name"],
        f"{row['property_value']} {row['property_unit']}",
        row["file"],
        row["year"],
    ),
    axis=1,
)


def agg_tc_tcn(x: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Aggregate the tc_tcn values for a given material."""
    x_list = list(x)
    # sort by year
    x_list.sort(key=lambda x: x[3])
    return x_list


properties = df.groupby("material", as_index=False)[["refno", "tc_tcn"]].agg(
    {"refno": "first", "tc_tcn": agg_tc_tcn}
)

both = properties["tc_tcn"].apply(
    lambda x: ("tc" in [x_[0] for x_ in x]) and ("tcn" in [x_[0] for x_ in x])
)

print(properties[both])
# save to csv
properties[both].to_csv(
    "/Users/ag2435/sci_llm/src/sci-llm/examples/Tc/supercon-arxiv-tc-tcn.csv",
    index=False,
)
