import argparse
import json
from pprint import pprint

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset, load_from_disk
from scipy.stats import pearsonr


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Data statistics"))
    parser.add_argument("--score", type=str, default="compreh")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--country", type=str, default="eu")

    args = parser.parse_args()

    return args


def statistics(args):
    ds = load_dataset(
        "csv",
        delimiter=",",
        data_files=f"/data1/nia13/{args.country}/{args.score}/{args.split}.csv",
        split="train",
    )

    tot_labels = {}
    tot_prof = {}
    tot_topik = {}
    prof_dict = {"Beginner": 0, "Intermediate": 1, "Advance": 2, "Fluent": 3}
    lab_per_prof = {"Beginner": {}, "Intermediate": {}, "Advance": {}, "Fluent": {}}
    lab_per_topik = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}

    labels_no0 = []
    prof_no0 = []
    topik_no0 = []
    labels = []
    prof = []

    for i in range(len(ds)):
        # extract information from json file
        audio_path, score = ds[i]["path"], ds[i][args.score]
        json_path = (
            f"/data1/nia13/{args.country}/lab"
            + audio_path.split("sound")[-1].split(".wav")[0]
            + ".json"
        )
        with open(json_path, "r") as json_file:
            spk_dict = json.load(json_file)
            if args.score == "compreh":
                org_score = spk_dict["EvaluationMetadata"]["ComprehendEval"]
            elif args.score == "fluency":
                org_score = spk_dict["EvaluationMetadata"]["FluencyEval"]
            elif args.score == "pronun":
                org_score = spk_dict["EvaluationMetadata"]["PronunProfEval"]
            prof_level = spk_dict["SpeakerMetadata"]["proficiency"]
            topik_level = spk_dict["SpeakerMetadata"]["topik_level"]

        assert score == org_score

        # count total labels, prof, topik
        tot_labels[org_score] = tot_labels.get(org_score, 0) + 1
        tot_prof[prof_level] = tot_prof.get(prof_level, 0) + 1
        tot_topik[topik_level] = tot_topik.get(topik_level, 0) + 1

        # count labels per self-proficiency and topik level
        lab_per_prof[prof_level][org_score] = (
            lab_per_prof[prof_level].get(org_score, 0) + 1
        )
        lab_per_topik[topik_level][org_score] = (
            lab_per_topik[topik_level].get(org_score, 0) + 1
        )

        # used for pearson correlation and linear regression plots b/t labels/prof/topik
        if topik_level != 0:
            labels_no0.append(org_score)
            prof_no0.append(prof_dict[prof_level])
            topik_no0.append(topik_level)
        labels.append(org_score)
        prof.append(prof_dict[prof_level])

    print(f"==={args.country} {args.score} {args.split}===")
    print("LABELS:", tot_labels)
    print("PROF:", tot_prof)
    print("TOPIK:", tot_topik)

    print("LABELS SUM:", sum(tot_labels.values()))
    print("PROF SUM:", sum(tot_prof.values()))
    print("TOPIK SUM:", sum(tot_topik.values()))

    print("LABELS PER PROF:")
    pprint(lab_per_prof)
    print("LABELS PER TOPIK:")
    pprint(lab_per_topik)

    prof_labels_res, topik_labels_res, topik_prof_res = correlation_plot(
        args, labels_no0, prof_no0, topik_no0, labels, prof
    )
    print("PROF-LABELS PCC:", prof_labels_res)
    print("TOPIK-LABELS PCC (NO 0 LEVEL):", topik_labels_res)
    print("TOPIK-PROF PCC: (NO 0 LEVEL)", topik_prof_res)


# plot linear regression b/t labels/prof/topik
def correlation_plot(args, labels_no0, prof_no0, topik_no0, labels, prof):
    fig, ax = plt.subplots(figsize=(8, 4))
    prof_labels_res = pearsonr(prof, labels)[0]
    topik_labels_res = pearsonr(topik_no0, labels_no0)[0]
    topik_prof_res = pearsonr(topik_no0, prof_no0)[0]

    g = sns.regplot(
        x=prof,
        y=labels,
        line_kws={"color": "#91bfdb", "alpha": 1.0, "lw": 2},
        ci=None,
        scatter=False,
        label=f"Prof-Labels Corr (PCC: {prof_labels_res:.2f})",
    )
    g = sns.regplot(
        x=topik_no0,
        y=labels_no0,
        line_kws={"color": "#d73027", "alpha": 1.0, "lw": 2},
        ci=None,
        scatter=False,
        label=f"Topik-Labels Corr (PCC: {topik_labels_res:.2f})",
    )
    g = sns.regplot(
        x=topik_no0,
        y=prof_no0,
        line_kws={"color": "#fee090", "alpha": 1.0, "lw": 2},
        ci=None,
        scatter=False,
        label=f"Topik-Prof Corr (PCC: {topik_prof_res:.2f})",
    )
    g.set(ylim=(0, 5))
    g.set(xlim=(0, 6))
    g.set_title(
        f"Correlation Plot for {args.country} {args.score} {args.split} Human Evaulators",
        fontsize=12,
        fontweight="bold",
    )
    g.legend(loc="lower left", fontsize=12)

    # fig = df.plot.scatter(x="prof", y="labels", marker="o", figsize=(10, 5))
    # fig.savefig("df.png")

    plt.tight_layout()
    plt.savefig(f"corr/{args.country}_{args.score}_{args.split}.png")

    return prof_labels_res, topik_labels_res, topik_prof_res


if __name__ == "__main__":
    args = parse_arguments()
    statistics(args)

    print("- Stats finished")
