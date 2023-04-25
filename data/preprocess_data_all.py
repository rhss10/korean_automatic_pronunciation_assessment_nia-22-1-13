import json

COUNTRY = "cj"
SPLIT = "val"
SCORE = "compreh"
PATH = "/data2/rhss10/speech_corpus/NIA-13/nia13"
MIN = 1417
new_file = open(f"./{SPLIT}_{SCORE}_{COUNTRY}_noRW_BAL.csv", "w")
labels = {}

with open(f"{PATH}/{COUNTRY}/{SCORE}/{SPLIT}.csv", "r") as org_file:
    next(org_file)
    print("path,compreh,fluency,pronun,topik,prof", file=new_file)
    for line in org_file:
        score, path = line.split(",")
        json_path = (
            f"{PATH}/{COUNTRY}/lab" + path.split("sound")[-1].split(".wav")[0] + ".json"
        )
        with open(json_path, "r") as json_file:
            spk_dict = json.load(json_file)
            compreh = spk_dict["EvaluationMetadata"]["ComprehendEval"]
            fluency = spk_dict["EvaluationMetadata"]["FluencyEval"]
            pronun = spk_dict["EvaluationMetadata"]["PronunProfEval"]
            prof_level = spk_dict["SpeakerMetadata"]["proficiency"]
            topik_level = spk_dict["SpeakerMetadata"]["topik_level"]
        assert score == str(compreh)  # NOTE: change compreh to custom score
        # NOTE: change compreh to custom score labels.get(compreh, 0) < MIN
        if fluency != 0:
            print(
                path.strip(),
                compreh,
                fluency,
                pronun,
                topik_level,
                prof_level,
                sep=",",
                file=new_file,
            )
            # NOTE: change compreh to custom score
            labels[compreh] = labels.get(compreh, 0) + 1


new_file.close()
print(labels)
print("- New csv file made")
