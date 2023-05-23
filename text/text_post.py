from collections import namedtuple
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import re


def text_ocr_filter(ocr_results, img_size=(2560, 1432)):
    important_ocr_results = []
    detection_threshold = 0.4
    width_threshold = (
        0.4  # the image could be cropped in pre instead for faster analysis
    )
    height_limit_top = 0.2  # here the top and bottom are also cropped from the analysis
    height_limit_bottom = 0.8

    # I am aware that the names dont corresspond to the indexed object but to the actual object contained at the indice but this is more descriptive.
    candidates = list(
        filter(
            lambda bounding_box_left_top: bounding_box_left_top[0][0][1]
            < height_limit_bottom * img_size[1]
            and bounding_box_left_top[0][0][1] > height_limit_top * img_size[1],
            list(
                filter(
                    lambda bounding_box_left_top: bounding_box_left_top[0][0][0]
                    < width_threshold * img_size[0],
                    list(
                        filter(lambda ocr_probability: ocr_probability[-1], ocr_results)
                    ),
                )
            ),
        )
    )  # im unsure if casting the inner filter into a list is necessary
    kills = _extract_kills(_allign_group(candidates))
    # important_ocr_results.append(kills)
    return kills  # important_ocr_results


def _allign_group(candidates, vertical_noise_amplitude_pixels=5):
    """here we group texts which are vertically alligned to form 'phrase' groups"""
    df_candidates = pd.DataFrame(
        [[candidate[0][0][1], candidate[1]] for candidate in candidates],
        columns=["y_top_left", "string"],
    )
    only_top_left = pd.DataFrame(df_candidates["y_top_left"])
    model = AgglomerativeClustering(
        distance_threshold=2 * vertical_noise_amplitude_pixels,
        linkage="complete",
        n_clusters=None,
    ).fit(only_top_left)

    result = (
        df_candidates.assign(
            group=model.labels_,
            center=df_candidates.groupby(model.labels_).transform(
                lambda x: (x.max() + x.min()) / 2
            ),
        )
        .groupby(["group"])
        .agg({"string": " ".join, "y_top_left": "first"})
    )

    return (
        result  # , model.labels_.max()  # returning grouped object and no of groups -1
    )


def _extract_kills(candidate_groups):
    """parsing of alligned phrase groups to obtain kill data"""
    # we are only interested if the group has 4 or 2 members
    Kill = namedtuple("Kill", ["killer", "killed", "gold_earned" "shared_with"])
    kills = list(
        candidate_groups[
            candidate_groups.string.str.contains("(\d{2,}|(hero))$", regex=True)
        ].string
    )
    extracted_kills = []
    for kill in kills:
        if kill[-1:-4] == "hero":
            nos = list(map(int, re.findall("\d{2,}", kill)))
            peeps = re.findall("\w+\s\[.*?\]", kill)
            extracted_kills.append(Kill(peeps[0], peeps[1], nos[:-1], nos[-1]))

    return extracted_kills
