import utils
import pandas as pd
import numpy as np

class BaselineClassifier():
    def get_overlaps(self, data_dict):
        overlaps = []
        for claimId in data_dict:
            for articleId in data_dict[claimId]["articles"]:
                article = data_dict[claimId]["articles"][articleId]
                stance = article[1]
                headline = article[0]
                claim = data_dict[claimId]["claim"]
                overlaps.append([claimId, stance, utils.calculate_overlap(claim,headline)])
                colnames = ["claimId", "stance","overlap"]
        return pd.DataFrame(overlaps, columns=colnames)

    #only for training data
    def calculate_classifier_thresholds(self, training_df):
        mean_overlap_per_stance = training_df.groupby(by="stance")['overlap'].agg(np.mean).values
        minFor = np.max(mean_overlap_per_stance)
        maxAgainst = np.min(mean_overlap_per_stance)
        thresholds = {}
        thresholds["minFor"] = minFor
        thresholds["maxAgainst"] = maxAgainst
        return thresholds

    def predict(self, thresholds, test_data):
        predicted_labels = ["for" if overlap > thresholds["minFor"] else "against" if overlap <= thresholds["maxAgainst"] else "observing" for overlap in test_data]
        return predicted_labels