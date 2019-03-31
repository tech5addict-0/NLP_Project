import numpy as np
import re
from logger import Logger


def cosine_similarity_by_vector(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def calculate_alignment_score(file, word1, word2):
    min_score = -10
    maxscore = 10
    with open(file, 'r') as ppdb_lex:
        lines = ppdb_lex.readlines()
        matches = re.findall(word1,lines)
        if matches:
            final_matches = re.findall(word2,matches)
            if final_matches:
                for line in final_matches:
                    match_build = line.split("|||")
                    if ((match_build[1] == word1 & match_build[2] == word2) | (match_build[1] == word2 & match_build[2] == word1)):
                        #calc results
                        metrics = [metric for metric in match_build[3].split(" ")]
                        scorings = {key_val[0]: int(key_val[1]) for key_val in [scoring.split("=") for scoring in metrics]}
                        alignment_score = -np.log(scorings['p(f|e)']) -np.log(scorings['p(f|e)']) - np.log(scorings['p(e|f,LHS)']) - np.log(scorings['p(f|e,LHS)']) + 0.3 * (-np.log(scorings['p(LHS|e)'])) + 0.3 * (-np.log(scorings['p(LHS|f)'])) + 100 * scorings['RarityPenalty']
                        return alignment_score
