from typing import List
import crf
import BiLSTM_CRF

class Solution:
    def crf_predict(self, sentences: List[str]) -> List[str]:
        model = crf.CRF()
        model.loadModel(crf.CHECKPOINTPATH)
        results = model.predict(sentences)
        return results

    def dnn_predict(self, sentences: List[str]) -> List[str]:
        model = BiLSTM_CRF.load_model()
        results = BiLSTM_CRF.predict(model,sentences)
        return results
