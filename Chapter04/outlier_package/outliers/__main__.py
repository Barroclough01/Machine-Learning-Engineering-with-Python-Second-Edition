from .utils.data import create_data
from .detectors.detection_models import DetectionModels
from .detectors import pipelines
from .definitions import MODEL_CONFIG_PATH
import pandas as pd


if __name__ == "__main__":
    data = create_data()
    models = DetectionModels(MODEL_CONFIG_PATH).get_models()
    for model in models:
        detector = pipelines.OutlierDetector(model=model)
        result = detector.detect(data)
        df_data_results = pd.DataFrame(data=zip(data, result), columns=['features', 'result'])
        print(len(df_data_results[df_data_results['result']==-1]))
        print(len(result[result==-1]))