from pipeline import MLPipeline
from knn import KNN
from cnn import CNN
# import visualisation

SAVELOCATION = '../data/posters/'
RAW_MOVIES = pd.read_csv('../data/poster_data.csv')
TODAY = pd.Timestamp(year=2020, month=3, day=10)
MOVIES = pd.read_csv('../data/clean_poster_data.csv', index_col=0)

PREPROCESSED_DATA = '../data/preprocessed/'

MLPipelineHandler = MLPipeline(verbose=True)
# MLPipelineHandler.ingestData()

MLPipelineHandler.loadPreprocessedData(PREPROCESSED_DATA)

