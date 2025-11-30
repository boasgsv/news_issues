import pathlib

# Paths
BASE_DIR = pathlib.Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data Sources
IPTC_ISSUES_PATH = DATA_DIR / "iptc_node_issues.parquet"
IPTC_GDOWN_ID = "18I5Xf2Gx6effyfri8BnoT2DHIemMKxw6"
IPTC_GDOWN_ID = "18I5Xf2Gx6effyfri8BnoT2DHIemMKxw6"

# Datasets
DATASET_ID_NYT = "tumanovalexander/nyt-articles-data"
DATASET_ID_LARGE = "jordankrishnayah/45m-headlines-from-2007-2022-10-largest-sites"

# Selection: 'nyt' or 'large'
SELECTED_DATASET = 'large' 

# Model Parameters
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
FAISS_INDEX_DIM = 768
SIMILARITY_THRESHOLD = 0.55
TOP_K_ISSUES = 3

# Output Files
HEADLINES_WITH_SENTIMENT_PATH = OUTPUT_DIR / "headlines_with_sentiment.csv"
EDGES_OUTPUT_PATH = OUTPUT_DIR / "edges.csv"
NODES_OUTPUT_PATH = OUTPUT_DIR / "nodes.csv"
GRAPH_IMAGE_PATH = OUTPUT_DIR / "graph_visualization.html"

# Execution
SAMPLE_SIZE = 1000 # Number of headlines to process for testing
