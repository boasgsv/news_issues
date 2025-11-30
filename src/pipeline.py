import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .config import (
    IPTC_ISSUES_PATH, IPTC_GDOWN_ID, 
    SENTENCE_TRANSFORMER_MODEL, SIMILARITY_THRESHOLD, TOP_K_ISSUES,
    HEADLINES_WITH_SENTIMENT_PATH, EDGES_OUTPUT_PATH, NODES_OUTPUT_PATH,
    SAMPLE_SIZE, SELECTED_DATASET, GRAPH_IMAGE_PATH
)
from .data import load_news_data, load_iptc_issues
from .sentiment import VaderSentimentAnalyzer
from .search import generate_embeddings, create_faiss_index, search_top_issues, l2_normalize
from .graph import build_issue_event_graph, visualize_graph
from .evaluation import calculate_silhouette_score

def run_pipeline():
    print("--- Starting News Issues Analysis Pipeline ---")
    print(f"Selected Dataset: {SELECTED_DATASET}")

    # 1. Load Data
    print("Loading data...")
    try:
        df_issues = load_iptc_issues(IPTC_ISSUES_PATH, gdown_id=IPTC_GDOWN_ID)
        print(f"Loaded {len(df_issues)} IPTC issues.")
        
        df_headlines = load_news_data(dataset_name=SELECTED_DATASET)
        print(f"Loaded {len(df_headlines)} headlines.")
        
        # Sampling for verification if SAMPLE_SIZE is set
        if SAMPLE_SIZE and len(df_headlines) > SAMPLE_SIZE:
            print(f"Sampling {SAMPLE_SIZE} headlines for processing...")
            df_headlines = df_headlines.head(SAMPLE_SIZE).copy()
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Sentiment Analysis
    print("Initializing Sentiment Analyzer...")
    analyzer = VaderSentimentAnalyzer()
    
    print("Running sentiment analysis...")
    neg, neu, pos = [], [], []
    for txt in tqdm(df_headlines['Headline']):
        scores = analyzer.analyze(txt)
        neg.append(scores['neg'])
        neu.append(scores['neu'])
        pos.append(scores['pos'])
        
    df_headlines['negative'] = neg
    df_headlines['neutral'] = neu
    df_headlines['positive'] = pos
    
    # User Filter: Keep only negative headlines
    print("Filtering for negative headlines (negative > positive)...")
    df_headlines = df_headlines[df_headlines['negative'] > df_headlines['positive']].copy()
    print(f"Headlines after filtering: {len(df_headlines)}")
    
    if df_headlines.empty:
        print("No headlines remaining after filtering. Exiting.")
        return
    
    # 3. Model Selection & Semantic Search
    from .config import CANDIDATE_MODELS
    
    best_score = -1.0
    best_model_name = None
    best_df_enriched = None
    best_G = None
    
    print(f"Starting Model Selection. Candidates: {CANDIDATE_MODELS}")
    
    # Pre-compute IPTC texts as they are constant
    issue_texts = (df_issues['issue_name'] + ' ' + df_issues['issue_description']).tolist()
    
    for model_name in CANDIDATE_MODELS:
        print(f"\nEvaluating model: {model_name}...")
        try:
            print("Loading Sentence Transformer...")
            model = SentenceTransformer(model_name)
            
            print("Generating embeddings for IPTC issues...")
            issue_embeddings = generate_embeddings(issue_texts, model)
            
            print("Building FAISS index...")
            # Normalize for Cosine Similarity
            issue_matrix_norm = l2_normalize(issue_embeddings)
            index = create_faiss_index(issue_matrix_norm)
            
            print("Searching top issues for headlines...")
            df_enriched = search_top_issues(
                df_headlines, 
                model, 
                index, 
                df_issues, 
                k=TOP_K_ISSUES
            )
            
            # Evaluation (Silhouette Score)
            print("Calculating Silhouette Score...")
            headline_texts = df_enriched['Headline'].astype(str).tolist()
            headline_embeddings = generate_embeddings(headline_texts, model)
            labels = df_enriched['issue_top1'].tolist()
            
            score = calculate_silhouette_score(headline_embeddings, labels)
            print(f"Model: {model_name} | Silhouette Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_df_enriched = df_enriched.copy()
                
                # Build graph for the best model so far (or just wait for end)
                # We can build it at the end to save time if graph building is slow, 
                # but it's fast compared to embeddings.
                
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            continue

    print(f"\n--- Model Selection Complete ---")
    print(f"Best Model: {best_model_name} | Best Score: {best_score:.4f}")
    
    if best_df_enriched is None:
        print("No model successfully evaluated. Exiting.")
        return

    # 4. Graph Construction (using best result)
    print("Building Issue-Event Graph using best model results...")
    G = build_issue_event_graph(best_df_enriched, threshold=SIMILARITY_THRESHOLD)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Visualize Graph
    visualize_graph(G, output_path=GRAPH_IMAGE_PATH)
    
    # 5. Save Results
    print("Saving results...")
    best_df_enriched.to_csv(HEADLINES_WITH_SENTIMENT_PATH, index=False)
    
    # Save Edges
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({'source': u, 'target': v, 'weight': data.get('weight', 0)})
    pd.DataFrame(edges_data).to_csv(EDGES_OUTPUT_PATH, index=False)
    
    # Save Nodes
    nodes_data = []
    for n, data in G.nodes(data=True):
        nodes_data.append({'id': n, 'label': data.get('txt', ''), 'type': data.get('type', '')})
    pd.DataFrame(nodes_data).to_csv(NODES_OUTPUT_PATH, index=False)
    
    print(f"Pipeline completed successfully. Outputs in {HEADLINES_WITH_SENTIMENT_PATH.parent}")

if __name__ == "__main__":
    run_pipeline()
