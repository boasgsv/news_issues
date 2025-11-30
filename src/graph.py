import networkx as nx
import pandas as pd
from tqdm import tqdm

def build_issue_event_graph(df: pd.DataFrame, threshold: float = 0.55) -> nx.Graph:
    """
    Builds a graph where nodes are Events (Headlines) and Issues, and edges represent similarity.
    """
    G = nx.Graph()
    node_id_map = {}
    
    def get_id(name):
        if name not in node_id_map:
            node_id_map[name] = len(node_id_map) + 1
        return node_id_map[name]

    print("Building graph...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        event_node = row['Headline']
        # Skip empty headlines
        if not isinstance(event_node, str) or not event_node.strip():
            continue
            
        event_id = get_id(event_node)
        
        # Add event node
        if event_id not in G:
            G.add_node(event_id, txt=event_node, type="event")
            
        # Check top 3 issues
        for i in range(1, 4):
            issue_name = row.get(f'issue_top{i}')
            sim = row.get(f'sim_top{i}')
            
            if issue_name and sim and sim > threshold:
                issue_id = get_id(issue_name)
                if issue_id not in G:
                    G.add_node(issue_id, txt=issue_name, type="issue")
                
                G.add_edge(issue_id, event_id, weight=sim)
                
    return G

def visualize_graph(G: nx.Graph, output_path: str = None):
    """
    Visualizes the graph using Plotly and saves to HTML if output_path is provided.
    """
    import plotly.graph_objects as go
    
    print("Generating graph visualization...")
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        node_text.append(f"{node_data.get('txt', '')} ({node_data.get('type', '')})")
        
        if node_data.get('type') == 'issue':
            node_color.append('red')
        else:
            node_color.append('blue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2),
        text=node_text)

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Issue-Event Network',
                title_font=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
                
    if output_path:
        fig.write_html(str(output_path))
        print(f"Graph visualization saved to {output_path}")
    
    return fig
