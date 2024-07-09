import pandas as pd
from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import base64
import os

def create_dash_app(question_mapping):
    app = Dash(__name__)
    
    tabs = []
    
    for question, col in question_mapping.items():
        try:
            # Load pre-computed results
            sentiment_results = pd.read_csv(f"sentiment_analysis_results_{col}.csv")
            keyword_summary = pd.read_csv(f"keyword_analysis_results_{col}.csv")
            topic_info = pd.read_csv(f"topic_analysis_results_{col}.csv")
            
            with open(f"wordcloud_{col}.txt", "r") as f:
                wordcloud_img = f.read()
            
            with open(f"topic_summary_{col}.txt", "r") as f:
                topic_summary_text = f.read()
            
            # Create visualizations
            sentiment_fig = px.pie(sentiment_results, names='label', title=f'Sentiment Distribution for {question}')
            keyword_fig = px.bar(keyword_summary.head(20), x='Keyword', y='Count', title=f'Top 20 Keywords for {question}')
            
            topic_bar_fig = go.Figure()
            for _, row in topic_info.iterrows():
                if row['Topic'] != -1:
                    topic_bar_fig.add_trace(go.Bar(
                        x=[f"Topic {row['Topic']}"],
                        y=[row['Count']],
                        name=f"Topic {row['Topic']}",
                        text=[f"Count: {row['Count']}"],
                        hoverinfo='text'
                    ))
            topic_bar_fig.update_layout(
                title=f"Topic Analysis Results for {question}",
                xaxis_title="Topics",
                yaxis_title="Count of Responses",
                hovermode="closest"
            )
            
            tab_content = html.Div([
                html.H2(question),
                dcc.Graph(figure=sentiment_fig),
                dcc.Graph(figure=keyword_fig),
                html.Img(src=f"data:image/png;base64,{wordcloud_img}", style={'width': '100%', 'height': 'auto'}),
                dcc.Graph(figure=topic_bar_fig),
                html.Div([
                    html.H4("Topic Summary"),
                    html.Pre(topic_summary_text)
                ])
            ])
        
        except (FileNotFoundError, pd.errors.EmptyDataError):
            tab_content = html.Div([
                html.H2(question),
                html.P("No data available for this question.")
            ])
        
        tabs.append(dcc.Tab(label=question, children=tab_content))
    
    app.layout = html.Div([
        html.H1("Comprehensive Analysis of Open-Ended Questions"),
        dcc.Tabs(tabs)
    ])
    
    return app

if __name__ == "__main__":
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3",
        "Please select the reasons for not visiting the campus food pantry.": "Foodpantry_reasons",
        "What are your thoughts about food availability on your campus?": "Foodavail",
        "Please share why you feel unsafe?": "Unsafe_why",
        "Please explain why it is difficult to find housing either on-campus or off-campus?": "Housingdiff_why"
    }
    
    app = create_dash_app(question_mapping)
    app.run_server(debug=True, host='0.0.0.0', port=8080)