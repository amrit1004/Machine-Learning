import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load dataset
df = pd.read_csv(r"C:\Users\Amrit Singhal\Desktop\Machine Learning\each_ball_records.csv")
df.columns = df.columns.str.strip()  # Remove spaces if any

# Create figure with subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Score Progression", "Runs Heatmap"),
                    column_widths=[0.6, 0.4])

# 📈 Interactive Line Plot
fig.add_trace(go.Scatter(x=df["ballnumber"], y=df["score"], mode="lines+markers", 
                         marker=dict(color="blue"), name="Score Progression"),
              row=1, col=1)

# 🔥 Heatmap (Over vs Ballnumber)
heatmap_data = df.pivot_table(values="score", index="over", columns="ballnumber", aggfunc="sum")

heatmap_trace = go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                           colorscale="viridis", name="Runs Heatmap")  # Fixed colorscale

fig.add_trace(heatmap_trace, row=1, col=2)

# Customize layout
fig.update_layout(title_text="IPL 2023 Ball-by-Ball Analysis", height=600, width=1200,
                  xaxis_title="Ball Number", yaxis_title="Score",
                  template="plotly_dark")

# Show interactive plot
fig.show()
