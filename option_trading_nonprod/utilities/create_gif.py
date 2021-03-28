# Creating a gif to be used in an email or whatever seems suited

# https://stackoverflow.com/questions/55460434/how-to-export-save-an-animated-bubble-chart-made-with-plotly

# example code from aboves link
import random
import plotly.graph_objects as go
import pandas as pd
import gif

# Pandas DataFrame with random data
df = pd.DataFrame({
	't': list(range(10)) * 10,
	'x': [random.randint(0, 100) for _ in range(100)],
	'y': [random.randint(0, 100) for _ in range(100)]
})

# Gif function definition
@gif.frame
def plot(i):
	d = df[df['t'] == i]
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=d["x"],
		y=d["y"],
		mode="markers"
	))
	fig.update_layout(width=500, height=300)
	return fig

# Construct list of frames
frames = []
for i in range(10):
	frame = plot(i)
	frames.append(frame)

# Save gif from frames with a specific duration for each frame in ms
gif.save(frames, 'example.gif', duration=100)