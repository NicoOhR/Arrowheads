import ast
import math

import plotly.graph_objects as go
import plotly.express as px


def main():
    with open("output.txt", "r") as f:
        s = f.read()
    data = ast.literal_eval(s)

    if not data:
        raise ValueError(
            "No curves found in output.txt after popping the first element."
        )

    kept_indices = [i for i in range(len(data)) if i % 4 == 0]

    xs = [pt[0] for pt in data[0]]
    ys_ref = [math.sin(x) for x in xs]

    def padded_range(vals, pad_frac=0.05):
        vmin = min(vals)
        vmax = max(vals)
        if vmin == vmax:
            eps = 1.0 if vmin == 0 else abs(vmin) * 0.1
            return [vmin - eps, vmax + eps]
        span = vmax - vmin
        pad = span * pad_frac
        return [vmin - pad, vmax + pad]

    ys0 = [pt[1] for pt in data[0]]

    x_range = padded_range(xs)
    y_range0 = padded_range(ys0 + ys_ref)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys0,
                mode="lines",
                name="iter 0",
            ),
            go.Scatter(
                x=xs,
                y=ys_ref,
                mode="lines",
                name="sin(x)",
                line=dict(dash="dash"),
            ),
        ],
        layout=go.Layout(
            xaxis=dict(title="x", range=x_range),
            yaxis=dict(title="y_hat", range=y_range0),
            title="Network outputs over training iterations",
            template="plotly_dark",
        ),
    )

    frames = []
    for i in kept_indices:
        curve = data[i]
        ys = [pt[1] for pt in curve]
        y_range = padded_range(ys + ys_ref)

        frames.append(
            go.Frame(
                name=f"iter{i}",
                data=[
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=f"iter {i}",
                    )
                ],
                layout=dict(
                    xaxis=dict(range=x_range),
                    yaxis=dict(range=y_range),
                ),
            )
        )

    fig.frames = frames

    slider_steps = []
    for i in kept_indices:
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [f"iter{i}"],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                label=str(i),
            )
        )

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Iteration: "},
                pad={"t": 30},
                steps=slider_steps,
            )
        ]
    )
    fig.show()
    html_snippit = fig.to_html(full_html=False, auto_play=False, include_plotlyjs=False)
    with open("plot_snippit.html", "w") as f:
        f.write(html_snippit)


if __name__ == "__main__":
    main()
