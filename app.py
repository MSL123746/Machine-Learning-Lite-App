# --- Modern ML Playground: Minimal Working Version ---
import streamlit as st
import numpy as np
# type: ignore
import plotly.graph_objects as go

# --- Custom CSS for modern look and layout ---
# ...existing code...

st.set_page_config(layout="wide")

st.title("AI LEARNING CONTROLS")


# --- Top bar (practice rounds, filename) ---
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5em;'>
  <div style='font-size:1.1em;color:#64748b;'>PRACTICE ROUNDS: <b>001,167</b></div>
  <div style='font-size:1.1em;color:#64748b;'>YourUploadedDataset.csv</div>
</div>
""", unsafe_allow_html=True)




# --- Move network diagram to top center ---
st.markdown("""
<div style='width:100%;display:flex;justify-content:center;margin-bottom:0.5em;'>
  <div style='max-width:700px;width:100%;'>
""", unsafe_allow_html=True)
import plotly.graph_objects as go
import time
layers = [2, 4, 4, 1]
node_x = [0.1, 0.4, 0.7, 0.95]
# Vertically center output neuron between input neurons
input_y = np.linspace(0.2, 0.8, layers[0])
output_y = [(input_y[0] + input_y[-1]) / 2]
node_y = [
    input_y,
    np.linspace(0.2, 0.8, layers[1]),
    np.linspace(0.2, 0.8, layers[2]),
    output_y
]
node_colors = ['#ffbb66', '#a3e17e', '#a3e17e', '#8fd3f4']
edge_colors = ['#f59e42', '#22c55e', '#60a5fa', '#222']
net_placeholder = st.empty()
if st.session_state.get('animate_network', False):
    steps = 20
    for anim_i in range(steps):
        edge_traces = []
        arrow_shapes = []
        for li in range(len(layers)-1):
            color = edge_colors[li] if li < len(edge_colors) else '#888'
            for j, y1 in enumerate(node_y[li]):
                for k, y2 in enumerate(node_y[li+1]):
                    width = 2 + 2 * abs(np.sin(anim_i/steps*2*np.pi + j + k))
                    edge_traces.append(go.Scatter(
                        x=[node_x[li], node_x[li+1]],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(width=width, color=color),
                        hoverinfo='none', showlegend=False))
                    arrow_shapes.append(dict(
                        type="path",
                        path=f"M {node_x[li+1]-.01} {y2-.01} L {node_x[li+1]+.01} {y2} L {node_x[li+1]-.01} {y2+.01} Z",
                        fillcolor=color, line_color=color))
        node_traces = []
        for li, (xs, ys, c) in enumerate(zip(node_x, node_y, node_colors)):
            node_traces.append(go.Scatter(
                x=[xs]*len(ys),
                y=ys,
                mode='markers',
                marker=dict(size=36 if li==0 or li==len(layers)-1 else 32, color=c, line=dict(width=2, color='#888')),
                hoverinfo='none', showlegend=False))
        # (Network diagram is now only rendered at the top center, not here)

    # Buttons moved to top of page
    st.markdown("""
<div style='margin-bottom:1em;'>
    <table style='width:100%;font-size:0.98em;'>
        <tr><th>Feature1</th><th>Feature2</th><th>Label</th></tr>
        <tr><td>1</td><td>12.09</td><td>0</td></tr>
        <tr><td>2</td><td>9.08</td><td>1</td></tr>
    </table>
</div>
    """, unsafe_allow_html=True)

    # --- Center: Animated Network ---
    # Use a dedicated flag to trigger animation before training
    import plotly.graph_objects as go
    import time
    layers = [2, 4, 4, 1]
    node_x = [0.1, 0.4, 0.7, 0.95]  # Output node aligned to right
    # Vertically center output neuron between input neurons
    input_y = np.linspace(0.2, 0.8, layers[0])
    output_y = [(input_y[0] + input_y[-1]) / 2]
    node_y = [
        input_y,
        np.linspace(0.2, 0.8, layers[1]),
        np.linspace(0.2, 0.8, layers[2]),
        output_y
    ]
    node_colors = ['#ffbb66', '#a3e17e', '#a3e17e', '#8fd3f4']
    edge_colors = ['#f59e42', '#22c55e', '#60a5fa', '#222']
    # Remove vertical spacer so network aligns with right charts
    net_placeholder = st.empty()
    if st.session_state.get('animate_network', False):
        steps = 20
        for anim_i in range(steps):
            edge_traces = []
            arrow_shapes = []
            for li in range(len(layers)-1):
                color = edge_colors[li] if li < len(edge_colors) else '#888'
                for j, y1 in enumerate(node_y[li]):
                    for k, y2 in enumerate(node_y[li+1]):
                        width = 2 + 2 * abs(np.sin(anim_i/steps*2*np.pi + j + k))
                        edge_traces.append(go.Scatter(
                            x=[node_x[li], node_x[li+1]],
                            y=[y1, y2],
                            mode='lines',
                            line=dict(width=width, color=color),
                            hoverinfo='none', showlegend=False))
                        arrow_shapes.append(dict(
                            type="path",
                            path=f"M {node_x[li+1]-.01} {y2-.01} L {node_x[li+1]+.01} {y2} L {node_x[li+1]-.01} {y2+.01} Z",
                            fillcolor=color, line_color=color))
            node_traces = []
            for li, (xs, ys, c) in enumerate(zip(node_x, node_y, node_colors)):
                node_traces.append(go.Scatter(
                    x=[xs]*len(ys),
                    y=ys,
                    mode='markers',
                    marker=dict(size=36 if li==0 or li==len(layers)-1 else 32, color=c, line=dict(width=2, color='#888')),
                    hoverinfo='none', showlegend=False))
            fig = go.Figure(edge_traces + node_traces)
            for shape in arrow_shapes:
                fig.add_shape(**shape)
            fig.add_annotation(x=0.05, y=0.5, text="Forward Pass", showarrow=False, font=dict(size=20, color="#f59e42"), xanchor="center", yanchor="middle")
            fig.add_annotation(x=0.95, y=output_y[0], text="Output D", showarrow=False, font=dict(size=20, color="#60a5fa"), xanchor="center", yanchor="bottom")
            fig.add_annotation(x=0.5, y=0.95, text="Neural Networks", showarrow=False, font=dict(size=22, color="#444"), bgcolor="#ccc", bordercolor="#888", borderwidth=1, borderpad=4)
            fig.add_annotation(x=0.5, y=0.08, text="Multiple Hidden\nLayers", showarrow=False, font=dict(size=18, color="#3b7a2a"), xanchor="center", yanchor="middle")
            fig.update_layout(
                width=500, height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,1]),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,1]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)')
            net_placeholder.plotly_chart(fig, use_container_width=False)
            time.sleep(0.05)
        # After animation, trigger training
        st.session_state['animate_network'] = False
        st.session_state['start_learning'] = True
        st.rerun()
    else:
        # Show static reference-style network
        edge_traces = []
        arrow_shapes = []
        for li in range(len(layers)-1):
            color = edge_colors[li] if li < len(edge_colors) else '#888'
            for j, y1 in enumerate(node_y[li]):
                for k, y2 in enumerate(node_y[li+1]):
                    edge_traces.append(go.Scatter(
                        x=[node_x[li], node_x[li+1]],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(width=2, color=color),
                        hoverinfo='none', showlegend=False))
                    arrow_shapes.append(dict(
                        type="path",
                        path=f"M {node_x[li+1]-.01} {y2-.01} L {node_x[li+1]+.01} {y2} L {node_x[li+1]-.01} {y2+.01} Z",
                        fillcolor=color, line_color=color))
        node_traces = []
        for li, (xs, ys, c) in enumerate(zip(node_x, node_y, node_colors)):
            node_traces.append(go.Scatter(
                x=[xs]*len(ys),
                y=ys,
                mode='markers',
                marker=dict(size=36 if li==0 or li==len(layers)-1 else 32, color=c, line=dict(width=2, color='#888')),
                hoverinfo='none', showlegend=False))
        fig = go.Figure(edge_traces + node_traces)
        for shape in arrow_shapes:
            fig.add_shape(**shape)
        fig.add_annotation(x=0.05, y=0.5, text="Forward Pass", showarrow=False, font=dict(size=20, color="#f59e42"), xanchor="center", yanchor="middle")
        fig.add_annotation(x=0.95, y=output_y[0], text="Output D", showarrow=False, font=dict(size=20, color="#60a5fa"), xanchor="center", yanchor="bottom")
        fig.add_annotation(x=0.5, y=0.95, text="Neural Networks", showarrow=False, font=dict(size=22, color="#444"), bgcolor="#ccc", bordercolor="#888", borderwidth=1, borderpad=4)
        fig.add_annotation(x=0.5, y=0.08, text="Multiple Hidden\nLayers", showarrow=False, font=dict(size=18, color="#3b7a2a"), xanchor="center", yanchor="middle")
        fig.update_layout(
            width=500, height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,1]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)')
        net_placeholder.plotly_chart(fig, use_container_width=False)
    st.markdown("<div style='text-align:center;color:#64748b;margin-top:1em;'>Watch the AI 'brain' make connections!</div>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1.3, 1.7, 0.8], gap="large")

# --- Buttons for training and reset ---
with col_left:
    train_clicked = st.button("START LEARNING!", type="primary")
    reset_clicked = st.button("RESET")

# --- Right: Results ---
with col_right:
    # Always show dataset name and charts at the very top right, no gap
    if st.session_state.get('uploaded_filename'):
        st.markdown(f"<div style='text-align:right; color:#4682b4; font-size:16px; font-weight:600; margin-bottom:0px;'>{st.session_state['uploaded_filename']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Charts always at the top, no vertical gap
    chart_height = 320
    # Remove extra margin and vertical spacing so charts are at the top
    st.markdown("<div style='margin-top:0; padding-top:0; height:0;'></div>", unsafe_allow_html=True)
    results_ready = all(k in st.session_state for k in ['last_acc', 'last_loss', 'last_decision_fig'])
    if results_ready:
        import plotly.graph_objects as go
        st.markdown("<div style='text-align:center;font-size:1.2em;color:#64748b;margin-top:0.5em;'>ACCURACY</div>", unsafe_allow_html=True)
        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=st.session_state['last_acc'], title={'text': ""},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#22c55e"},
                   'steps': [
                       {'range': [0, 50], 'color': "#e0e7ef"},
                       {'range': [50, 80], 'color': "#fde68a"}],
                   'threshold': {'line': {'color': "#f59e42", 'width': 4}, 'thickness': 0.75, 'value': 90}})), use_container_width=True)
        st.markdown("<div style='text-align:center;font-size:1.1em;color:#64748b;margin-top:0.5em;'>LEARNING JOURNEY</div>", unsafe_allow_html=True)
        st.line_chart(st.session_state['last_loss'])
        st.pyplot(st.session_state['last_decision_fig'])
        st.markdown("""
            <div style='margin-top:0.5em;display:flex;align-items:center;gap:0.7em;background:#f7f9fa;border-radius:10px;padding:0.7em 1em;'>
              <span style='font-size:1.5em;color:#22c55e;'>★</span>
              <span style='font-size:1.1em;'><b>Great job! The AI is learning!</b></span>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Remove info box and extra vertical space so charts are always at the top
        st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)


## Removed duplicate single-column controls/data panel and duplicate top bar

if train_clicked:
    # Only trigger animation if not already running
    if not st.session_state.get('animate_network', False) and not st.session_state.get('start_learning', False):
        st.session_state['animate_network'] = True
        st.rerun()
    # After animation, do training and results (and nowhere else)
    elif st.session_state.get('start_learning', False):
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, log_loss

        # Generate dummy data
        X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=np.random.randint(0, 10000))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 10000))

        learning_rate_init = 0.001 + (st.session_state.get("training_speed", 60) / 100) * 0.199
        max_iter = int(10 + (st.session_state.get("focus_level", 80) / 100) * 190)

        # Create placeholders for live chart updates
        right_col_placeholder = st.empty()
        gauge_placeholder = st.empty()
        journey_placeholder = st.empty()
        decision_placeholder = st.empty()

        losses = []
        for i in range(1, max_iter+1):
            clf = MLPClassifier(hidden_layer_sizes=(4,4), learning_rate_init=learning_rate_init, max_iter=i, warm_start=True, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            loss = log_loss(y_test, y_prob)
            losses.append(loss)
            acc = accuracy_score(y_test, y_pred) * 100
            # Decision boundary plot
            fig2, ax = plt.subplots()
            h = .02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=plt.cm.PuBu, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.PuBu)
            ax.set_xticks([])
            ax.set_yticks([])
            # Update right pane charts live
            with right_col_placeholder.container():
                gauge_placeholder.plotly_chart(
                    go.Figure(go.Indicator(mode="gauge+number", value=acc, title={'text': ""},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#22c55e"},
                               'steps': [
                                   {'range': [0, 50], 'color': "#e0e7ef"},
                                   {'range': [50, 80], 'color': "#fde68a"}],
                               'threshold': {'line': {'color': "#f59e42", 'width': 4}, 'thickness': 0.75, 'value': 90}})),
                    use_container_width=True, key=f"gauge_{i}")
                journey_placeholder.line_chart(losses, use_container_width=True)
                decision_placeholder.pyplot(fig2)
            import time
            time.sleep(0.05)
        # Save final results to session state for right pane
        st.session_state['last_acc'] = acc
        st.session_state['last_loss'] = losses
        st.session_state['last_decision_fig'] = fig2
        st.session_state['start_learning'] = False
        st.rerun()


if reset_clicked:
    st.session_state.start_learning = False
    for k in ['last_acc', 'last_loss', 'last_decision_fig', 'last_net_fig']:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:#f7f9fa;border-radius:12px;padding:1em 1.5em;margin-top:1em;'>
    <b>HELPFUL TIPS</b>
    <div style='display:flex;align-items:center;gap:1em;margin-top:0.7em;'>
        <button style='background:#fff;border:2px solid #e0e7ef;border-radius:8px;padding:0.3em 0.8em;font-size:1.2em;'>?</button>
        <button style='background:#fff;border:2px solid #e0e7ef;border-radius:8px;padding:0.3em 0.8em;font-size:1.2em;'>💡 What are "weights"?</button>
        <button style='background:#fff;border:2px solid #e0e7ef;border-radius:8px;padding:0.3em 0.8em;font-size:1.2em;'>😕 loss?</button>
    </div>
</div>
    """, unsafe_allow_html=True)



