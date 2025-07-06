"""
e7plot.py is imported only when needed to avoid introducing a global dependency on Plotly across the entire package.
"""

from __future__ import annotations

import numpy as np
from e7awgsw import WaveSequence
from plotly import graph_objects as go


def plot_wseq(wseq: WaveSequence, device_index_at_user_zero: int = 0) -> go.Figure:
    """
    Create a Plotly figure for the WSEQ plot.

    Returns:
        go.Figure: A Plotly figure object containing the WSEQ plot.
    """
    fig = go.Figure()
    fig.update_layout(
        title="WSEQ Plot",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        template="plotly_white",
    )
    device_index_at_chunk_start = -device_index_at_user_zero
    for k, chunk in enumerate(wseq.chunk_list):
        wave_data = chunk.wave_data
        indices = device_index_at_chunk_start + np.arange(wave_data.num_samples)
        values = np.array([r + 1j * i for r, i in wave_data.samples])
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=np.real(values),
                mode="lines",
                name=f"real(chunk({k}))",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=np.imag(values),
                mode="lines",
                name=f"imag(chunk({k}))",
            )
        )
        device_index_at_chunk_start += wave_data.num_samples + chunk.num_blank_samples
    return fig
