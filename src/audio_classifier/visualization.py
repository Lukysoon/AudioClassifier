"""
Visualization module for 3D plots and heatmaps using Plotly.
"""

import base64
import hashlib
import webbrowser
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from .config import VisualizationConfig


def _encode_audio_base64(file_path: str) -> str:
    """
    Encode audio file to base64 data URI.

    Args:
        file_path: Path to the audio file.

    Returns:
        Base64 data URI string.
    """
    with open(file_path, "rb") as f:
        audio_data = f.read()

    ext = Path(file_path).suffix.lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac"
    }
    mime_type = mime_types.get(ext, "audio/mpeg")

    encoded = base64.b64encode(audio_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _path_to_id(file_path: str) -> str:
    """
    Convert file path to safe HTML ID.

    Args:
        file_path: File path to convert.

    Returns:
        12-character hex string.
    """
    return hashlib.md5(file_path.encode()).hexdigest()[:12]


class Visualizer:
    """Creates interactive 3D plots and heatmaps."""

    def __init__(self, config: VisualizationConfig | None = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_3d_scatter(
        self,
        df: pd.DataFrame,
        title: str = "Audio Embeddings"
    ) -> go.Figure:
        """
        Create interactive 3D scatter plot.

        Args:
            df: DataFrame with columns: file, label, x, y, z
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        # Get filename from full path for hover
        df = df.copy()
        df["filename"] = df["file"].apply(lambda x: Path(x).name)

        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="label",
            hover_name="filename",
            custom_data=["file"],
            title=title
        )

        fig.update_traces(
            marker=dict(
                size=self.config.marker_size,
                opacity=self.config.marker_opacity
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3"
            ),
            legend_title="Category",
            height=800
        )

        return fig

    def plot_distance_heatmap(
        self,
        distance_matrix: pd.DataFrame,
        title: str = "Category Distances"
    ) -> go.Figure:
        """
        Create distance matrix heatmap.

        Args:
            distance_matrix: Square DataFrame with category distances.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        fig = px.imshow(
            distance_matrix,
            labels=dict(x="Category", y="Category", color="Distance"),
            title=title,
            color_continuous_scale=self.config.color_scale,
            aspect="auto"
        )

        # Add text annotations
        fig.update_traces(
            text=distance_matrix.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=12)
        )

        fig.update_layout(height=600)

        return fig

    def save_figure(self, fig: go.Figure, filename: str) -> Path:
        """
        Save figure to HTML file.

        Args:
            fig: Plotly Figure object.
            filename: Output filename (without directory).

        Returns:
            Full path to saved file.
        """
        output_path = self.config.output_dir / filename
        fig.write_html(str(output_path))
        print(f"Saved: {output_path}")
        return output_path

    def open_in_browser(self, filepath: Path) -> None:
        """
        Open HTML file in default web browser.

        Args:
            filepath: Path to HTML file.
        """
        webbrowser.open(f"file://{filepath.absolute()}")

    def _generate_audio_player_html(self, audio_data: Dict[str, str]) -> str:
        """
        Generate HTML for audio elements and player UI.

        Args:
            audio_data: Dictionary mapping file paths to base64 data URIs.

        Returns:
            HTML string with audio elements and player UI.
        """
        audio_elements = []
        for file_path, data_uri in audio_data.items():
            safe_id = _path_to_id(file_path)
            audio_elements.append(
                f'<audio id="audio-{safe_id}" src="{data_uri}" preload="none"></audio>'
            )

        player_ui = '''
        <div id="audio-player-container" style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            z-index: 10000;
            display: none;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            min-width: 250px;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <button id="audio-stop-btn" style="
                    background: #e74c3c;
                    border: none;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    transition: background 0.2s;
                " onmouseover="this.style.background='#c0392b'" onmouseout="this.style.background='#e74c3c'">Stop</button>
                <div style="flex: 1;">
                    <div id="audio-filename" style="font-weight: 600; margin-bottom: 4px; font-size: 14px;"></div>
                    <div id="audio-status" style="font-size: 12px; opacity: 0.8; display: flex; align-items: center;">
                        <span id="audio-indicator" style="
                            display: inline-block;
                            width: 8px;
                            height: 8px;
                            background: #2ecc71;
                            border-radius: 50%;
                            margin-right: 8px;
                            animation: pulse 1s infinite;
                        "></span>
                        <span id="audio-time">Playing...</span>
                    </div>
                </div>
            </div>
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(0.9); }
            }
        </style>
        '''

        return "\n".join(audio_elements) + player_ui

    def _generate_click_handler_js(self, audio_data: Dict[str, str]) -> str:
        """
        Generate JavaScript for handling click events and audio playback.

        Args:
            audio_data: Dictionary mapping file paths to base64 data URIs.

        Returns:
            JavaScript code string.
        """
        audio_id_map = {path: _path_to_id(path) for path in audio_data.keys()}
        audio_id_json = str(audio_id_map).replace("'", '"')

        return f'''
        <script>
        (function() {{
            var currentAudio = null;
            var audioIdMap = {audio_id_json};

            function stopAudio() {{
                if (currentAudio) {{
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                    currentAudio = null;
                }}
                document.getElementById('audio-player-container').style.display = 'none';
            }}

            function formatTime(seconds) {{
                var mins = Math.floor(seconds / 60);
                var secs = Math.floor(seconds % 60);
                return mins + ':' + (secs < 10 ? '0' : '') + secs;
            }}

            function playAudio(filePath, filename) {{
                stopAudio();

                var audioId = audioIdMap[filePath];
                if (!audioId) {{
                    console.warn('Audio not found for:', filePath);
                    return;
                }}

                var audioEl = document.getElementById('audio-' + audioId);
                if (!audioEl) {{
                    console.warn('Audio element not found:', audioId);
                    return;
                }}

                currentAudio = audioEl;

                var container = document.getElementById('audio-player-container');
                container.style.display = 'block';
                document.getElementById('audio-filename').textContent = filename;
                document.getElementById('audio-time').textContent = 'Loading...';

                audioEl.ontimeupdate = function() {{
                    var current = formatTime(audioEl.currentTime);
                    var duration = formatTime(audioEl.duration || 0);
                    document.getElementById('audio-time').textContent = current + ' / ' + duration;
                }};

                audioEl.onended = function() {{
                    stopAudio();
                }};

                audioEl.onerror = function() {{
                    document.getElementById('audio-time').textContent = 'Error loading audio';
                    document.getElementById('audio-indicator').style.background = '#e74c3c';
                }};

                audioEl.play().then(function() {{
                    document.getElementById('audio-indicator').style.background = '#2ecc71';
                }}).catch(function(error) {{
                    console.error('Playback failed:', error);
                    document.getElementById('audio-time').textContent = 'Playback failed';
                    document.getElementById('audio-indicator').style.background = '#e74c3c';
                }});
            }}

            document.getElementById('audio-stop-btn').onclick = stopAudio;

            function attachClickHandler() {{
                var plotDiv = document.querySelector('.plotly-graph-div');
                if (plotDiv && plotDiv.on) {{
                    plotDiv.on('plotly_click', function(data) {{
                        if (data.points && data.points.length > 0) {{
                            var point = data.points[0];
                            var filePath = point.customdata[0];
                            var filename = point.hovertext || filePath.split('/').pop();
                            playAudio(filePath, filename);
                        }}
                    }});
                    console.log('Audio click handler attached');
                }} else {{
                    setTimeout(attachClickHandler, 100);
                }}
            }}

            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', attachClickHandler);
            }} else {{
                setTimeout(attachClickHandler, 100);
            }}
        }})();
        </script>
        '''

    def save_figure_with_audio(
        self,
        fig: go.Figure,
        filename: str,
        audio_files: List[str]
    ) -> Path:
        """
        Save figure to HTML with embedded audio playback functionality.

        Args:
            fig: Plotly Figure object.
            filename: Output filename.
            audio_files: List of audio file paths to embed.

        Returns:
            Full path to saved file.
        """
        output_path = self.config.output_dir / filename

        print("Embedding audio files...")
        audio_data = {}
        for file_path in audio_files:
            if Path(file_path).exists():
                audio_data[file_path] = _encode_audio_base64(file_path)

        html_content = fig.to_html(full_html=True, include_plotlyjs=True)

        audio_player_html = self._generate_audio_player_html(audio_data)
        click_handler_js = self._generate_click_handler_js(audio_data)

        injection_point = html_content.rfind("</body>")
        html_content = (
            html_content[:injection_point] +
            audio_player_html +
            click_handler_js +
            html_content[injection_point:]
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Saved: {output_path}")
        return output_path

    def save_and_open(
        self,
        fig: go.Figure,
        filename: str,
        audio_files: Optional[List[str]] = None
    ) -> Path:
        """
        Save figure and optionally open in browser.

        Args:
            fig: Plotly Figure object.
            filename: Output filename.
            audio_files: Optional list of audio file paths for playback.

        Returns:
            Full path to saved file.
        """
        if audio_files:
            filepath = self.save_figure_with_audio(fig, filename, audio_files)
        else:
            filepath = self.save_figure(fig, filename)

        if self.config.auto_open:
            self.open_in_browser(filepath)

        return filepath
