from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import callbacks
from dash_extensions import Keyboard

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'figure',
    #'height': 500,
    #'width': 700,
    'scale': 6 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

options = ['Expected Value', '50% Range', '95% Range', '2.5th Percentile', '25th Percentile', 'Median', '75th Percentile', '97.5th Percentile']

app.layout = html.Div(
    [
        dbc.Row(
        [
            dbc.Col([
                dcc.Upload(
                    id='upload-data-mean',
                    children=html.Div([
                        'Drag and Drop or Select File for Mean (.npy)'
                    ]),
                    style={
                        # 'width': '100vh',
                        'height': '5vh',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '2%'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                )
            ], width=4),  # Full width for this column
            dbc.Col([
                dcc.Upload(
                    id='upload-data-cov',
                    children=html.Div([
                        'Drag and Drop or Select File for Covariance (.npy)'
                    ]),
                    style={
                        # 'width': '100vh',
                        'height': '5vh',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '2%'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                )], width=4),
            dbc.Col([
                dcc.Upload(
                    id='upload-data-config',
                    children=html.Div([
                        'Drag and Drop or Select File for Config (.json)'
                    ]),
                    style={
                        # 'width': '100vh',
                        'height': '5vh',
                        'lineHeight': '40px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '2%'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                )], width=4)
        ], className="g-0"),
        dbc.Row(
        [
            dbc.Col([
                dcc.Graph(id='fig-time-series', style={'height': '25vh'}),
            ], width=12),
            dcc.Tabs([
                dcc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='fig-fourier-transform', style={'height': '25vh'})
                        ], width=8),
                        dbc.Col([
                            html.H6("Settings"),
                            html.Div(["Threshold: "]),
                            dcc.Slider(min=0, max=1, marks=None, id='threshold-slider', tooltip={"placement": "bottom", "always_visible": True}),
                            daq.BooleanSwitch(
                                on=False,
                                id="switch-global-correlations",
                                label="Global Correlations",
                                style={'display':'inline-block', 'margin-right': '30px'}
                            ),
                            daq.BooleanSwitch(
                                on=False,
                                id="switch-distance-noise",
                                label="Show Distance to Noise",
                                style={'display': 'inline-block'}
                            ),
                            dcc.Dropdown(['Correlation', 'Slice Time', 'Slice Scale'], id='dropdown-selection', value='Correlation'),
                            dcc.Dropdown(options, id='dropdown-view', value='Expected Value')
                        ], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='fig-expectation-value', style={'height': '40vh'}),
                            Keyboard(
                                captureKeys=["a", "d"], id="keyboard"
                            )
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='fig-distribution-vis', style={'height': '40vh'}, config=config)
                        ], width=6)
                    ])
                ], label="Percentiles", style={'line-height': 30, "padding":0}, selected_style={'line-height': 30, "padding":0}, value='percentiles'),
                dcc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='fig-fourier-transform-sensitivity', style={'height': '25vh'})
                        ], width=8),
                        dbc.Col([
                            html.H6("Settings"),
                            dcc.Dropdown(
                                id='dropdown-distribution-selection',
                                options=[
                                    {'label': 'Normal distribution', 'value': 'normal'},
                                    {'label': 'Uniform distribution', 'value': 'uniform'}
                                ],
                                value='uniform'
                            ),
                            html.Label('Mean', id="label-slider-1"),
                            html.Div([dcc.Slider(
                                id='slider1',
                                min=0,
                                max=10,
                                value=1,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )], id='slider1-div'),
                            html.Div([dcc.Slider(
                                id='sliderVal',
                                min=0,
                                max=10,
                                value=1,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )], id='sliderVal-div'),
                            html.Label('Variance', id="label-slider-2"),
                            html.Div([dcc.Slider(
                                id='slider2',
                                min=0,
                                max=10,
                                value=1,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )], id='slider2-div'),
                            html.Label('Scaling', id="label-slider-3"),
                            html.Div([dcc.Slider(
                                id='slider3',
                                min=0,
                                max=10,
                                value=1,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )], id='slider3-div'),
                            daq.BooleanSwitch(
                                on=False,
                                id="switch-relative-difference",
                                label="Relative Difference"
                            ),
                        ], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='fig-expectation-value-sensitivity', style={'height': '40vh'})
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='fig-wavelet-variance-sensitivity', style={'height': '40vh'})
                        ], width=6)
                    ])
                ], label="Sensitivity", style={'line-height': 30, "padding":0}, selected_style={'line-height': 30, "padding":0}, value='sensitivity')
            ], style={"height":30}, id='tabs', value='percentiles')
        ]),
        # dbc.Row(
        # [
        #     dbc.Col([
        #         # dcc.Graph(id='fig-expectation-value', style={'height': '40vh'}),
        #         # html.Div([
        #         #     daq.BooleanSwitch(
        #         #         on=False,
        #         #         id="switch-global-correlations",
        #         #         label="Global Correlations"
        #         #     ),
        #         #     dcc.Slider(min=0, max=1, id='threshold-slider')
        #         # ], style={'height': '10vh'}),
        #         # dcc.Graph(id='fig-fourier-transform', style={'height': '20vh'}),
        #     ]),
        #     dbc.Col([
        #         dbc.Row([
        #             dbc.Col([
        #                 dcc.Graph(id='fig-kernel', style={'height': '18vh'}),
        #             ]),
        #             dbc.Col([
        #                 html.H2("Controls"),
        #                 dcc.Dropdown(
        #                     id='dropdown',
        #                     options=[
        #                         {'label': 'Option 1', 'value': 'opt1'},
        #                         {'label': 'Option 2', 'value': 'opt2'},
        #                         {'label': 'Option 3', 'value': 'opt3'}
        #                     ],
        #                     value='opt1'
        #                 ),
        #             ])
        #         ]),
        #         dcc.Graph(id='fig-percentile5', style={'height': '18vh'}),
        #         dcc.Graph(id='fig-percentile25', style={'height': '18vh'}),
        #         dcc.Graph(id='fig-percentile50', style={'height': '18vh'}),
        #         dcc.Graph(id='fig-percentile75', style={'height': '18vh'}),
        #         dcc.Graph(id='fig-percentile95', style={'height': '18vh'})
        #     ], width=4),  # Taking 4 units of width
        # ]
        # ),
        dcc.Store(id='store-data'),
        dcc.Store(id='store-perturbation'),
        dcc.Store(id='store-mask')
    ], style={'width': '100vw'}
)

if __name__ == '__main__':
    app.run_server(debug=False)