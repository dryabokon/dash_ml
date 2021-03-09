import numpy
import pandas as pd
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
# ----------------------------------------------------------------------------------------------------------------------
import utils_dash_draw
import utils_dash_business_logic
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './assets/'
# ----------------------------------------------------------------------------------------------------------------------
df, idx_target = pd.read_csv(folder_in + 'traffic_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'electricity_hourly_small.txt', delimiter=','), 1
#df, idx_target = pd.read_csv(folder_in + 'monthly_passengers.txt', delimiter='\t'), 1
# ----------------------------------------------------------------------------------------------------------------------
n_start = df.shape[0]//3
n_steps_ahead = 20
# ----------------------------------------------------------------------------------------------------------------------
dark_mode = True
app = dash.Dash(__name__,external_stylesheets=([dbc.themes.BOOTSTRAP] if dark_mode else [dbc.themes.BOOTSTRAP]))
# ----------------------------------------------------------------------------------------------------------------------
U = utils_dash_draw.DashDraw(folder_out,dark_mode)
B = utils_dash_business_logic.Business_logic(app,folder_out,dark_mode)
# ----------------------------------------------------------------------------------------------------------------------
div_main = html.Div(
    [
        dbc.Row([dbc.Col([U.draw_image_URL_unframed(app.get_asset_url(U.URL_image_header),U.clr_header)], width=12)],align='center',no_gutters=True),
        dbc.Row([dbc.Col([U.draw_text_framed('Time Series prediction', color=U.clr_banner)],width=12)], align='center'),
        html.Br(),

        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_main_TS",U.clr_chart)], width=12)], align='center'),

        html.Br(),
        dbc.Row([dbc.Col([U.draw_text_framed('Stats', color=U.clr_banner)],width=12)], align='center'),
        html.Br(),
        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_ACC_01", U.clr_chart)], width=6),
                 dbc.Col([U.draw_chart_unframed("chart_ACC_02", U.clr_chart)], width=6)], align='center'),
        html.Br(),

        dcc.Interval(id='graph-update',interval=1000,n_intervals=0),
    ]
)
# ----------------------------------------------------------------------------------------------------------------------
@app.callback([Output('chart_main_TS', 'children'),
               Output('chart_ACC_01', 'children'),
               Output('chart_ACC_02', 'children')],
              [Input('graph-update', 'n_intervals')])
def update_graph(n):

    plots_acc = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))] * 2))
    plot_TS = [html.Img(src=app.get_asset_url(U.URL_empty))]

    if n_start+n<df.shape[0]:
        limit = n_start+n
        plot_TS,plots_acc = B.get_TS_prediction_plot_html(df.iloc[:limit],idx_target,n_start+5, n_steps_ahead,plots_acc)


    return plot_TS,\
           plots_acc[0],plots_acc[1]
# # ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    app.layout = html.Div([dbc.Card(dbc.CardBody(div_main), color=U.clr_pad)])
    app.run_server(debug=False)