import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
# ----------------------------------------------------------------------------------------------------------------------
import utils_dash_draw
import utils_dash_business_logic
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './assets/'
# ----------------------------------------------------------------------------------------------------------------------
# TODO PR-curves, feature correlation, VIF,  cross validation scores,
# TODO target selection, feature filtering
# ----------------------------------------------------------------------------------------------------------------------
dark_mode = True
# ----------------------------------------------------------------------------------------------------------------------
app = dash.Dash(external_stylesheets=([dbc.themes.BOOTSTRAP] if dark_mode else [dbc.themes.BOOTSTRAP]))
# ----------------------------------------------------------------------------------------------------------------------
U = utils_dash_draw.DashDraw(folder_out,dark_mode)
B = utils_dash_business_logic.Business_logic(app,folder_out,dark_mode)
# ----------------------------------------------------------------------------------------------------------------------
div_main = html.Div(
    [
        dbc.Row([dbc.Col([U.draw_image_URL_unframed(app.get_asset_url(U.URL_image_header),U.clr_header)], width=12)],align='center',no_gutters=True),

        dbc.Row([dbc.Col([U.draw_picker('upload-data','Binary classification',color=U.clr_sub_header)], width=12)], align='center'),
        html.Br(),

        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_pairplot_01",U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_pairplot_02",U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_pairplot_03",U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_pairplot_04",U.clr_chart)], width=3),], align='center'),

        html.Br(),
        dbc.Row([dbc.Col([U.draw_text_framed('Principal components', color=U.clr_banner)],width=12)], align='center'),
        html.Br(),
        dbc.Row([dbc.Col([U.draw_text_unframed("SVD"   , U.clr_chart)], width=3),
                 dbc.Col([U.draw_text_unframed("PCA"   , U.clr_chart)], width=3),
                 dbc.Col([U.draw_text_unframed("tSNE"  , U.clr_chart)], width=3),
                 dbc.Col([U.draw_text_unframed("ISOMAP", U.clr_chart)], width=3),], align='center'),
        html.Br(),
        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_dim_SVD", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_dim_PCA", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_dim_tSNE", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_dim_ISOMAP", U.clr_chart)], width=3), ], align='center'),

        html.Br(),
        dbc.Row([dbc.Col([U.draw_text_framed('Feature importance', color=U.clr_banner)], width=12)], align='center'),
        html.Br(),

        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_FI_01", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_FI_02", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_FI_03", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_FI_04", U.clr_chart)], width=3),], align='center'),

        html.Br(),
        dbc.Row([dbc.Col([U.draw_text_framed('Prediction metrics', color=U.clr_banner)], width=12)], align='center'),
        html.Br(),
        dbc.Row([dbc.Col([U.draw_text_unframed("LM", U.clr_chart)], width=3),
                 dbc.Col([U.draw_text_unframed("SVM", U.clr_chart)], width=3),
                 dbc.Col([U.draw_text_unframed("RF", U.clr_chart)], width=3),
                 dbc.Col([U.draw_text_unframed("KNN", U.clr_chart)], width=3), ], align='center'),
        html.Br(),

        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_AUC_train_01", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_AUC_train_02", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_AUC_train_03", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_AUC_train_04", U.clr_chart)], width=3)], align='center'),
        html.Br(),
        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_AUC_test_01", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_AUC_test_02", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_AUC_test_03", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_AUC_test_04", U.clr_chart)], width=3)], align='center'),
        html.Br(),
        dbc.Row([dbc.Col([U.draw_chart_unframed("chart_dnst_01", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_dnst_02", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_dnst_03", U.clr_chart)], width=3),
                 dbc.Col([U.draw_chart_unframed("chart_dnst_04", U.clr_chart)], width=3)], align='center'),
        html.Br(),
    ])
# ----------------------------------------------------------------------------------------------------------------------
@app.callback([Output('chart_pairplot_01', 'children'),
               Output('chart_pairplot_02', 'children'),
               Output('chart_pairplot_03', 'children'),
               Output('chart_pairplot_04', 'children'),

               Output('chart_dim_SVD', 'children'),
               Output('chart_dim_PCA', 'children'),
               Output('chart_dim_tSNE', 'children'),
               Output('chart_dim_ISOMAP', 'children'),

               Output('chart_FI_01', 'children'),
               Output('chart_FI_02', 'children'),
               Output('chart_FI_03', 'children'),
               Output('chart_FI_04', 'children'),

               Output('chart_AUC_train_01', 'children'),
               Output('chart_AUC_train_02', 'children'),
               Output('chart_AUC_train_03', 'children'),
               Output('chart_AUC_train_04', 'children'),

               Output('chart_AUC_test_01', 'children'),
               Output('chart_AUC_test_02', 'children'),
               Output('chart_AUC_test_03', 'children'),
               Output('chart_AUC_test_04', 'children'),

               Output('chart_dnst_01', 'children'),
               Output('chart_dnst_02', 'children'),
               Output('chart_dnst_03', 'children'),
               Output('chart_dnst_04', 'children'),
               ],
              [Input('upload-data', 'contents')])

def update_graph(contents):

    plots_pair  = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))]*4))
    plots_pc    = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))]*4))
    plots_fi    = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))]*4))
    plots_train = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))]*4))
    plots_test  = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))]*4))
    plots_dnst  = dict((i, plt) for i, plt in enumerate([html.Img(src=app.get_asset_url(U.URL_empty))] * 4))

    if contents:
        df0 = U.parse_data(contents[0])
        idx_target = 0
        B.clear_cache()
        plots_pair = B.get_pairplots(df0,idx_target,plots_pair)
        plots_pc = B.get_pc(df0, idx_target, plots_pc)
        plots_fi = B.get_feature_importance(df0, idx_target,plots_fi)
        plots_train, plots_test = B.get_roc(df0, idx_target,plots_train,plots_test)
        plots_dnst = B.get_density(df0, idx_target,plots_dnst)

    return plots_pair[0],plots_pair[1],plots_pair[2],plots_pair[3],\
           plots_pc[0],plots_pc[1],plots_pc[2],plots_pc[3], \
           plots_fi[0],plots_fi[1],plots_fi[2],plots_fi[3],\
           plots_train[0],plots_train[1],plots_train[2],plots_train[3],\
           plots_test[0],plots_test[1],plots_test[2],plots_test[3],\
           plots_dnst[0],plots_dnst[1],plots_dnst[2],plots_dnst[3]
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    layout_main = html.Div([dbc.Card(dbc.CardBody(div_main), color=U.clr_pad)])
    app.layout = layout_main
    app.run_server(debug=False)

