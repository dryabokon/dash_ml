import io
from io import BytesIO
import pandas as pd
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import base64
# ----------------------------------------------------------------------------------------------------------------------
class DashDraw(object,):
    dct_style_image = {'textAlign': 'center', 'height': '150px', 'margin': '0px', 'padding': '0px', 'object-fit': 'scale-down'}
    dct_style_text  = {'textAlign': 'center','object-fit': 'scale-down'}
    URL_image_header = './logo/header.png'
# ----------------------------------------------------------------------------------------------------------------------
    def __init__(self,folder_out=None,dark_mode=False):
        self.folder_out = folder_out
        self.dark_mode = dark_mode

        if dark_mode:
            self.clr_chart = "#2B2B2B"
            self.clr_pad = "#222222"
            self.clr_grid = "#404040"
            self.clr_banner = "#404040"
            self.clr_header = "#086A6A"
            self.clr_sub_header = "#214646"
            self.URL_empty = './logo/empty_dark2.png'
            self.plotly_template =  'plotly_dark'
            self.dct_style_text['color'] = "#FFFFFF"
        else:
            self.clr_chart = "#FFFFFF"
            self.clr_pad = "#EEEEEE"
            self.clr_grid = "#C0C0C0"
            self.clr_banner = "#E0E0E0"
            self.clr_header = "#086A6A"
            self.clr_sub_header = "#83B4B4"
            self.URL_empty = './logo/empty_light2.png'
            self.plotly_template = 'plotly_white'
            self.dct_style_text['color'] = "#222222"

        return
# ----------------------------------------------------------------------------------------------------------------------
    def filename_img_to_uri(self, filename):
        buffer_img = open(filename, 'rb').read()
        encoded = base64.b64encode(buffer_img)
        result = 'data:image/png;base64,{}'.format(encoded)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def fig_to_uri(self, in_fig, close_all=True, **save_args):
        out_img = BytesIO()
        in_fig.savefig(out_img, format='png', **save_args, facecolor=in_fig.get_facecolor())
        if close_all:
            in_fig.clf()
            plt.close('all')
        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)
# ----------------------------------------------------------------------------------------------------------------------
    def draw_text_framed(self,text,color):
        return html.Div([dbc.Card(dbc.CardBody([html.Div([html.H2(text)], style=self.dct_style_text)]), color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_text_unframed(self, text, color):
        return html.Div([dbc.Card(([html.Div([html.H2(text)], style=self.dct_style_text)]), color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_fig_mpl(self,fig_mpl,color):
        return html.Div([dbc.Card(dbc.CardBody([html.Div([html.Img(src=self.fig_to_uri(fig_mpl))], style=self.dct_style_text)]),color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_image_URL_framed(self, URL,color):
        return html.Div([dbc.Card(dbc.CardBody([html.Div([html.Img(src=URL)], style=self.dct_style_text)]),color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_image_URL_unframed(self, URL, color):
        return html.Div([dbc.Card(([html.Div([html.Img(src=URL)], style=self.dct_style_text)]),color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_chart_framed(self,id,color):
        return html.Div([dbc.Card(dbc.CardBody([html.Div(id=id, style=self.dct_style_text)]),color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_chart_unframed(self, id, color):
        return html.Div([dbc.Card(([html.Div(id=id, style=self.dct_style_text)]), color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_figure_px(self,fig_px):
        #https://plotly.com/python/reference/layout/
        fig_px.update_xaxes(gridcolor=self.clr_grid,showline=False)
        fig_px.update_yaxes(gridcolor=self.clr_grid,showline=False)
        fig_px.update_layout(
            template=self.plotly_template,
            margin_l=0,
            margin_r=0,
            margin_t=0,
            margin_b=0,
            plot_bgcolor=self.clr_chart,
            paper_bgcolor=self.clr_chart,
        )
        return  html.Div([dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_px)],style=self.dct_style_image),color=self.clr_chart)])
# ----------------------------------------------------------------------------------------------------------------------
    def draw_picker(self, id, text, color):
        return html.Div([dbc.Card(dbc.CardBody([html.Div([self.get_picker(id, text)], style=self.dct_style_text)]), color=color)])
# ----------------------------------------------------------------------------------------------------------------------
    def get_picker(self,id,text):
        return dcc.Upload(id=id,children=html.Div([html.H2(text)]),style=self.dct_style_text,multiple=True)
# ----------------------------------------------------------------------------------------------------------------------
    def parse_data(self,contents):
        content_type, content_string = contents.split(',')
        df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')),delimiter='\t')
        df = self.prepreocess_data(df)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def prepreocess_data(self,df):
        #titanic case
        if 'alive' in df.columns.to_numpy():df.drop(labels=['alive'], axis=1, inplace=True)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def prepare_icon(self):
        from PIL import Image
        img = Image.open('./assets/Image1.png')
        img.save(self.folder_out+'favicon.ico', format='ICO', sizes=[(32, 32)])
        return
# ----------------------------------------------------------------------------------------------------------------------

