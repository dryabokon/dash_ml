import pandas as pd
import os
import numpy
import dash_html_components as html
import tempfile
from plotly.tools import mpl_to_plotly
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_LM
from classifier import classifier_RF
from classifier import classifier_SVM
from classifier import classifier_KNN
# ----------------------------------------------------------------------------------------------------------------------
from TS import TS_AutoRegression
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_plot_v2
import tools_ML_v2
import tools_TS
import tools_IO
import tools_feature_importance
# ----------------------------------------------------------------------------------------------------------------------
class Business_logic(object):
    def __init__(self,app,folder_out,dark_mode):
        self.folder_out = folder_out
        self.P = tools_plot_v2.Plotter(folder_out,dark_mode)
        self.app = app
        self.filename_retro_df = 'retro.csv'

        self.TS = tools_TS.tools_TS(Classifier=TS_AutoRegression.TS_AutoRegression(folder_out), dark_mode=dark_mode, folder_out=folder_out)
        self.clear_cache()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def clear_cache(self,list_of_masks='*.*'):
        tools_IO.remove_files(self.folder_out, list_of_masks=list_of_masks, create=False)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_pairplots(self,df0,idx_target,pairplots):

        FI = tools_feature_importance.evaluate_feature_importance(df0, idx_target)
        best_idx = numpy.argsort(-FI['F_score'].to_numpy())
        best_features = FI['features'].to_numpy()[best_idx][:4]
        target = df0.columns[idx_target]

        cnt = 0
        for i in range(len(best_features)):
            for j in range(i + 1, len(best_features)):
                c1, c2 = best_features[i], best_features[j]
                df = df0[[target, c1, c2]]
                df = df.dropna()
                df[target] = (df[target] <= 0).astype(int)
                df = tools_DF.hash_categoricals(df)

                URL = next(tempfile._get_candidate_names()) + '.png'
                self.P.plot_2D_features_v3(df,remove_legend=True,add_noice=True,transparency=0.75,filename_out=URL)
                pairplots[cnt] = [html.Img(src=self.app.get_asset_url(URL))]
                cnt += 1
                if cnt == 4: break
            if cnt == 4: break

        return pairplots
# ----------------------------------------------------------------------------------------------------------------------
    def get_pc(self, df0, idx_target, pca_plots):

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        df = df0.dropna()
        df[target] = (df[target] <= 0).astype(int)
        df = tools_DF.hash_categoricals(df)

        self.P.plot_SVD(df, idx_target, 'dim_SVD.png')
        self.P.plot_PCA(df, idx_target, 'dim_PCA.png')
        self.P.plot_tSNE(df, idx_target, 'dim_tSNE.png')
        self.P.plot_ISOMAP(df, idx_target, 'dim_ISOMAP.png')

        for i,filename in enumerate(['dim_SVD.png','dim_PCA.png','dim_tSNE.png','dim_ISOMAP.png']):
            URL = next(tempfile._get_candidate_names()) + '.png'
            os.rename(self.folder_out+filename, self.folder_out+URL)
            pca_plots[i] = [html.Img(src=self.app.get_asset_url(URL))]

        return pca_plots
# ----------------------------------------------------------------------------------------------------------------------
    def get_feature_importance(self, df, idx_target,fi_plots):

        FI = tools_feature_importance.evaluate_feature_importance(df, idx_target)
        for i,t in enumerate(['F_score', 'R2', 'C', 'XGB']):
            filename_out = 'FI_%s.png' % t
            self.P.plot_hor_bars(FI[t].to_numpy(), FI['features'].to_numpy(), legend=t, filename_out=filename_out)
            URL = next(tempfile._get_candidate_names()) + '.png'
            os.rename(self.folder_out+filename_out, self.folder_out+URL)
            fi_plots[i] = [html.Img(src=self.app.get_asset_url(URL))]

        return fi_plots
# ----------------------------------------------------------------------------------------------------------------------
    def get_roc(self, df0, idx_target,plots_train, plots_test):

        columns = df0.columns.to_numpy()
        target = columns[idx_target]
        df = df0.dropna()
        df[target] = (df[target] <= 0).astype(int)
        df = tools_DF.hash_categoricals(df)

        for i,C in enumerate([classifier_LM.classifier_LM(),classifier_SVM.classifier_SVM(),classifier_RF.classifier_RF(),classifier_KNN.classifier_KNN()]):
            ML = tools_ML_v2.ML(C,self.folder_out,self.P.dark_mode)
            ML.E2E_train_test_df(df, idx_target,do_pca=False)

            URL = next(tempfile._get_candidate_names()) + '.png'
            os.rename(self.folder_out + 'ROC_train.png', self.folder_out + URL)
            plots_train[i] = [html.Img(src=self.app.get_asset_url(URL))]

            URL = next(tempfile._get_candidate_names()) + '.png'
            os.rename(self.folder_out + 'ROC_test.png', self.folder_out + URL)
            plots_test[i] = [html.Img(src=self.app.get_asset_url(URL))]

        return plots_train, plots_test
# ----------------------------------------------------------------------------------------------------------------------
    def get_density(self, df0, idx_target,plots_dnst):
        FI = tools_feature_importance.evaluate_feature_importance(df0, idx_target)
        best_idx = numpy.argsort(-FI['F_score'].to_numpy())
        best_features = FI['features'].to_numpy()[best_idx][:4]
        target = df0.columns[idx_target]
        df = df0[[target, best_features[0], best_features[1]]]

        df = df.dropna()
        df[target] = (df[target] <= 0).astype(int)
        df = tools_DF.hash_categoricals(df)

        for i,C in enumerate([classifier_LM.classifier_LM(),classifier_SVM.classifier_SVM(),classifier_RF.classifier_RF(),classifier_KNN.classifier_KNN()]):
            ML = tools_ML_v2.ML(C,self.folder_out,self.P.dark_mode)
            ML.E2E_train_test_df(df, 0, do_pca=False)
            ML.plot_density_2d(df, idx_target=0,N =30, filename_out='density.png')

            URL = next(tempfile._get_candidate_names()) + '.png'
            os.rename(self.folder_out + 'density.png', self.folder_out + URL)
            plots_dnst[i] = [html.Img(src=self.app.get_asset_url(URL))]

        return plots_dnst
# ----------------------------------------------------------------------------------------------------------------------
    def get_TS_prediction_plot_html(self, df0,idx_target,train_size,n_steps,plots_acc):
        self.clear_cache(list_of_masks='*.png')

        if not os.path.exists(self.folder_out+self.filename_retro_df):
            df_retro = pd.DataFrame({'GT': df0.iloc[:, idx_target],
                                     'predict': numpy.full(df0.shape[0], numpy.nan),
                                     'predict_ahead': numpy.full(df0.shape[0], numpy.nan),
                                     'predict_ahead_min': numpy.full(df0.shape[0], numpy.nan),
                                     'predict_ahead_max': numpy.full(df0.shape[0], numpy.nan),
                                     })
            df_retro.to_csv(self.folder_out + self.filename_retro_df, index=False, sep='\t')
        else:
            df_retro = pd.read_csv(self.folder_out + self.filename_retro_df, sep='\t')



        filename_out = 'pred_ahead_%s.png' % self.TS.classifier.name

        df_step = self.TS.predict_n_steps_ahead(df0, idx_target,n_steps=20,do_debug=False)
        df_step['GT'] = numpy.full(n_steps, numpy.nan)
        df_step['predict'] = numpy.full(n_steps, numpy.nan)

        df_retro = df_retro.append(df_step, ignore_index=True)
        x_range = [max(0, df_retro.shape[0] - n_steps * 10), df_retro.shape[0]]
        self.P.TS_matplotlib(df_retro, [0, 2, 1], None, idxs_fill=[3, 4], x_range=x_range,filename_out=filename_out)
        df_retro.drop(numpy.arange(df_retro.shape[0] - df_step.shape[0] + 1, df_retro.shape[0], 1), axis=0,inplace=True)
        df_retro['GT'].iloc[-1] = float(df0.iloc[-1, idx_target])
        df_retro['predict'].iloc[-1] = df_retro['predict_ahead'].iloc[-1]
        df_retro['predict_ahead'] = numpy.nan
        df_retro.to_csv(self.folder_out + self.filename_retro_df, index=False, sep='\t')

        URL = next(tempfile._get_candidate_names()) + '.png'
        os.rename(self.folder_out + filename_out, self.folder_out + URL)
        plot_TS  = [html.Img(src=self.app.get_asset_url(URL))]

        xxx = df_retro[['GT','predict']][train_size:].dropna()
        Y_test = xxx['GT'].to_numpy()
        Y_test_pred = xxx['predict'].to_numpy()

        if Y_test.shape[0]>=2:
            self.P.plot_fact_predict(Y_test, Y_test_pred,filename_out='test_fact_pred.png')
            self.P.plot_hist(Y_test - Y_test_pred ,filename_out='test_err.png')

            for i, filename in enumerate(['test_fact_pred.png','test_err.png']):
                URL = next(tempfile._get_candidate_names()) + '.png'
                os.rename(self.folder_out + filename, self.folder_out + URL)
                plots_acc[i]=([html.Img(src=self.app.get_asset_url(URL))])

        return plot_TS,plots_acc
# ----------------------------------------------------------------------------------------------------------------------