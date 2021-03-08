import os
import numpy
import dash_html_components as html
import tempfile
# ----------------------------------------------------------------------------------------------------------------------
from classifier import classifier_LM
from classifier import classifier_RF
from classifier import classifier_SVM
from classifier import classifier_KNN

import tools_DF
import tools_plot_v2
import tools_ML_v2
import tools_IO
import tools_feature_importance
# ----------------------------------------------------------------------------------------------------------------------
class Business_logic(object):
    def __init__(self,app,folder_out,dark_mode):
        self.folder_out = folder_out
        self.P = tools_plot_v2.Plotter(folder_out,dark_mode)
        self.app = app
        return
# ----------------------------------------------------------------------------------------------------------------------
    def clear_cache(self):
        tools_IO.remove_files(self.folder_out, list_of_masks='*.*', create=False)
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

