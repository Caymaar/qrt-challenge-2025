import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
from IPython.display import HTML
from pathlib import Path
from autoviz import AutoViz_Class
from datetime import datetime

def display_html_report(html_file):
    def start_server():
        server_address = ('', 8000)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        httpd.serve_forever()

    # Démarrer le serveur dans un thread séparé
    thread = threading.Thread(target=start_server)
    thread.daemon = True
    thread.start()

    # Style CSS amélioré avec arrière-plan
    html_content = f'''
    <div style="
        padding: 20px;
        background: #f5f5f5;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <iframe 
            src="http://localhost:8000/{html_file}" 
            width="100%" 
            height="800px"
            style="
                border: none;
                border-radius: 8px;
                background: white;
            "
        ></iframe>
    </div>
    '''

    display(HTML(html_content))

class ShapReport:
    def __init__(self, model, X_train, predict_function=None, nb_features=None):
        """
        :param model: Le modèle dont on veut expliquer les prédictions.
        :param X_train: Le jeu de données d'arrière-plan (pandas DataFrame).
        :param predict_function: Fonction de prédiction. Si None, on utilise model.predict.
        """
        self.model = model
        self.X_train = X_train.copy()
        if predict_function is None:
            self.predict_function = model.predict
        else:
            self.predict_function = predict_function

        if nb_features is None:
            self.nb_features = len(X_train.columns)
        else:
            self.nb_features = nb_features

        # Création de l'explainer SHAP à partir de la fonction de prédiction et du jeu de données
        self.explainer = shap.Explainer(self.predict_function, self.X_train)
        self.shap_values = self.explainer(self.X_train)

    def _fig_to_base64(self, fig):
        """Convertit une figure matplotlib en chaîne base64."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

    def generate_report(self, output_file="shap_report.html"):
        """
        Génère un rapport SHAP HTML en enregistrant plusieurs graphiques.
        
        :param output_html: Chemin du fichier HTML généré.
        """
        plots = {}

        # 1. Summary Plot (bar)
        plt.figure()
        shap.summary_plot(self.shap_values, self.X_train, plot_type="bar", show=False, max_display=self.nb_features)
        fig_bar = plt.gcf()
        plots['summary_bar'] = self._fig_to_base64(fig_bar)

        # 2. Summary Plot (dot)
        plt.figure()
        shap.summary_plot(self.shap_values, self.X_train, show=False, max_display=self.nb_features)
        fig_dot = plt.gcf()
        plots['summary_dot'] = self._fig_to_base64(fig_dot)

        # 3. Dependence Plots pour les variables les plus importantes
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        top_feature_indices = np.argsort(mean_abs_shap)[::-1][:10]
        top_features = self.X_train.columns[top_feature_indices]

        for feature in top_features:
            plt.figure()
            shap.dependence_plot(feature, self.shap_values.values, self.X_train, show=False)
            fig_dep = plt.gcf()
            plots[f'dependence_{feature}'] = self._fig_to_base64(fig_dep)

        # 4. (NOUVEAU) Feature Importances du modèle s'il en dispose
        feature_importances_html = ""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

            # On crée un bar plot horizontal pour illustrer les importances
            # --> Augmentation de la taille de la figure
            plt.figure(figsize=(25, 8))  # Ajustez selon vos préférences (largeur x hauteur)
            sorted_idx = np.argsort(importances)

            plt.barh(range(len(importances)), importances[sorted_idx], align='center')
            plt.yticks(range(len(importances)), self.X_train.columns[sorted_idx])
            plt.xlabel("Feature Importance")
            plt.title("Feature Importances (Model-based)")

            # --> Ajustement automatique des marges
            plt.tight_layout()

            fig_fi = plt.gcf()

            # On l'enregistre en base64
            plots['model_feature_importances'] = self._fig_to_base64(fig_fi)

            # On prépare le HTML associé
            feature_importances_html = f"""
            <div class="plot">
                <h2>Feature Importances (Model-based)</h2>
                <img src="data:image/png;base64,{plots['model_feature_importances']}" 
                    alt="Feature Importances du modèle">
            </div>
            """

        # Création du template HTML final avec trois colonnes (et une section optionnelle)
        dependence_plots_html = ""
        for feature in top_features:
            dependence_plots_html += f"""
                <div class="plot">
                    <h2>Dependence Plot pour la variable : {feature}</h2>
                    <img src="data:image/png;base64,{plots['dependence_' + feature]}" 
                        alt="Dependence Plot pour {feature}">
                </div>
            """

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Rapport SHAP</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {shap.getjs()}
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1, h2 {{
                    color: #333;
                }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .column {{
                    flex: 1;
                    min-width: 300px;
                }}
                .plot {{
                    margin-bottom: 30px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    padding: 5px;
                    background: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <h1>Rapport SHAP</h1>
            <div class="container">
                <!-- Colonne 1 : Summary Plot (Bar) -->
                <div class="column">
                    <div class="plot">
                        <h2>Summary Plot (Bar)</h2>
                        <img src="data:image/png;base64,{plots['summary_bar']}" alt="Summary Plot Bar">
                    </div>
                    {feature_importances_html}
                </div>
                <!-- Colonne 2 : Summary Plot (Dot) -->
                <div class="column">
                    <div class="plot">
                        <h2>Summary Plot (Dot)</h2>
                        <img src="data:image/png;base64,{plots['summary_dot']}" alt="Summary Plot Dot">
                    </div>
                </div>
                <!-- Colonne 3 : Dependence Plots -->
                <div class="column">
                    {dependence_plots_html}
                </div>
            </div>
        </body>
        </html>
        """

        # Enregistrement du rapport HTML
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"Le rapport SHAP a été sauvegardé dans : {output_file}")
        self.output_file = output_file

    def display(self, output_file=None):
        if output_file is None:
            output_file = self.output_file
        display_html_report(output_file)

class EDAReport:
    def __init__(self, df, target_variables, custom_plot_dir="report/eda", title=None, output_file=None, sep=",", header=0, verbose=0, lowess=False, chart_format="html"):
        """
        Initialise le rapport EDA.
        
        :param df: DataFrame à analyser.
        :param target_variables: Liste des variables cibles sur lesquelles réaliser l'EDA.
        :param custom_plot_dir: Répertoire principal pour sauvegarder les graphiques AutoViz.
        :param title: Titre du rapport (et nom du sous-dossier). Par défaut, "EDA_" suivi de la date et heure actuelle.
        :param output_file: Chemin complet du fichier HTML de rapport. Si None, il sera enregistré dans custom_plot_dir/<title>/ sous le nom 'EDA_report.html'.
        :param sep: Délimiteur utilisé dans les données (passé à AutoViz).
        :param header: Numéro de la ligne contenant les noms de colonnes (passé à AutoViz).
        :param verbose: Niveau de verbosité pour AutoViz.
        :param lowess: Indique si le lissage Lowess doit être appliqué.
        :param chart_format: Format des graphiques générés par AutoViz.
        """
        self.df = df
        self.target_variables = target_variables
        self.custom_plot_dir = Path(custom_plot_dir)
        self.title = title if title is not None else "EDA_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = output_file
        self.sep = sep
        self.header = header
        self.verbose = verbose
        self.lowess = lowess
        self.chart_format = chart_format
        
        # Instance d'AutoViz_Class
        self.AV = AutoViz_Class()
        
        # Dictionnaire des réglages pour chaque type de graphique (nom de fichier sans le préfixe)
        self.files_settings = {
            "distplots_cats.html": {"title": "CATEGORIES", "width": "100%", "height": "400"},
            "pair_scatters.html": {"title": "PAIR SCATTERS", "width": "100%", "height": "400"},
            "scatterplots.html": {"title": "SCATTERPLOTS", "width": "100%", "height": "400"},
            "cat_var_plots.html": {"title": "CATEGORICAL VARIABLES", "width": "100%", "height": "400"},
            "heatmaps.html": {"title": "HEATMAPS", "width": "100%", "height": "800"},
            "distplots_nums.html": {"title": "NUMERICAL DISTRIBUTIONS", "width": "100%", "height": "400"},
            "violinplots.html": {"title": "VIOLIN PLOTS", "width": "100%", "height": "800"},
            "kde_plots.html": {"title": "KDE PLOTS", "width": "100%", "height": "400"},
        }

    def generate_report(self, max_rows=150000, max_cols=30):
        """
        Génère le rapport EDA en exécutant AutoViz pour chaque variable cible, puis
        combine les graphiques générés dans un fichier HTML unique avec un en-tête pour chaque variable.
        
        Returns:
            str: Le chemin du rapport HTML généré.
        """
        # Création du répertoire de sauvegarde pour le rapport (custom_plot_dir/title)
        report_dir = self.custom_plot_dir / self.title
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Pour chaque variable cible, exécuter AutoViz et sauvegarder les graphiques dans report_dir/<target_variable>
        for target_variable in self.target_variables:
            print(f"Analyse pour la variable cible '{target_variable}'")
            # Utiliser un sous-dossier dédié à chaque variable cible
            save_dir = report_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            _ = self.AV.AutoViz(
                filename="",
                sep=self.sep,
                depVar=target_variable,
                dfte=self.df,
                header=self.header,
                lowess=self.lowess,
                chart_format=self.chart_format,
                max_rows_analyzed=min(self.df.shape[0], max_rows),
                max_cols_analyzed=min(self.df.shape[1], max_cols),
                save_plot_dir=str(save_dir),
                verbose=self.verbose
            )
        
        # Construction du contenu HTML final
        final_contents = []
        # En-tête principal du rapport
        final_contents.append('<h1 style="text-align: center; margin: 30px 0;">Rapport EDA</h1>')
        
        for target_variable in self.target_variables:
            target_dir = report_dir / target_variable
            if target_dir.exists():
                # En-tête unique pour la variable cible (centré, en majuscules)
                final_contents.append(f'<h2 style="text-align: center; margin: 20px 0;">{target_variable.upper()}</h2>')
                # Container pour les graphiques de cette variable cible
                plots_html = '<div class="container">'
                for html_file in sorted(target_dir.glob("*.html")):
                    file_name = html_file.name
                    settings = self.files_settings.get(file_name, {"title": file_name, "width": "100%", "height": "800"})
                    # Calculer le chemin relatif pour l'iframe
                    relative_path = os.path.relpath(html_file, report_dir)
                    plots_html += f"""
                    <div class="plot">
                        <h3 style="margin: 0; padding: 5px 0;">{settings['title']}</h3>
                        <iframe src="{relative_path}" width="{settings['width']}" height="{settings['height']}" 
                                style="border: none; display: block; margin: 0; padding: 0;"></iframe>
                    </div>
                    """
                plots_html += "</div>"
                final_contents.append(plots_html)
            else:
                print(f"Le répertoire {target_dir} n'existe pas.")
        
        combined_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.title} - Rapport EDA AutoViz</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background: #f5f5f5;
                }}
                h1, h2, h3 {{
                    color: #333;
                    text-align: center;
                }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }}
                .plot {{
                    background: white;
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    width: 100%;
                    max-width: 1100px;
                }}
                iframe {{
                    width: 100%;
                    border: none;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            {"".join(final_contents)}
        </body>
        </html>
        """
        
        # Définir le chemin de sortie du rapport
        if self.output_file is None:
            self.output_file = report_dir / "EDA_report.html"
        else:
            self.output_file = Path(self.output_file)
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(combined_html)
        
        print(f"Le rapport EDA a été sauvegardé dans : {self.output_file}")
    
    def display(self, output_file=None):
        if output_file is None:
            output_file = self.output_file
        display_html_report(output_file)