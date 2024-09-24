import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import openpyxl

# Imports pour l'inférence
import torch
import torch.nn as nn
from transformers import CamembertTokenizer
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import os

# Définir le chemin du répertoire courant
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Définition des classes du modèle
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        concat_attention = attn_output.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        if mask is not None:
            scores += mask
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, look_ahead_mask):
        attn_output = self.mha(x, x, x, look_ahead_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, look_ahead_mask):
        seq_len = x.size(1)
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, look_ahead_mask)
        return x

    @staticmethod
    def positional_encoding(position_max, d_model):
        pos_encoding = np.zeros((position_max, d_model))
        for pos in range(position_max):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        return torch.tensor(pos_encoding, dtype=torch.float32).unsqueeze(0)

class GPT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_encoding, rate=0.1):
        super(GPT, self).__init__()
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, pe_encoding, rate)
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, decoder_input, training=False):
        decoder_input = decoder_input.to(next(self.parameters()).device).long()
        look_ahead_mask = self.create_look_ahead_mask(decoder_input.size(1)).to(decoder_input.device)
        dec_output = self.decoder(decoder_input, training, look_ahead_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    @staticmethod
    def create_look_ahead_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask[mask == 1] = float('-inf')
        mask[mask == 0] = 0
        return mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 256 

@st.cache_resource
def load_tokenizer():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    return tokenizer

@st.cache_resource
def load_model():
    tokenizer = load_tokenizer()
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab) + 1
    num_layers = 8  
    d_model = 512
    num_heads = 8
    dff = 2048
   
    # Téléchargement des poids depuis Hugging Face (compte Finsaco)
    try:
        model_path = hf_hub_download(
            repo_id="Finsaco/GPT-Slim",    
            filename="best_model.pth",
            revision="master"  # Assurez-vous que la branche est correcte
        )
        print(f"Poids téléchargés dans : {model_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement des poids depuis Hugging Face : {e}")
        raise

    model = GPT(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                vocab_size=vocab_size, pe_encoding=SEQ_LENGTH, rate=0.2)
  
    try:
        state_dict = torch.load(model_path, map_location=device)

        # Afficher la taille des matrices d'embedding et de la dernière couche
        print("Taille de 'decoder.embedding.weight' dans le modèle sauvegardé :", state_dict['decoder.embedding.weight'].size())
        print("Taille de 'final_layer.weight' dans le modèle sauvegardé :", state_dict['final_layer.weight'].size())

        # Charger le state_dict dans le modèle
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        print("Modèle chargé avec succès avec la taille de vocabulaire ajustée.")
    except Exception as e:
        print(f"Erreur lors du chargement des poids dans le modèle : {e}")
        raise

    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    # Appliquer top_k
    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, top_k)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits[logits < min_top_k] = filter_value

    # Appliquer top_p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Retirer les tokens dont la probabilité cumulative dépasse top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Décaler les masques pour inclure le premier token qui dépasse top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value

    return logits

def generate_tokens(model, tokenizer, input_text, num_generated_tokens=250, temperature=0.7, top_k=0, top_p=0.9,
                   repetition_penalty=1.2, max_words_per_line=20):
    model.eval()
    
    # Encodage du texte d'entrée
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Initialisation des tokens générés
    generated_tokens = input_ids.clone()
    generated_tokens_list = []  # Pour stocker uniquement les tokens générés
    
    with torch.no_grad():
        for _ in range(num_generated_tokens):
            # Utiliser uniquement les derniers SEQ_LENGTH tokens pour le modèle
            input_window = generated_tokens[:, -SEQ_LENGTH:]
            
            # Passer les tokens à travers le modèle
            outputs = model(input_window, training=False)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Appliquer la pénalité de répétition
            for token_id in set(generated_tokens[0].tolist()):
                next_token_logits[0, token_id] /= repetition_penalty
            
            # Appliquer le filtrage top_k/top_p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # Calculer les probabilités
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            # Échantillonner le prochain token
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            
            # Vérifier si le token de fin est généré
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Ajouter le token généré à la séquence complète
            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
            
            # Stocker le token généré séparément
            generated_tokens_list.append(next_token_id.item())
            
            # La fenêtre glissante est automatiquement gérée par input_window = generated_tokens[:, -SEQ_LENGTH:]
    
    # Décoder uniquement les tokens générés
    generated_text = tokenizer.decode(
        generated_tokens_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Formater le texte généré en paragraphes
    formatted_generated_text = generated_text.replace('. ', '.\n\n')
    
    # Insérer des sauts de ligne pour les lignes dépassant max_words_per_line mots
    lines = formatted_generated_text.split('\n')
    processed_lines = []
    for line in lines:
        words = line.split()
        if len(words) > max_words_per_line:
            # Diviser la ligne en segments de max_words_per_line mots
            chunks = [
                ' '.join(words[i:i + max_words_per_line])
                for i in range(0, len(words), max_words_per_line)
            ]
            new_line = '\n'.join(chunks)
            processed_lines.append(new_line)
        else:
            processed_lines.append(line)
    processed_generated_text = '\n'.join(processed_lines)
    
    # Affichage final avec les étiquettes
    final_output = f"[PROMPT :] {input_text}\n\n[TEXTE GÉNÉRÉ :]\n{processed_generated_text}"
    return final_output


# Configuration de l'application avec thème automatique
st.set_page_config(page_title="Présentation du Projet LM", layout="wide", initial_sidebar_state="expanded")

# Fonction pour détecter le thème de Streamlit
def get_plotly_template():
    try:
        theme = st.get_option("theme.base")
        if theme == "dark":
            return "plotly_dark"
        else:
            return "plotly_white"
    except:
        # Par défaut, utiliser 'plotly_white'
        return "plotly_white"

# Fonction de conversion SI à float
def convert_si_to_float(value):
    multipliers = {
        'n': 1e-9,
        'µ': 1e-6,
        'm': 1e-3,
        '': 1
    }
    if isinstance(value, str):
        for key, multiplier in multipliers.items():
            if value.endswith(key):
                try:
                    # Remplacer le préfixe SI et convertir en float
                    return float(value.replace(key, '')) * multiplier
                except ValueError:
                    return None
    return value

# Création des pages
def introduction():
    st.title("Introduction")
    st.write("""
    Ce projet a pour objectif de construire un modèle de génération de texte en français, 
    inspiré des architectures GPT-1 et GPT-2.\n
    À travers cette démo, nous allons explorer 
    les différentes étapes de notre démarche, de la conception du modèle à l'inférence finale, 
    en passant par l'analyse comparative avec les modèles GPT-1 et GPT-2 et l'analyse des mètriques de notre entraînement.
    """)

def schema_modele():
    st.title("Schéma du Modèle")
    st.write("Voici le schéma de notre modèle :")

    # Chemin vers votre image (chemin relatif)
    image_path = os.path.join(CURRENT_DIR, "Schéma Transformers du projet LM Pytorch.png")

    # Vérifier que le fichier existe
    try:
        # Convertir l'image en base64 pour l'intégrer dans le code HTML
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        # Code HTML pour intégrer OpenSeadragon
        html_code = f"""
        <div id="openseadragon1" style="width: 100%; height: 80vh;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.js"></script>
        <script type="text/javascript">
            var viewer = OpenSeadragon({{
                id: "openseadragon1",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
                tileSources: {{
                    type: 'image',
                    url: 'data:image/png;base64,{encoded_string}'
                }},
                gestureSettingsMouse: {{
                    clickToZoom: true,
                    dblClickToZoom: true,
                    flickEnabled: true,
                    pinchToZoom: true
                }}
            }});
        </script>
        """

        # Afficher le code HTML dans Streamlit
        st.components.v1.html(html_code, height=600)

    except FileNotFoundError:
        st.error(f"L'image n'a pas été trouvée à l'emplacement : {image_path}")

def comparaison_modeles():
    st.title("Comparaison des Modèles")

    # Données des modèles
    data = {
        'Paramètre': [
            'Tokenizer',
            'Nombre de couches',
            'Dimension des embeddings',
            'Nombre de têtes d\'attention',
            'Taille du contexte',
            'Batch Size',
            'Nombre de paramètres',
            'Taille du dataset',
            'Nombre de GPU utilisés',
            'Temps d\'entraînement (heures sur 1 GPU NVIDIA RTX 4090)',
            'Perplexité'
        ],
        'Unité': [
            'Tokens',
            'Couches',
            'Dimensions',
            'Têtes',
            'Tokens',
            'Batch',
            'Paramètres',
            'Corpus',
            'GPUs',
            'Heures',
            'Score'
        ],
        'GPT-1': [
            'BPE (vocab ~40k)', 12, 768, 12, 512, 64, '117M', 
            'BookCorpus (~7 000 livres)', 8, 1_095, '18.4 (BookCorpus)'
        ],
        'GPT-2 Small': [
            'BPE (vocab ~50k)', 12, 768, 12, 1024, 64, '124M', 
            'WebText (~8 millions de pages web)', 256, 16_349, '29.41 (WikiText-2)'
        ],
        'GPT-Slim (Notre modèle)': [
            'Camembert (vocab ~32k)', 8, 512, 8, 256, 32, '58M', 
            '19 échantillonnages (1320 livres sur 13200 livres)', 1, 195, '19.63 (Corpus livres français)'
        ]
    }

    # Conversion en DataFrame
    df = pd.DataFrame(data)

    # Affichage du tableau des paramètres
    st.subheader("Tableau Comparatif des Paramètres")
    st.dataframe(df.set_index('Paramètre'))

    st.write("""
    **Perplexité** : La perplexité est une métrique utilisée pour évaluer la qualité d'un modèle de langage. Elle mesure à quel point le modèle est "surpris" par les données de test. Une perplexité plus faible indique que le modèle prédit mieux les données.
    """)

    gpu_types = {
        'GPT-1': 'V100',
        'GPT-2 Small': 'V100',
        'GPT-Slim (Notre modèle)': 'RTX 4090'
    }

    params_to_plot = [
        'Nombre de couches',
        'Dimension des embeddings',
        'Nombre de têtes d\'attention',
        'Taille du contexte',
        'Nombre de GPU utilisés',
        'Temps d\'entraînement (heures sur 1 GPU NVIDIA RTX 4090)'
    ]

    st.subheader("Visualisation des Paramètres Clés")
    for param in params_to_plot:
        unit = df.loc[df['Paramètre'] == param, 'Unité'].values[0]
        values = {
            'Modèle': ['GPT-1', 'GPT-2 Small', 'GPT-Slim (Notre modèle)'],
            'Valeur': [
                df.loc[df['Paramètre'] == param, 'GPT-1'].values[0],
                df.loc[df['Paramètre'] == param, 'GPT-2 Small'].values[0],
                df.loc[df['Paramètre'] == param, 'GPT-Slim (Notre modèle)'].values[0]
            ]
        }
        param_df = pd.DataFrame(values)

        # Vérifier si les valeurs sont numériques
        if isinstance(values['Valeur'][0], (int, float)):
            # Création du graphique
            fig = go.Figure()

            if param == 'Nombre de GPU utilisés':
                # Ajouter le type de GPU dans le texte
                param_df['Text'] = [
                    f"{val} * {gpu_types[model]}" for model, val in zip(param_df['Modèle'], param_df['Valeur'])
                ]

                # Séparer les données pour GPT-Slim
                df_gpu_main = param_df[param_df['Modèle'] != 'GPT-Slim (Notre modèle)']
                df_gpu_slim = param_df[param_df['Modèle'] == 'GPT-Slim (Notre modèle)']

                # Ajouter les barres pour GPT-1 et GPT-2 Small
                fig.add_trace(go.Bar(
                    x=df_gpu_main['Modèle'],
                    y=df_gpu_main['Valeur'],
                    text=df_gpu_main['Text'],
                    textposition='auto',
                    marker_color=['#636EFA', '#EF553B'],  # Couleurs originales
                    name=param
                ))

                # Ajouter la barre pour GPT-Slim
                fig.add_trace(go.Bar(
                    x=df_gpu_slim['Modèle'],
                    y=df_gpu_slim['Valeur'],
                    text=df_gpu_slim['Text'],
                    textposition='auto',
                    marker_color='#00CC96',  # Couleur originale verte
                    name=param
                ))

            else:
                fig.add_trace(go.Bar(
                    x=param_df['Modèle'],
                    y=param_df['Valeur'],
                    text=param_df['Valeur'],
                    textposition='auto',
                    marker_color=['#636EFA', '#EF553B', '#00CC96'],
                    name=param
                ))

            # Mise en forme
            fig.update_layout(
                title=f"{param}",
                yaxis_title=f"{param} ({unit})",
                xaxis_title="Modèle",
                template=get_plotly_template(),
                showlegend=False,
                height=400
            )

            # Affichage du graphique
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"**{param}**")
            st.write(f"- GPT-1 : {values['Valeur'][0]}")
            st.write(f"- GPT-2 Small : {values['Valeur'][1]}")
            st.write(f"- GPT-Slim (Notre modèle) : {values['Valeur'][2]}")

    st.write("""
    **Notes Importantes :**
    
    - Les temps d'entraînement sont exprimés en heures sur 1 GPU NVIDIA RTX 4090 (~82.6 TFlops en FP32).
    - Pour GPT-1 et GPT-2 Small, les temps d'entraînement sont ceux qui seraient nécessaires sur une RTX 4090, selon les estimations.
    """)

def logs_entrainement():
    st.title("Mètriques d'entraînement")

    # Chemin vers le fichier Excel (chemin relatif)
    excel_path = os.path.join(CURRENT_DIR, "training data.xlsx")

    # Chargement du fichier Excel
    try:
        df = pd.read_excel(excel_path)

        # Vérification des colonnes nécessaires
        colonnes_requises = ['Shuffle 10%', 'Epoch', 'Training Loss', 'Training Accuracy',
                             'Validation Loss', 'Validation Accuracy',
                             'Training Perplexity', 'Validation Perplexity', 'Learning rate']
        if not all(col in df.columns for col in colonnes_requises):
            st.error("Le fichier Excel ne contient pas toutes les colonnes requises.")
            return

        # Conversion des valeurs 'Learning rate' si nécessaire
        df['Learning rate'] = df['Learning rate'].apply(convert_si_to_float)

        # Création d'un compteur global d'époques
        df = df.sort_values(['Shuffle 10%', 'Epoch']).reset_index(drop=True)
        df['Global Epoch'] = df.index + 1  # Commence à 1

        # Définir 'Global Epoch' comme index et nommer l'index
        df.set_index('Global Epoch', inplace=True)
        df.index.name = 'Global Epoch'

        # Affichage du DataFrame brut avec formatage des colonnes 'Learning rate'
        st.subheader("Tableau des Logs d'Entraînement")

        # Utilisation de pandas Styler pour formater le DataFrame
        styled_df = df.style.format({
            'Learning rate': '{:.7f}'
        })

        st.dataframe(styled_df)

        # Agrégation des données par Global Epoch
        df_agg = df.groupby('Global Epoch').agg({
            'Training Loss': ['mean', 'std'],
            'Validation Loss': ['mean', 'std'],
            'Training Accuracy': ['mean', 'std'],
            'Validation Accuracy': ['mean', 'std'],
            'Training Perplexity': ['mean', 'std'],
            'Validation Perplexity': ['mean', 'std'],
            'Learning rate': ['mean', 'std']
        }).reset_index()

        # Aplatir les colonnes multi-index
        df_agg.columns = ['Global Epoch',
                          'Training Loss Mean', 'Training Loss Std',
                          'Validation Loss Mean', 'Validation Loss Std',
                          'Training Accuracy Mean', 'Training Accuracy Std',
                          'Validation Accuracy Mean', 'Validation Accuracy Std',
                          'Training Perplexity Mean', 'Training Perplexity Std',
                          'Validation Perplexity Mean', 'Validation Perplexity Std',
                          'Learning rate Mean', 'Learning rate Std']

        # Fonction de traçage avec moyenne et écart-type
        def plot_metric(df, epoch_col, mean_col, std_col, title, y_label, log_y=False):
            template = get_plotly_template()
            fig = go.Figure()

            # Tracer la moyenne
            fig.add_trace(go.Scatter(
                x=df[epoch_col],
                y=df[mean_col],
                mode='lines+markers',
                name='Moyenne',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))

            # Tracer la bande d'écart-type
            fig.add_trace(go.Scatter(
                x=df[epoch_col],
                y=df[mean_col] + df[std_col],
                mode='lines',
                name='Écart-Type +',
                line=dict(color='rgba(31, 119, 180, 0.2)'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df[epoch_col],
                y=df[mean_col] - df[std_col],
                mode='lines',
                name='Écart-Type -',
                line=dict(color='rgba(31, 119, 180, 0.2)'),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)',
                showlegend=False
            ))

            # Configuration de l'échelle et du format des ticks
            if log_y:
                fig.update_yaxes(type="log")
            if 'Learning rate' in y_label:
                fig.update_yaxes(
                    tickformat=".1e",          # Notation scientifique
                    exponentformat='e',        # Utiliser 'e' pour l'exposant
                    showexponent='all',        # Afficher l'exposant pour toutes les ticks
                    title_font=dict(size=14)
                )
            else:
                fig.update_yaxes(
                    tickformat=".2f",          # Format par défaut pour les autres métriques
                    title_font=dict(size=14)
                )

            # Mise en forme
            fig.update_layout(
                title=title,
                xaxis_title="Époque Globale",
                yaxis_title=y_label,
                template=template,
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=80, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Création des graphiques pour les métriques
        st.subheader("Évolution des Métriques au Cours des Époques")

        # Liste des métriques disponibles
        metrics = {
            'Loss': ['Training Loss Mean', 'Validation Loss Mean'],
            'Accuracy': ['Training Accuracy Mean', 'Validation Accuracy Mean'],
            'Perplexity': ['Training Perplexity Mean', 'Validation Perplexity Mean'],
            'Learning Rate': ['Learning rate Mean']
        }

        # Traçage pour chaque catégorie de métriques
        for metric_category, metric_names in metrics.items():
            st.subheader(f"Évolution de la {metric_category}")
            for metric in metric_names:
                title = f"{metric} au Fil des Époques"
                y_label = metric.replace(' Mean', '').strip()
                log_scale = True if any(keyword in metric for keyword in ['Loss', 'Perplexity', 'Learning rate']) else False
                plot_metric(
                    df_agg,
                    "Global Epoch",
                    metric,
                    metric.replace(' Mean', ' Std'),
                    title=title,
                    y_label=y_label,
                    log_y=log_scale
                )

    except FileNotFoundError:
        st.error(f"Le fichier Excel n'a pas été trouvé à l'emplacement : {excel_path}")

def inference_modele():
    st.title("Inférence du Modèle : GPT-Slim")
    st.write("""
    Génération de texte à partir d'un prompt.
    Vous pouvez ajuster les paramètres pour influencer la génération du texte.
    """)

    # Charger le tokenizer et le modèle
    tokenizer = load_tokenizer()
    model = load_model()

    # Champ de saisie pour le prompt (limité à 150 mots)
    input_text = st.text_area("Entrez votre prompt (maximum 150 mots) :", height=150)
    input_word_count = len(input_text.split())
    if input_word_count > 150:
        st.warning(f"Votre prompt contient {input_word_count} mots. Veuillez réduire à 150 mots maximum.")
        return

    # Sliders pour ajuster les paramètres
    st.sidebar.subheader("Paramètres de Génération")
    temperature = st.sidebar.slider("Température : Ajuste la créativité du texte généré (0.1 = précis, 2.0 = créatif)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    top_k = st.sidebar.slider("Top-k : Sélectionne parmi les k mots les plus probables (0 = désactivé)", min_value=0, max_value=100, value=15, step=1)
    top_p = st.sidebar.slider("Top-p : Contrôle la diversité des choix de mots (0 = désactivé, 1 = très varié)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    max_length = st.sidebar.slider("Longueur du texte généré (Plus de tokens = plus de temps)", min_value=10, max_value=250, value=100, step=10)

    # Bouton pour lancer la génération
    if st.button("Générer du texte"):
        with st.spinner("Génération du texte en cours..."):
            output_text = generate_tokens(
                model, tokenizer, input_text, num_generated_tokens=max_length,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=1.0, max_words_per_line=25
            )
        st.subheader("Texte Généré")
        st.write(output_text)   

# Dictionnaire des pages
pages = {
    "Introduction": introduction,
    "Schéma du Modèle": schema_modele,
    "Comparaison des Modèles": comparaison_modeles,
    "Logs d'Entraînement": logs_entrainement,
    "Inférence du Modèle": inference_modele
}

# Barre latérale pour la navigation
st.sidebar.title("LLM from scratch")
selection = st.sidebar.radio("Menu", list(pages.keys()))

# Affichage de la page sélectionnée
page = pages[selection]
page()

# Fonction pour charger l'image en base64
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# Chargement de l'image LinkedIn (chemin relatif)
linkedin_icon_path = os.path.join(CURRENT_DIR, "linkedin.png")
try:
    linkedin_icon_base64 = get_image_as_base64(linkedin_icon_path)
except FileNotFoundError:
    linkedin_icon_base64 = ""

# Ajout d'une section pour les contributeurs après le menu de navigation
st.sidebar.title("Contributeurs")

# HTML pour les contributeurs avec l'icône LinkedIn
contributors_html = f"""
<p>
    <a href="https://www.linkedin.com/in/hervehadjadj/" target="_blank">
        <img src="data:image/png;base64,{linkedin_icon_base64}" width="20" style="vertical-align:middle;margin-right:5px;">
        Hervé Hadjadj
    </a>
</p>
<p>
    <a href="https://www.linkedin.com/in/lise-raivard-81a59b76/" target="_blank">
        <img src="data:image/png;base64,{linkedin_icon_base64}" width="20" style="vertical-align:middle;margin-right:5px;">
        Lise Guiot
    </a>
</p>
<p>
    <a href="https://www.linkedin.com/in/florence-josse-94875ab3/" target="_blank">
        <img src="data:image/png;base64,{linkedin_icon_base64}" width="20" style="vertical-align:middle;margin-right:5px;">
        Florence Josse
    </a>
</p>
"""

# Affichage dans la barre latérale
st.sidebar.markdown(contributors_html, unsafe_allow_html=True)
