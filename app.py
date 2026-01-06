import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests
import gc
from scipy.signal import butter, lfilter

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M3", page_icon="🎧", layout="wide")

# --- CONSTANTES ET PROFILS HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- STYLES CSS PERSONNALISÉS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .metric-container { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; transition: 0.3s; }
    .metric-container:hover { border-color: #6366F1; transform: translateY(-3px); }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { padding: 40px; border-radius: 25px; text-align: center; margin: 15px 0; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .profile-tag { background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin: 2px; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE TRAITEMENT ---

def apply_perceptual_filter(y, sr):
    """Simule la courbe de réponse fréquentielle humaine."""
    nyq = 0.5 * sr
    # On isole la plage 100Hz - 5000Hz (là où l'oreille perçoit les notes)
    low, high = 100 / nyq, 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_enhanced_chroma(y, sr, tuning):
    """Extrait l'empreinte harmonique purifiée."""
    # Séparation Harmonique / Percussive (on jette les batteries)
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    # CQT avec haute résolution (24 bins par octave) pour un meilleur tuning
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24)
    # Masquage souple pour supprimer le bruit de fond spectral
    chroma = librosa.util.softmask(chroma, librosa.segment.recurrence_filter(chroma, mode='affinity'))
    return chroma

def solve_key_logic(chroma_vector):
    """Analyse la corrélation entre les fréquences et les profils musicaux."""
    best_score, best_key, best_root, best_mode = -1, "", 0, "major"
    winners = {}

    # Normalisation du vecteur pour l'oreille
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)

    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                # Calcul de corrélation de Pearson
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                note_str = f"{NOTES_LIST[i]} {mode}"
                
                if score > p_max:
                    p_max, p_note = score, note_str
                
                # Bonus pour Bellman (plus proche de la perception humaine moderne)
                total_score = score * 1.2 if p_name == "bellman" else score
                if total_score > best_score:
                    best_score, best_root, best_mode, best_key = total_score, i, mode, note_str
        winners[p_name] = p_note
        
    return {"key": best_key, "score": best_score, "root": best_root, "mode": best_mode, "details": winners}

def get_camelot(key_str):
    try:
        n, m = key_str.split(" ")
        return BASE_CAMELOT_MINOR.get(n, "??") if m == 'minor' else BASE_CAMELOT_MAJOR.get(n, "??")
    except: return "??"

# --- COMPOSANTS UI ---

def play_chord_button(note_mode, uid):
    if not note_mode or " " not in note_mode: return ""
    n, m = note_mode.split(' ')
    js_id = f"btn_{uid}".replace(".","").replace("#","s")
    return components.html(f"""
    <button id="{js_id}" style="background:#6366F1;color:white;border:none;border-radius:12px;padding:12px;cursor:pointer;font-weight:bold;width:100%;">🔊 TESTER {n} {m.upper()}</button>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    document.getElementById('{js_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const intervals = '{m}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(it => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{n}'] * Math.pow(2, it/12), now);
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.2, now+0.1); g.gain.exponentialRampToValueAtTime(0.01, now+1.5);
            o.connect(g); g.connect(ctx.destination); o.start(now); o.stop(now+1.5);
        }});
    }};
    </script>""", height=60)

# --- ANALYSE PRINCIPALE ---

@st.cache_data(show_spinner=False)
def process_audio(file_bytes, file_name):
    try:
        # Chargement et Tuning
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Filtrage Perceptif
        y_filt = apply_perceptual_filter(y, sr)
        
        # Analyse par segments (pour la timeline et la stabilité)
        step, timeline = 8, []
        votes = Counter()
        
        for start in range(0, int(duration) - step, step):
            y_seg = y_filt[int(start*sr):int((start+step)*sr)]
            if np.max(np.abs(y_seg)) < 0.01: continue # Skip silence
            
            chroma = get_enhanced_chroma(y_seg, sr, tuning)
            res = solve_key_logic(np.mean(chroma, axis=1))
            
            weight = int(res['score'] * 100)
            votes[res['key']] += weight
            timeline.append({"Temps": start, "Note": res['key'], "Conf": round(res['score']*100, 1)})

        if not timeline: return {"error": "Audio trop court ou silencieux"}

        # Décision Finale
        final_key = votes.most_common(1)[0][0]
        avg_conf = int(pd.DataFrame(timeline)[pd.DataFrame(timeline)['Note'] == final_key]['Conf'].mean())
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Graphique Plotly
        df_tl = pd.DataFrame(timeline)
        fig = px.line(df_tl, x="Temps", y="Note", markers=True, category_orders={"Note": NOTES_ORDER}, template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=30,b=0))
        
        # Rapport complet
        full_chroma = get_enhanced_chroma(y, sr, tuning)
        final_details = solve_key_logic(np.mean(full_chroma, axis=1))

        output = {
            "name": file_name, "tempo": int(float(tempo)), "tuning": round(tuning, 2),
            "key": final_key, "camelot": get_camelot(final_key), "conf": avg_conf,
            "details": final_details['details'], "timeline": timeline,
            "plot": fig.to_image(format="png", width=1000, height=450)
        }
        del y, y_filt, df_tl; gc.collect()
        return output
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE UTILISATEUR ---

st.title("🎧 RCDJ228 M3")
uploaded_files = st.file_uploader("📂 Chargez vos fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    pbar = st.progress(0)
    container = st.container()
    
    for i, f in enumerate(uploaded_files):
        res = process_audio(f.read(), f.name)
        
        if "error" in res:
            st.error(f"Erreur sur {f.name}: {res['error']}")
            continue

        with container.expander(f"📊 {res['name']}", expanded=True):
            bg_color = "linear-gradient(135deg, #1e3a8a, #581c87)" if res['conf'] > 70 else "linear-gradient(135deg, #334155, #0f172a)"
            st.markdown(f"""
                <div class="final-decision-box" style="background:{bg_color};">
                    <p style="margin:0; opacity:0.8; letter-spacing:3px;">TONALITÉ DÉTECTÉE</p>
                    <h1 style="font-size:5em; margin:10px 0; font-weight:900;">{res['key']}</h1>
                    <h2 style="margin:0;">CAMELOT: {res['camelot']} | CONFIANCE: {res['conf']}%</h2>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-container">TEMPO<br><span class="value-custom">{res["tempo"]} BPM</span></div>', unsafe_allow_html=True)
            with c2: play_chord_button(res['key'], f.name)
            with c3: 
                st.markdown('<div class="metric-container">PROFILS ANALYSÉS</div>', unsafe_allow_html=True)
                for p, val in res['details'].items():
                    st.markdown(f"<span class='profile-tag'>{p}: {val}</span>", unsafe_allow_html=True)
            
            st.plotly_chart(px.line(pd.DataFrame(res['timeline']), x="Temps", y="Note", markers=True, category_orders={"Note": NOTES_ORDER}, template="plotly_dark"), use_container_width=True)

            # Envoi Telegram
            try:
                cap = f"🎧 *RAPPORT PRO*\n📂 `{res['name']}`\n🎹 *{res['key']}* ({res['camelot']})\n🔥 Confiance: `{res['conf']}%`"
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                              files={'photo': res['plot']}, data={'chat_id': CHAT_ID, 'caption': cap, 'parse_mode': 'Markdown'})
            except: pass

        pbar.progress((i + 1) / len(uploaded_files))

if st.sidebar.button("🧹 Nettoyer le cache"):
    st.cache_data.clear()
    st.rerun()
