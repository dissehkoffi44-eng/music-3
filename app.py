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
st.set_page_config(page_title="RCDJ228 Key Ultimate", page_icon="🎧", layout="wide")

# --- CONSTANTES HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

# Profils Harmoniques
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

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .metric-container { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; transition: 0.3s; }
    .metric-container:hover { border-color: #6366F1; transform: translateY(-3px); }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { padding: 40px; border-radius: 25px; text-align: center; margin: 15px 0; border: 1px solid rgba(255,255,255,0.1); }
    .solid-note-box { background: rgba(99, 102, 241, 0.1); border: 1px dashed #6366F1; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS LOGIQUES ---

def apply_bandpass_filter(y, sr):
    nyq = 0.5 * sr
    low, high = 50 / nyq, 1500 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode == 'minor' else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def solve_key(chroma_avg):
    best_score, best_key, best_root, best_mode = -1, "", 0, "major"
    
    # Pour stocker les gagnants individuels de chaque profil
    profile_winners = {"krumhansl": "", "temperley": "", "bellman": ""}
    scores_k, scores_t, scores_b = {}, {}, {}

    for mode in ["major", "minor"]:
        for i in range(12):
            k = np.corrcoef(chroma_avg, np.roll(PROFILES["krumhansl"][mode], i))[0, 1]
            t = np.corrcoef(chroma_avg, np.roll(PROFILES["temperley"][mode], i))[0, 1]
            b = np.corrcoef(chroma_avg, np.roll(PROFILES["bellman"][mode], i))[0, 1]
            
            note_name = f"{NOTES_LIST[i]} {mode}"
            scores_k[note_name] = k
            scores_t[note_name] = t
            scores_b[note_name] = b
            
            avg_score = (k + t + b) / 3
            if avg_score > best_score:
                best_score, best_root, best_mode, best_key = avg_score, i, mode, note_name

    # Récupérer les notes max par profil
    profile_winners["krumhansl"] = max(scores_k, key=scores_k.get)
    profile_winners["temperley"] = max(scores_t, key=scores_t.get)
    profile_winners["bellman"] = max(scores_b, key=scores_b.get)
    
    return {"key": best_key, "score": best_score, "root": best_root, "mode": best_mode, "details": profile_winners}

def refine_with_harmonic_rules(note_solide_obj, key_fin_obj):
    root_s, mode_s = note_solide_obj['root'], note_solide_obj['mode']
    root_f, mode_f = key_fin_obj['root'], key_fin_obj['mode']
    is_dominante = (root_f + 5) % 12 == root_s
    if is_dominante and mode_f == "major" and mode_s == "minor":
        return note_solide_obj['key'], "Dominante V"
    if root_s == root_f and mode_s == mode_f:
        return key_fin_obj['key'], "Résolue"
    return note_solide_obj['key'], "Stable"

def get_sine_witness(note_mode_str, key_suffix=""):
    if note_mode_str == "N/A": return ""
    parts = note_mode_str.split(' ')
    note, mode = parts[0], parts[1].lower()
    unique_id = f"play_{note}_{mode}_{key_suffix}".replace("#", "s").replace(" ", "")
    return components.html(f"""
    <button id="{unique_id}" style="background:#6366F1;color:white;border:none;border-radius:20px;padding:10px 20px;cursor:pointer;font-weight:bold;width:100%;">▶ TEST {note} {mode[:3].upper()}</button>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    let ctx = null;
    document.getElementById('{unique_id}').onclick = function() {{
        if(!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach((inter, i) => {{
            const osc = ctx.createOscillator(); const g = ctx.createGain();
            osc.type = 'triangle'; osc.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, inter/12), now + (i*0.02));
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.3, now+0.05); g.gain.exponentialRampToValueAtTime(0.01, now+2);
            osc.connect(g); g.connect(ctx.destination); osc.start(now); osc.stop(now+2);
        }});
    }};
    </script>""", height=50)

# --- COEUR DE L'ANALYSE ---

@st.cache_data(show_spinner=False, max_entries=5)
def get_full_analysis(file_bytes, file_name):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        y_harm = librosa.effects.harmonic(y, margin=3.0)
        y_filt = apply_bandpass_filter(y_harm, sr)
        duration = librosa.get_duration(y=y, sr=sr)

        step, timeline = 8, []
        votes = Counter()
        
        for start in range(0, int(duration) - step, step):
            y_seg = y_filt[int(start*sr):int((start+step)*sr)]
            rms = np.mean(librosa.feature.rms(y=y_seg))
            if rms < 0.005: continue 
            
            chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning)
            res_obj = solve_key(np.mean(chroma, axis=1))
            key, score = res_obj['key'], res_obj['score']
            
            weight = int(score * 100) + int(rms * 500)
            votes[key] += weight
            timeline.append({"Temps": start, "Note": key, "Conf": round(score*100, 1)})

        if not timeline: return None
        df_tl = pd.DataFrame(timeline)
        
        note_solide_str = votes.most_common(1)[0][0]
        ns_parts = note_solide_str.split(' ')
        note_solide_obj = {"key": note_solide_str, "root": NOTES_LIST.index(ns_parts[0]), "mode": ns_parts[1].lower()}

        y_end = y_harm[int(max(0, duration-8)*sr):]
        chroma_end = np.mean(librosa.feature.chroma_cens(y=y_end, sr=sr, tuning=tuning), axis=1)
        key_fin_obj = solve_key(chroma_end)

        final_decision, type_res = refine_with_harmonic_rules(note_solide_obj, key_fin_obj)
        is_res = True if type_res != "Stable" else False
        conf_finale = int(df_tl[df_tl['Note'] == final_decision]['Conf'].mean())
        if is_res: conf_finale = min(conf_finale + 5, 100)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bg = "linear-gradient(135deg, #1D976C, #93F9B9)" if conf_finale > 82 else "linear-gradient(135deg, #2193B0, #6DD5ED)"
        
        fig = px.line(df_tl, x="Temps", y="Note", markers=True, title=f"Stabilité: {file_name}", category_orders={"Note": NOTES_ORDER})
        fig.update_traces(line=dict(color="white"), marker=dict(color="white")) 
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', margin=dict(l=60, r=30, t=80, b=60))

        res = {
            "file_name": file_name, 
            "tempo": int(float(tempo)),
            "tuning": round(tuning, 2),
            "rec": {"note": final_decision, "conf": conf_finale, "bg": bg, "type": type_res},
            "note_solide": note_solide_str, 
            "is_res": is_res, 
            "timeline": timeline,
            "profiles_final": key_fin_obj['details'], # On prend les détails de la fin pour le rapport
            "plot_bytes": fig.to_image(format="png", width=1200, height=600, scale=2)
        }
        
        del y, y_harm, y_filt, y_end, df_tl, chroma
        gc.collect()
        return res
    except Exception as e:
        return {"error": str(e), "file_name": file_name}

# --- INTERFACE ---
st.title("🎧 RCDJ228 Key Ultimate PRO (Multi-Profile)")

files = st.file_uploader(f"📂 CHARGER LES FICHIERS", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

if files:
    total = len(files)
    prog_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    for idx, f in reversed(list(enumerate(files))):
        status_text.text(f"Traitement {idx+1}/{total} : {f.name}...")
        fid = f"{f.name}_{f.size}"
        f.seek(0)
        f_bytes = f.read()
        data = get_full_analysis(f_bytes, f.name)
        
        if data and "error" not in data:
            with results_container.expander(f"📊 {data['file_name']}", expanded=True):
                st.markdown(f"""
                    <div class="final-decision-box" style="background:{data['rec']['bg']};">
                        <h1 style="font-size:4.5em; margin:0; font-weight:900;">{data['rec']['note']}</h1>
                        <h2 style="margin:0;">CAMELOT: {get_camelot_pro(data['rec']['note'])} • CERTITUDE: {data['rec']['conf']}%</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f'<div class="metric-container">BPM<br><span class="value-custom">{data["tempo"]}</span></div>', unsafe_allow_html=True)
                with c2: get_sine_witness(data['rec']['note'], fid)
                with c3: st.write("**Détails Profils :**"); st.caption(f"Krum: {data['profiles_final']['krumhansl']} | Temp: {data['profiles_final']['temperley']} | Bell: {data['profiles_final']['bellman']}")

                fig_st = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                st.plotly_chart(fig_st, use_container_width=True)

            # --- ENVOI TELEGRAM AVEC DÉTAILS PROFIL ---
            try:
                main_count = sum(1 for s in data['timeline'] if s['Note'] == data['rec']['note'])
                stability = int((main_count / len(data['timeline'])) * 100) if len(data['timeline']) > 0 else 0
                
                cap = (
                    f"🎧 *RAPPORT HARMONIQUE PRO*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📂 *FICHIER :* `{data['file_name']}`\n"
                    f"🎹 *RÉSULTAT :* *{data['rec']['note']}* ({get_camelot_pro(data['rec']['note'])})\n"
                    f"├─ Certitude : `{data['rec']['conf']}%` | BPM : `{data['tempo']}`\n"
                    f"└─ Stabilité : `{stability}%` | Mode : `{data['rec']['type']}`\n\n"
                    f"🛡 *DÉTAILS DES PROFILS :*\n"
                    f"├─ 🧩 *Krumhansl :* `{data['profiles_final']['krumhansl']}`\n"
                    f"├─ 📐 *Temperley :* `{data['profiles_final']['temperley']}`\n"
                    f"└─ 🔔 *Bellman :* `{data['profiles_final']['bellman']}`\n\n"
                    f"🚀 *Généré par RCDJ228 Key Ultimate*"
                )
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                              files={'photo': data['plot_bytes']}, 
                              data={'chat_id': CHAT_ID, 'caption': cap, 'parse_mode': 'Markdown'})
            except Exception as e:
                st.warning(f"Erreur Telegram : {e}")

        prog_bar.progress((total - idx) / total)
        gc.collect()

    status_text.text(f"✅ Analyse terminée.")

if st.sidebar.button("🧹 VIDER LE CACHE"):
    st.cache_data.clear()
    st.rerun()
