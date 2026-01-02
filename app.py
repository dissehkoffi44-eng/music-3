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
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION SÉCURISÉE & SECRETS ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 HARMONIC 3", page_icon="🎧", layout="wide")

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .metric-container { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #888; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { 
        padding: 45px; border-radius: 20px; text-align: center; margin: 10px 0; 
        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
    }
    .solid-note-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed #6366F1;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS LOGIQUES ---

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def validate_coherence(chroma_avg, proposed_key):
    """L'oreille interne : compare l'empreinte réelle au profil idéal."""
    try:
        parts = proposed_key.split(" ")
        note_name, mode = parts[0], parts[1].lower()
        idx = NOTES_LIST.index(note_name)
        theoretical_profile = np.roll(PROFILES[mode], idx)
        correlation = np.corrcoef(chroma_avg, theoretical_profile)[0, 1]
        return correlation
    except: return 0

def detect_perfect_cadence(n1, n2):
    try:
        root1, root2 = n1.split()[0], n2.split()[0]
        idx1, idx2 = NOTES_LIST.index(root1), NOTES_LIST.index(root2)
        if (idx1 + 7) % 12 == idx2: return True, n1
        if (idx2 + 7) % 12 == idx1: return True, n2
        return False, n1
    except: return False, n1

def detect_relative_key(n1, n2):
    try:
        c1, c2 = get_camelot_pro(n1), get_camelot_pro(n2)
        if c1 == "??" or c2 == "??": return False, n1
        v1, m1 = int(c1[:-1]), c1[-1]
        v2, m2 = int(c2[:-1]), c2[-1]
        if v1 == v2 and m1 != m2: return True, (n1 if m1 == 'A' else n2)
        return False, n1
    except: return False, n1

def upload_to_telegram(file_buffer, filename, caption, plot_bytes=None):
    try:
        file_buffer.seek(0)
        url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        response = requests.post(url_doc, files=files, data=data, timeout=30).json()
        if plot_bytes and response.get("ok"):
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            requests.post(url_photo, files={'photo': ('graph.png', plot_bytes)}, data={'chat_id': CHAT_ID}, timeout=30)
        return response.get("ok", False)
    except: return False

def get_sine_witness(note_mode_str, key_suffix=""):
    if note_mode_str == "N/A": return ""
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #888;">{note} {mode[:3].upper()} PIANO</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null; let activeNodes = [];
    function playNote(freq, startTime) {{
        const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain();
        osc.type = 'triangle'; osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(0.4, startTime + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.01, startTime + 2.5);
        osc.connect(gain); gain.connect(audioCtx.destination);
        osc.start(startTime); osc.stop(startTime + 2.6);
        return {{osc, gain}};
    }}
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            const isMinor = '{mode}' === 'minor';
            const intervals = isMinor ? [0, 3, 7, 12] : [0, 4, 7, 12];
            const now = audioCtx.currentTime;
            intervals.forEach((interval, index) => {{
                const freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                activeNodes.push(playNote(freq, now + (index * 0.02)));
            }});
            setTimeout(() => {{ this.innerText = '▶'; this.style.background = '#6366F1'; activeNodes = []; }}, 2500);
        }}
    }};
    </script>
    """, height=40)

def analyze_segment_pro(y_seg, sr, tuning):
    chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    rms = np.mean(librosa.feature.rms(y=y_seg))
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score, rms

@st.cache_data(show_spinner="Analyse Harmonique Haute Précision...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Empreinte globale pour l'auto-vérification
    chroma_global = np.mean(librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning), axis=1)
    
    step = 6 
    segments_data = []
    for start_t in range(0, int(duration) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        segments_data.append((y_seg, sr, tuning, start_t))

    timeline_data = []
    votes_weighted = []
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: (analyze_segment_pro(x[0], x[1], x[2]), x[3]), segments_data))

    for (res_key, score, rms), start_t in results:
        weight = int(rms * 100) + 1
        votes_weighted.extend([res_key] * weight)
        timeline_data.append({
            "Temps": start_t, "Note": res_key, 
            "Camelot": get_camelot_pro(res_key), "Confiance": round(float(score)*100, 1)
        })

    if not timeline_data: return None
    df_tl = pd.DataFrame(timeline_data)
    
    counts = Counter(votes_weighted)
    n1 = counts.most_common(1)[0][0]
    n2 = counts.most_common(2)[1][0] if len(counts) > 1 else n1
    
    # --- SYSTÈME D'AUTO-CORRECTION (L'oreille humaine) ---
    score_n1 = validate_coherence(chroma_global, n1)
    score_n2 = validate_coherence(chroma_global, n2)
    note_solide = df_tl['Note'].mode()[0]
    score_solide = validate_coherence(chroma_global, note_solide)

    # Si la note n1 est moins cohérente que la note solide ou n2, on corrige
    if score_solide > score_n1 + 0.1:
        n1 = note_solide
    elif score_n2 > score_n1 + 0.1:
        n1 = n2
        
    is_rel, rel_pref = detect_relative_key(n1, n2)
    if is_rel: n1 = rel_pref
    is_cad, cad_root = detect_perfect_cadence(n1, n2)
    if is_cad: n1 = cad_root

    solid_conf = int(df_tl[df_tl['Note'] == note_solide]['Confiance'].mean())
    final_coherence = validate_coherence(chroma_global, n1)
    musical_score = int(final_coherence * 100)

    # Dynamisation du background selon certitude
    bg = "linear-gradient(135deg, #1D976C, #93F9B9)" if musical_score > 80 else "linear-gradient(135deg, #2193B0, #6DD5ED)"
    if musical_score < 60: bg = "linear-gradient(135deg, #e67e22, #f1c40f)" # Orange si douteux

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    fig_tg = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark")
    fig_tg.update_layout(yaxis={'categoryorder':'array', 'categoryarray':NOTES_ORDER})
    plot_img = fig_tg.to_image(format="png", width=800, height=400)

    res = {
        "file_name": file_name, "tempo": int(float(tempo)),
        "recommended": {"note": n1, "conf": musical_score, "bg": bg},
        "note_solide": note_solide, "solid_conf": solid_conf,
        "timeline": timeline_data, "is_cadence": is_cad, "is_relative": is_rel,
        "energy": int(np.clip(musical_score/10, 1, 10)), "plot_img": plot_img
    }
    del y, y_harm; gc.collect()
    return res

# --- INTERFACE ---
st.title("🎧 RCDJ228 HARMONIC 3")

with st.sidebar:
    st.header("⚙️ SYSTÈME")
    if st.button("🧹 RESET CACHE"):
        st.session_state.processed_files = {}
        st.cache_data.clear()
        st.rerun()

if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []

files = st.file_uploader("📂 AUDIO FILES", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])
tabs = st.tabs(["🚀 ANALYSEUR", "📜 HISTORIQUE"])

with tabs[0]:
    if files:
        for f in reversed(files):
            fid = f"{f.name}_{f.size}"
            if fid not in st.session_state.processed_files:
                f_bytes = f.read()
                res = get_full_analysis(f_bytes, f.name)
                if res:
                    tg_cap = f"🎵 *RAPPORT PRO*\n📄 `{res['file_name']}`\n🎹 `{res['recommended']['note'].upper()}` ({get_camelot_pro(res['recommended']['note'])})\n🎯 Fidélité: {res['recommended']['conf']}%"
                    upload_to_telegram(io.BytesIO(f_bytes), f.name, tg_cap, res["plot_img"])
                    st.session_state.processed_files[fid] = res
                    st.session_state.order_list.insert(0, fid)

        for fid in st.session_state.order_list:
            res = st.session_state.processed_files[fid]
            with st.expander(f"📊 {res['file_name']}", expanded=True):
                st.markdown(f'<div class="final-decision-box" style="background:{res["recommended"]["bg"]};"><h1>{res["recommended"]["note"]}</h1><h2>CAMELOT: {get_camelot_pro(res["recommended"]["note"])} • CERTITUDE: {res["recommended"]["conf"]}%</h2></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="solid-note-box">💎 NOTE STABLE: {res["note_solide"]} ({res["solid_conf"]}% de confiance temporelle)</div>', unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f'<div class="metric-container">BPM<br><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)
                with c2: get_sine_witness(res["recommended"]["note"], fid)
                with c3: st.markdown(f'<div class="metric-container">COHÉRENCE<br><div class="value-custom">{res["recommended"]["conf"]}%</div></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="metric-container">CADENCE<br><div class="value-custom">{"OUI" if res["is_cadence"] else "NON"}</div></div>', unsafe_allow_html=True)
                
                st.plotly_chart(px.line(pd.DataFrame(res['timeline']), x="Temps", y="Note", template="plotly_dark").update_layout(yaxis={'categoryorder':'array', 'categoryarray':NOTES_ORDER}), use_container_width=True)

with tabs[1]:
    if st.session_state.processed_files:
        st.dataframe(pd.DataFrame([{"Fichier": r["file_name"], "Note": r['recommended']['note'], "Camelot": get_camelot_pro(r['recommended']['note']), "BPM": r["tempo"]} for r in st.session_state.processed_files.values()]))

gc.collect()
