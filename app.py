# %cd /content/Qwen3-TTS-Colab

from subtitle import subtitle_maker
from process_text import text_chunk
from qwen_tts import Qwen3TTSModel
import subprocess
import os
import gradio as gr
import numpy as np
import torch
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import snapshot_download
from hf_downloader import download_model
import gc 
import random 
import time
import tempfile
from datetime import datetime
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
  HF_TOKEN=None

# Global model holders
loaded_models = {}
MODEL_SIZES = ["0.6B", "1.7B"]

# --- CONSTANTES Y LISTAS DE OPCIONES ---

SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]

# Lista de idiomas soportados por el modelo para la generación
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", 
    "Spanish", "Portuguese", "Russian"
]

GENDER_OPTIONS = ["None", "Female", "Male"]

AGE_OPTIONS = [
    "None", "Child", "Teenager", "Young Adult", "Adult", 
    "Middle-aged", "Elderly", "Old"
]

# --- NUEVO: MAPEO DE CÓDIGOS A NOMBRES (Expandido) ---
# Se han agregado DA, NO, SV, MS y otros comunes para cubrir la captura de pantalla.
LANG_CODE_TO_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "pt": "Portuguese",
    "ru": "Russian", "it": "Italian", "nl": "Dutch", "tr": "Turkish",
    "pl": "Polish", "cs": "Czech", "ar": "Arabic", "hu": "Hungarian",
    "fi": "Finnish", "vi": "Vietnamese", "uk": "Ukrainian", "el": "Greek",
    "id": "Indonesian", "th": "Thai", "hi": "Hindi", 
    # Agregados para corregir tu captura:
    "da": "Danish", "no": "Norwegian", "sv": "Swedish", "ms": "Malay",
    "bg": "Bulgarian", "ro": "Romanian", "sk": "Slovak", "sl": "Slovenian",
    "hr": "Croatian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
    "he": "Hebrew", "fa": "Persian", "ur": "Urdu", "bn": "Bengali"
}

# --- CONFIGURACIÓN PARA MULTILENGUAJE ---
VOICE_WAV_ROOT = "Voice TTS"
os.makedirs(VOICE_WAV_ROOT, exist_ok=True)

# Estructura: VOICE_DB[Genero][Categoria][Idioma] = { "Nombre": "Ruta Absoluta" }
MULTI_DB = {"Female": {}, "Male": {}, "Other": {}}

def scan_multilang_voices():
    """Escanea la carpeta Voice TTS y organiza por Genero -> Categoria -> Idioma usando rutas absolutas."""
    global MULTI_DB
    MULTI_DB = {"Female": {}, "Male": {}, "Other": {}}
    
    abs_root = os.path.abspath(VOICE_WAV_ROOT)
    
    if not os.path.exists(abs_root): 
        print(f"⚠️ Advertencia: No se encontró la carpeta '{abs_root}'")
        return

    print(f"📂 Escaneando librería de audio en: {abs_root}...")

    for cat_name in sorted(os.listdir(abs_root)):
        cat_path = os.path.join(abs_root, cat_name)
        if not os.path.isdir(cat_path) or cat_name.startswith("."): continue
        
        gender_key = "Other"
        clean_cat_name = cat_name
        
        if "_female" in cat_name.lower():
            gender_key = "Female"
            clean_cat_name = cat_name.replace("_female", "").replace("_", " ").title()
        elif "_male" in cat_name.lower():
            gender_key = "Male"
            clean_cat_name = cat_name.replace("_male", "").replace("_", " ").title()
        else:
            clean_cat_name = cat_name.replace("_", " ").title()
            
        if clean_cat_name not in MULTI_DB[gender_key]:
            MULTI_DB[gender_key][clean_cat_name] = {}

        for lang_code in sorted(os.listdir(cat_path)):
            lang_path = os.path.join(cat_path, lang_code)
            if os.path.isdir(lang_path):
                if lang_code not in MULTI_DB[gender_key][clean_cat_name]:
                    MULTI_DB[gender_key][clean_cat_name][lang_code] = {}
                
                for f in sorted(os.listdir(lang_path)):
                    if f.lower().endswith((".mp3", ".wav")):
                        full_path = os.path.join(lang_path, f)
                        voice_name = os.path.splitext(f)[0]
                        MULTI_DB[gender_key][clean_cat_name][lang_code][voice_name] = full_path

scan_multilang_voices()

# --- DICCIONARIO DE EMOCIONES ---
EMOTION_MAP = {
    "None": "", "Happy": "Speaking with a cheerful and upbeat tone, smiling voice", "Sad": "Speaking with a sorrowful and downcast tone, heavy with emotion", "Angry": "Speaking with a furious and aggressive tone, sharp and intense", "Excited": "Speaking with an enthusiastic and high-energy tone, fast-paced", "Whispering": "Speaking in a hushed and soft whisper, barely audible", "Shouting": "Speaking with a very loud and booming voice, projecting force", "Terrified": "Speaking with a trembling and fearful tone, full of panic", "Crying": "Speaking with a tearful and broken voice, sobbing intermittently", "Laughing": "Speaking while chuckling, with a joyful and amused tone", "Serious": "Speaking with a grave and solemn tone, strictly business", "Sarcastic": "Speaking with a mocking and cynical tone, emphasizing irony", "Sleepy": "Speaking with a drowsy and slow tone, yawning occasionally", "Drunk": "Speaking with a slurred and unsteady voice, disoriented", "Robotic": "Speaking with a mechanical and flat tone, lacking human emotion", "Professional": "Speaking with a polished and formal tone, clear and articulate", "Flirty": "Speaking with a playful and charming tone, slightly breathy", "Disgusted": "Speaking with a repulsed and scornful tone, curling the lip", "Surprised": "Speaking with a shocked and amazed tone, rising pitch", "Nervous": "Speaking with a shaky and hesitant tone, stuttering slightly", "Confident": "Speaking with a bold and assured tone, strong and steady", "Monotone": "Speaking with a flat and unvaried pitch, boring and dull", "Melancholic": "Speaking with a deep and wistful sadness, slow and reflective", "Energetic": "Speaking with a lively and dynamic tone, full of vigor", "Calm": "Speaking with a peaceful and soothing tone, relaxed and steady", "Mysterious (Basic)": "Speaking with an enigmatic and secretive tone", "Panic": "Speaking with a frantic and breathless tone, high urgency", "Seductive": "Speaking with a smooth and alluring tone, low and captivating", "Warm": "Speaking with a friendly and inviting tone, kind and gentle", "Epic Narrator": "Deep and resonant voice, slow-paced rhythm for grandiose moments", "Intimate Narrator": "Close and whispered tone, as if sharing a secret", "Mysterious Narrator": "Low intonation with strategic pauses that generate intrigue", "Documentary Narrator": "Neutral, clear, and authoritative voice that conveys credibility", "Nostalgic Narrator": "Soft tone with slight melancholy that evokes memories of the past", "Energetic Narrator": "Vibrant voice and accelerated rhythm that generates enthusiasm", "Trustworthy Narrator": "Warm and stable tone that inspires security and trust", "Premium Narrator": "Refined and slow-paced voice that suggests luxury and exclusivity", "Conversational Narrator": "Relaxed and natural style, like a recommendation between friends", "Motivational Narrator": "Ascending intonation that conveys inspiration and calls to action", "Playful Narrator": "Animated voice with pitch changes and exaggerated expressiveness", "Gentle Narrator": "Soft and warm tone that generates comfort and affection", "Comic Narrator": "Agile rhythm with funny emphasis that invites laughter", "Protective Narrator": "Enveloping and calm voice that conveys safety", "Whimsical Narrator": "Magical and light intonation, like in a fantasy tale", "ASMR Narrator": "Soft and precise whispers designed for sensory relaxation", "News Anchor Narrator": "Impeccable diction, neutral and objective tone", "Audioguide Narrator": "Clear and moderate voice, easy to follow for long periods", "Meditative Narrator": "Slow rhythm with prolonged pauses that induces deep calm", "Ironic Narrator": "Subtly sarcastic tone with controlled double meaning"
}

# --- Helper Functions ---

def set_seed(seed):
    if seed == -1 or seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def get_model_path(model_type: str, model_size: str) -> str:
    try:
      return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
      return download_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}", download_folder="./qwen_tts_model", redownload= False)

def clear_other_models(keep_key=None):
    global loaded_models
    keys_to_delete = [k for k in loaded_models if k != keep_key]
    for k in keys_to_delete:
        try:
            del loaded_models[k]
        except Exception:
            pass
    for k in keys_to_delete:
        loaded_models.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_type: str, model_size: str):
    global loaded_models
    key = (model_type, model_size)
    if key in loaded_models:
        return loaded_models[key]
      
    clear_other_models(keep_key=key)
    model_path = get_model_path(model_type, model_size)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    loaded_models[key] = model
    return model

def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if x.dtype != np.float32: x = x.astype(np.float32)
    max_val = np.max(np.abs(x))
    if max_val > 0: x = x / (max_val + eps)
    if clip: x = np.clip(x, -1.0, 1.0)
    return x

def _audio_to_tuple(audio):
    """
    Función MAESTRA de carga de audio.
    Usa pydub para normalizar CUALQUIER formato a WAV 16k Mono antes de leerlo.
    Esto soluciona los problemas de ruido/silencio con MP3.
    """
    if audio is None: return None
    
    path_to_load = None
    if isinstance(audio, str): path_to_load = audio
    
    if path_to_load:
        if not os.path.exists(path_to_load):
            print(f"❌ Error: Archivo no encontrado: {path_to_load}")
            return None
        
        try:
            # PASO 1: Conversión forzada con Pydub (MP3/M4A -> WAV 16kHz Mono)
            # Esto elimina problemas de codecs que causan ruido estático.
            audio_segment = AudioSegment.from_file(path_to_load)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Exportar a un buffer temporal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                audio_segment.export(temp_wav.name, format="wav")
                temp_path = temp_wav.name
            
            # PASO 2: Leer el WAV limpio con Librosa
            wav, sr = librosa.load(temp_path, sr=16000, mono=True)
            wav = _normalize_audio(wav)
            
            # Limpieza
            try: os.remove(temp_path)
            except: pass
            
            # Debug
            amp = np.max(np.abs(wav))
            # print(f"🔊 Audio Processed ({os.path.basename(path_to_load)}): Amp={amp:.4f}, SR={sr}")
            
            return wav, int(sr)
            
        except Exception as e:
            print(f"❌ Error crítico procesando audio: {e}")
            return None
    
    # Tupla directa (micrófono)
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
        
    return None

def transcribe_reference(audio_path, mode_input, language="English"):
    if not audio_path: return ""
    print(f"🎙️ Transcribing: {os.path.basename(audio_path)}...")
    src_lang = language if language != "Auto" else "English"
    try:
        results = subtitle_maker(audio_path, src_lang)
        transcript = results[7]
        return transcript if transcript else ""
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

# --- Audio Processing Utils ---
def remove_silence_function(file_path, minimum_silence=100):
    try:
        output_path = file_path.replace(".wav", "_no_silence.wav").replace(".mp3", "_no_silence.wav")
        sound = AudioSegment.from_file(file_path)
        audio_chunks = split_on_silence(sound, min_silence_len=minimum_silence, silence_thresh=-45, keep_silence=50)
        combined = AudioSegment.empty()
        for chunk in audio_chunks: combined += chunk
        combined.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error removing silence: {e}")
        return file_path

def process_audio_output(audio_path, make_subtitle, remove_silence, language="Auto"):
    final_audio_path = audio_path
    if remove_silence:
        final_audio_path = remove_silence_function(audio_path)
    default_srt, custom_srt, word_srt, shorts_srt = None, None, None, None
    if make_subtitle:
        try:
            results = subtitle_maker(final_audio_path, language)
            default_srt = results[0]; custom_srt = results[1]; word_srt = results[2]; shorts_srt = results[3]
        except Exception as e: print(f"Subtitle generation error: {e}")
    return final_audio_path, default_srt, custom_srt, word_srt, shorts_srt

def stitch_chunk_files(chunk_files,output_filename):
    if not chunk_files: return None
    combined_audio = AudioSegment.empty()
    print(f"Stitching {len(chunk_files)} audio files...")
    for f in chunk_files:
        try:
            segment = AudioSegment.from_file(f) 
            combined_audio += segment
        except Exception as e: print(f"Error appending chunk {f}: {e}")
    combined_audio.export(output_filename, format="wav")
    for f in chunk_files:
        try:
            if os.path.exists(f): os.remove(f)
        except Exception as e: pass
    return output_filename

def log_msg(buffer, msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    new_line = f"[{timestamp}] {msg}"
    print(new_line)
    return buffer + new_line + "\n"

# --- Generators ---

def generate_voice_design(text, language, gender, age, emotion_key, manual_desc, seed, remove_silence, make_subs):
    log_buffer = log_msg("", "🚀 SYSTEM INITIATED: Voice Design Mode")
    yield None, log_buffer, None, None, None, None
    if not text or not text.strip(): return
    actual_seed = set_seed(int(seed))
    
    log_buffer = log_msg(log_buffer, f"🎲 Seed Used: {actual_seed}")
    yield None, log_buffer, None, None, None, None

    prompt_parts = []
    if emotion_key and emotion_key != "None": prompt_parts.append(f"Voice: {emotion_key} {EMOTION_MAP.get(emotion_key, '')}")
    if gender and gender != "None": prompt_parts.append(f"Gender: {gender}")
    if age and age != "None": prompt_parts.append(f"Age: {age}")
    auto_prompt = ", ".join(prompt_parts)
    full_prompt = f"{auto_prompt}. {manual_desc}" if (auto_prompt and manual_desc) else (auto_prompt if auto_prompt else manual_desc)
    
    log_buffer = log_msg(log_buffer, f"📝 Prompt: {full_prompt}")
    yield None, log_buffer, None, None, None, None

    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("VoiceDesign", "1.7B")
        
        for i, chunk in enumerate(text_chunks):
            log_buffer = log_msg(log_buffer, f"▶️ Generating Chunk {i+1}/{len(text_chunks)}...")
            yield None, log_buffer, None, None, None, None
            wavs, sr = tts.generate_voice_design(text=chunk.strip(), language=language, instruct=full_prompt.strip(), non_streaming_mode=True, max_new_tokens=2048)
            temp_filename = f"temp_chunk_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            del wavs; torch.cuda.empty_cache(); gc.collect()
        
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        log_buffer = log_msg(log_buffer, "✅ PROCESS COMPLETE.")
        yield final_audio, log_buffer, srt1, srt2, srt3, srt4
    except Exception as e:
        log_buffer = log_msg(log_buffer, f"❌ CRITICAL ERROR: {str(e)}"); yield None, log_buffer, None, None, None, None

def generate_custom_voice(text, language, speaker, emotion_key, manual_desc, seed, model_size, remove_silence, make_subs):
    log_buffer = log_msg("", "🚀 SYSTEM INITIATED: Custom Voice Mode")
    yield None, log_buffer, None, None, None, None
    if not text or not text.strip(): return
    actual_seed = set_seed(int(seed))
    
    log_buffer = log_msg(log_buffer, f"🎲 Seed Used: {actual_seed}")
    yield None, log_buffer, None, None, None, None

    prompt_parts = []
    if emotion_key and emotion_key != "None": prompt_parts.append(f"Voice: {emotion_key} {EMOTION_MAP.get(emotion_key, '')}")
    full_instruct = f"{', '.join(prompt_parts)}. {manual_desc}".strip(' ,.') if (prompt_parts or manual_desc) else None
    
    log_buffer = log_msg(log_buffer, f"🗣️ Speaker ID: {speaker}")
    yield None, log_buffer, None, None, None, None

    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("CustomVoice", model_size)
        
        for i, chunk in enumerate(text_chunks):
            log_buffer = log_msg(log_buffer, f"▶️ Generating Chunk {i+1}/{len(text_chunks)}...")
            yield None, log_buffer, None, None, None, None
            wavs, sr = tts.generate_custom_voice(text=chunk.strip(), language=language, speaker=speaker.lower().replace(" ", "_"), instruct=full_instruct, non_streaming_mode=True, max_new_tokens=2048)
            temp_filename = f"temp_custom_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            del wavs; torch.cuda.empty_cache(); gc.collect()
              
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        log_buffer = log_msg(log_buffer, "✅ PROCESS COMPLETE.")
        yield final_audio, log_buffer, srt1, srt2, srt3, srt4
    except Exception as e:
        log_buffer = log_msg(log_buffer, f"❌ CRITICAL ERROR: {str(e)}"); yield None, log_buffer, None, None, None, None

def smart_generate_clone(ref_audio, ref_text, target_text, language, mode, seed, model_size, remove_silence, make_subs):
    log_buffer = log_msg("", "🚀 CLONING SEQUENCE INITIATED...")
    yield None, log_buffer, None, None, None, None

    if not target_text or not target_text.strip(): return
    if not ref_audio: log_buffer = log_msg(log_buffer, "❌ Error: Reference audio required."); yield None, log_buffer, None, None, None, None; return

    actual_seed = set_seed(int(seed))
    
    log_buffer = log_msg(log_buffer, f"🎲 Seed Used: {actual_seed}")
    yield None, log_buffer, None, None, None, None

    use_xvector_only = ("Fast" in mode)
    final_ref_text = ref_text
    
    # 1. Procesar Audio (Con Blindaje Pydub)
    audio_tuple = _audio_to_tuple(ref_audio)
    if not audio_tuple:
        log_buffer = log_msg(log_buffer, "❌ Error: Could not process reference audio. Check format/path.")
        yield None, log_buffer, None, None, None, None
        return

    # 2. Auto-Transcripción (Si falta texto)
    if not use_xvector_only and (not final_ref_text or not final_ref_text.strip()):
        log_buffer = log_msg(log_buffer, "🎙️ Text missing. Attempting fallback transcription...")
        yield None, log_buffer, None, None, None, None
        try:
            if isinstance(ref_audio, str):
                final_ref_text = transcribe_reference(ref_audio, True, language)
                log_buffer = log_msg(log_buffer, f"📝 Auto-detected: {final_ref_text[:30]}...")
            else:
                log_buffer = log_msg(log_buffer, "⚠️ Cannot transcribe raw audio. Using Fast Mode.")
                use_xvector_only = True
        except Exception as e:
            log_buffer = log_msg(log_buffer, f"❌ Transcribe failed: {e}")

    cleaned_ref_text = final_ref_text.strip() if (final_ref_text and final_ref_text.strip()) else None

    # 3. Generación
    try:
        log_buffer = log_msg(log_buffer, f"⚙️ Loading Model: Base {model_size}...")
        yield None, log_buffer, None, None, None, None
        
        text_chunks, tts_filename = text_chunk(target_text, language, char_limit=280)
        chunk_files = []
        tts = get_model("Base", model_size)
        
        for i, chunk in enumerate(text_chunks):
            log_buffer = log_msg(log_buffer, f"🧬 Cloning Chunk {i+1}/{len(text_chunks)}...")
            yield None, log_buffer, None, None, None, None
            
            wavs, sr = tts.generate_voice_clone(
                text=chunk.strip(),
                language=language,
                ref_audio=audio_tuple,
                ref_text=cleaned_ref_text,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            temp_filename = f"temp_clone_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            del wavs; torch.cuda.empty_cache(); gc.collect()

        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        log_buffer = log_msg(log_buffer, "✅ CLONING COMPLETE.")
        yield final_audio, log_buffer, srt1, srt2, srt3, srt4
    except Exception as e:
        log_buffer = log_msg(log_buffer, f"❌ CRITICAL ERROR: {str(e)}"); yield None, log_buffer, None, None, None, None

# --- GENERADOR MULTILENGUAJE: PUENTE DIRECTO ---
def generate_multilang_preset(target_text, ref_audio, ref_text, language, seed, model_size, remove_silence, make_subs):
    # Debug: Check inputs
    print(f"DEBUG Multilang: Audio={ref_audio}, TextLen={len(str(ref_text))}")
    yield from smart_generate_clone(ref_audio, ref_text, target_text, language, "High-Quality (Audio + Transcript)", seed, model_size, remove_silence, make_subs)


# --- UI Construction ---

def on_mode_change(mode): return gr.update(visible=("High-Quality" in mode))

def build_ui():
    futuristic_css = """
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Orbitron:wght@700&family=Roboto+Mono&display=swap');
    :root { --neon-cyan: #00f3ff; --neon-pink: #ff00ff; --dark-bg: #0a0a0f; --panel-bg: rgba(20, 20, 35, 0.95); --card-bg: #151520; }
    .gradio-container { background-color: var(--dark-bg) !important; background-image: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #000000 100%); font-family: 'Source Sans Pro', sans-serif !important; color: var(--neon-cyan) !important; }
    #main-title h1 { font-family: 'Orbitron', sans-serif !important; text-transform: uppercase; letter-spacing: 3px; text-shadow: 0 0 10px var(--neon-cyan); text-align: center; }
    .dark-panel, .tabs, .tabitem { background-color: var(--panel-bg) !important; border: 1px solid #333; border-radius: 8px !important; }
    textarea, input[type="text"], .gr-dropdown, .gr-input, .gr-checkbox { background-color: rgba(0,0,0,0.6) !important; border: 1px solid #444 !important; color: white !important; font-family: 'Source Sans Pro', sans-serif !important; font-size: 16px !important; }
    .generate-btn { background: linear-gradient(90deg, var(--neon-cyan), #0055ff) !important; border: none !important; color: black !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1px; box-shadow: 0 0 15px var(--neon-cyan); font-family: 'Orbitron', sans-serif !important; }
    .generate-btn:hover { transform: scale(1.02); }
    .log-box textarea { background-color: #050505 !important; border: 1px solid var(--neon-pink) !important; color: var(--neon-pink) !important; font-family: 'Roboto Mono', monospace !important; font-size: 0.9rem !important; }
    .group-header { color: var(--neon-pink); font-weight: bold; margin-bottom: 5px; text-transform: uppercase; font-family: 'Orbitron', sans-serif !important; font-size: 0.9rem; }
    .yt-btn { display: inline-flex; align-items: center; gap: 10px; color: #ff00ff; text-decoration: none; border: 1px solid #ff00ff; padding: 8px 20px; border-radius: 5px; font-family: 'Orbitron', sans-serif; font-weight: bold; transition: all 0.3s; background: rgba(255, 0, 255, 0.05); }
    .yt-btn:hover { background: #ff00ff; color: #000; box-shadow: 0 0 15px #ff00ff; }
    """
    
    theme = gr.themes.Monochrome(primary_hue="cyan", neutral_hue="slate", radius_size=gr.themes.sizes.radius_none)

    with gr.Blocks(theme=theme, css=futuristic_css, title="Qwen3-TTS Ultimate") as demo:
        with gr.Row(elem_id="main-title"):
             gr.HTML("""<div style="text-align: center; padding: 20px; background-color: #202030; border-radius: 10px; margin-bottom: 20px;"><h1 style="font-size: 2.5em; color: #00f3ff; margin: 0;">⚡ Qwen3-TTS // ULTIMATE ⚡</h1><p style="font-size: 1em; color: #E0E0E0; margin: 10px 0; font-family: 'Source Sans Pro';">High-Fidelity Neural Speech Synthesis & Cloning</p><br><a href="https://www.youtube.com/@IA.Sistema.de.Interes" target="_blank" class="yt-btn">IA Sistema de Interés</a></div>""")

        with gr.Tabs(elem_classes="dark-panel"):
            with gr.TabItem("🌍 MULTILENGUAJE"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // LIBRARY SELECTION //", elem_classes="group-header")
                        with gr.Row():
                            ml_gender = gr.Dropdown(label="1. Gender", choices=["Female", "Male", "Other"], value="Female")
                            ml_category = gr.Dropdown(label="2. Category", choices=[], interactive=True)
                        with gr.Row():
                            ml_lang_filter = gr.Dropdown(label="3. Voice Language", choices=[], interactive=True)
                            ml_voice_sel = gr.Dropdown(label="4. Select Voice", choices=[], interactive=True)
                        gr.Markdown("---")
                        gr.Markdown("### // CLONING INPUT //", elem_classes="group-header")
                        
                        ml_ref_audio = gr.Audio(label="Reference Audio (Auto-loaded)", type="filepath", interactive=False)
                        ml_ref_text = gr.Textbox(label="Reference Transcription (Auto-loaded/Generated)", lines=2, interactive=True)
                        
                        ml_target_text = gr.Textbox(label="Target Text", lines=3, placeholder="Escribe aquí el texto a generar...")
                        with gr.Row():
                            ml_gen_lang = gr.Dropdown(label="Generation Language", choices=LANGUAGES, value="Auto")
                            ml_model_size = gr.Dropdown(label="Model", choices=MODEL_SIZES, value="1.7B")
                        ml_btn = gr.Button("► GENERATE FROM PRESET", elem_classes="generate-btn")
                        with gr.Accordion("⚙️ EXTRA SETTINGS", open=False):
                            with gr.Row():
                                ml_seed = gr.Number(label="Seed", value=-1)
                                ml_silence = gr.Checkbox(label="Remove Silence", value=False)
                                ml_subs = gr.Checkbox(label="Make Subs", value=False)
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                         gr.Markdown("### // OUTPUT //", elem_classes="group-header")
                         ml_audio_out = gr.Audio(label="Result", type="filepath")
                         ml_status = gr.Textbox(label="Process Log", lines=10, elem_classes="log-box")
                         with gr.Accordion("📝 SUBTITLES", open=False):
                             with gr.Row():
                                 m_srt1 = gr.File(label="SRT"); m_srt2 = gr.File(label="TXT")
                             with gr.Row():
                                 m_srt3 = gr.File(label="Word"); m_srt4 = gr.File(label="Shorts")

                def update_categories(gender):
                    cats = sorted(list(MULTI_DB.get(gender, {}).keys()))
                    return gr.update(choices=cats, value=cats[0] if cats else None)
                
                # --- FUNCIÓN CORREGIDA Y ROBUSTA ---
                # Esta función ahora devuelve tuplas (Nombre Visible, Valor Interno)
                # El "Valor Interno" es el código de la carpeta (ej: "es") que el sistema necesita.
                def update_langs(gender, cat):
                    if not cat or cat not in MULTI_DB.get(gender, {}): return gr.update(choices=[])
                    
                    # Obtener los códigos de las carpetas (ej: "es", "en")
                    folder_codes = sorted(list(MULTI_DB[gender][cat].keys()))
                    
                    display_choices = []
                    for code in folder_codes:
                        # Buscamos el nombre bonito en el diccionario global
                        # Si no existe, usamos el código en mayúsculas como respaldo
                        display_name = LANG_CODE_TO_NAME.get(code.lower(), code.upper())
                        
                        # CREAMOS LA TUPLA: (Lo que se ve, Lo que se envía al sistema)
                        display_choices.append((display_name, code))
                        
                    # Importante: El valor por defecto debe ser el CÓDIGO (el segundo elemento)
                    first_val = folder_codes[0] if folder_codes else None
                    
                    return gr.update(choices=display_choices, value=first_val)
                # --------------------------------------------------------

                def update_voices(gender, cat, lang):
                    # 'lang' aquí recibirá el código (ej: "es") gracias a la tupla anterior
                    if not cat or not lang: return gr.update(choices=[])
                    try:
                        # Buscamos en MULTI_DB usando el código
                        voices = sorted(list(MULTI_DB[gender][cat][lang].keys()))
                        return gr.update(choices=voices, value=voices[0] if voices else None)
                    except: return gr.update(choices=[])
                
                def load_voice_data_and_transcribe(gender, cat, lang, voice):
                    if not voice: return None, ""
                    try:
                        # 'lang' es el código (ej: "es")
                        audio_path = MULTI_DB.get(gender, {}).get(cat, {}).get(lang, {}).get(voice)
                        if not audio_path or not os.path.exists(audio_path):
                            return None, "Error: Audio file not found in DB."

                        txt_path = os.path.splitext(audio_path)[0] + ".txt"
                        text_content = ""
                        
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    text_content = f.read().strip()
                            except: pass
                        
                        if not text_content:
                            try:
                                text_content = transcribe_reference(audio_path, True, "Auto")
                            except Exception as e:
                                print(f"Warning: Transcribe failed for {voice}: {e}")
                                text_content = "" 
                            
                        return audio_path, text_content
                    except Exception as e:
                        print(f"Error loading voice: {e}")
                        return None, f"Error: {e}"

                ml_gender.change(update_categories, inputs=ml_gender, outputs=ml_category)
                ml_category.change(update_langs, inputs=[ml_gender, ml_category], outputs=ml_lang_filter)
                ml_lang_filter.change(update_voices, inputs=[ml_gender, ml_category, ml_lang_filter], outputs=ml_voice_sel)
                
                ml_voice_sel.change(load_voice_data_and_transcribe, inputs=[ml_gender, ml_category, ml_lang_filter, ml_voice_sel], outputs=[ml_ref_audio, ml_ref_text])
                
                demo.load(lambda: update_categories("Female"), outputs=ml_category).then(
                    lambda g, c: update_langs(g, c), inputs=[ml_gender, ml_category], outputs=ml_lang_filter
                )

                ml_btn.click(
                    generate_multilang_preset,
                    inputs=[ml_target_text, ml_ref_audio, ml_ref_text, ml_gen_lang, ml_seed, ml_model_size, ml_silence, ml_subs],
                    outputs=[ml_audio_out, ml_status, m_srt1, m_srt2, m_srt3, m_srt4]
                )

            # ... Resto de pestañas igual ...
            with gr.TabItem("✨ VOICE DESIGN"):
                # (Código anterior...)
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // INPUT CONFIG //", elem_classes="group-header")
                        design_text = gr.Textbox(label="Text to Synthesize", lines=4, value="It's in the top drawer...", placeholder="Enter text here...")
                        design_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        gr.Markdown("---")
                        gr.Markdown("### // CHARACTER BUILDER //", elem_classes="group-header")
                        with gr.Row():
                            design_gender = gr.Dropdown(label="Gender", choices=GENDER_OPTIONS, value="None", scale=1)
                            design_age = gr.Dropdown(label="Age", choices=AGE_OPTIONS, value="None", scale=1)
                        design_emotion = gr.Dropdown(label="Narrator Style / Emotion", choices=list(EMOTION_MAP.keys()), value="None")
                        design_instruct = gr.Textbox(label="Manual Tweak (Optional)", lines=1, placeholder="e.g. 'breathing heavily'")
                        gr.Markdown("---")
                        design_btn = gr.Button("► GENERATE AUDIO", elem_classes="generate-btn")
                        with gr.Accordion("⚙️ SYSTEM PARAMETERS", open=False):
                            with gr.Row():
                                design_seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                                design_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                                design_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // OUTPUT TERMINAL //", elem_classes="group-header")
                        design_audio_out = gr.Audio(label="Audio Waveform", type="filepath")
                        design_status = gr.Textbox(label="System Log", interactive=False, elem_classes="log-box", lines=12)
                        with gr.Accordion("📝 DATA STREAMS (SUBTITLES)", open=False):
                             with gr.Row():
                                d_srt1 = gr.File(label="SRT (Whisper)"); d_srt2 = gr.File(label="Readable TXT")
                             with gr.Row():
                                d_srt3 = gr.File(label="Word-level"); d_srt4 = gr.File(label="Shorts JSON")
                design_btn.click(generate_voice_design, inputs=[design_text, design_language, design_gender, design_age, design_emotion, design_instruct, design_seed, design_rem_silence, design_make_subs], outputs=[design_audio_out, design_status, d_srt1, d_srt2, d_srt3, d_srt4])

            with gr.TabItem("🧬 VOICE CLONE"):
                # (Código anterior...)
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // TARGET //", elem_classes="group-header")
                        clone_target_text = gr.Textbox(label="Text to Speak", lines=3, placeholder="Enter text...")
                        gr.Markdown("### // REFERENCE SAMPLE //", elem_classes="group-header")
                        clone_ref_audio = gr.Audio(label="Upload Audio Ref", type="filepath")
                        clone_ref_text = gr.Textbox(label="Ref Transcription (Auto)", lines=1)
                        with gr.Row():
                            clone_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                            clone_model_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="1.7B")
                        clone_mode = gr.Dropdown(label="Processing Mode", choices=["High-Quality (Audio + Transcript)", "Fast (Audio Only)"], value="High-Quality (Audio + Transcript)")
                        gr.Markdown("---")
                        clone_btn = gr.Button("► INITIATE CLONING", elem_classes="generate-btn")
                        with gr.Accordion("⚙️ SYSTEM PARAMETERS", open=False):
                             with gr.Row():
                                clone_seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                                clone_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                                clone_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // OUTPUT TERMINAL //", elem_classes="group-header")
                        clone_audio_out = gr.Audio(label="Cloned Waveform", type="filepath")
                        clone_status = gr.Textbox(label="System Log", interactive=False, elem_classes="log-box", lines=12)
                        with gr.Accordion("📝 DATA STREAMS", open=False):
                             with gr.Row():
                                c_srt1 = gr.File(label="Original"); c_srt2 = gr.File(label="Readable")
                             with gr.Row():
                                c_srt3 = gr.File(label="Word-level"); c_srt4 = gr.File(label="Shorts/Reels")
                clone_mode.change(on_mode_change, inputs=[clone_mode], outputs=[clone_ref_text])
                clone_ref_audio.change(transcribe_reference, inputs=[clone_ref_audio, clone_mode, clone_language], outputs=[clone_ref_text])
                clone_btn.click(smart_generate_clone, inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_mode, clone_seed, clone_model_size, clone_rem_silence, clone_make_subs], outputs=[clone_audio_out, clone_status, c_srt1, c_srt2, c_srt3, c_srt4])

            with gr.TabItem("🗣️ PRESET VOICES"):
                # (Código anterior...)
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // INPUT CONFIG //", elem_classes="group-header")
                        tts_text = gr.Textbox(label="Text to Synthesize", lines=4, value="Hello! This is a demo of our TTS capabilities.")
                        with gr.Row():
                            tts_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="English")
                            tts_speaker = gr.Dropdown(label="Speaker ID", choices=SPEAKERS, value="Ryan")
                        gr.Markdown("### // STYLE MODIFIER //", elem_classes="group-header")
                        tts_emotion = gr.Dropdown(label="Narrator Style / Emotion", choices=list(EMOTION_MAP.keys()), value="None")
                        tts_instruct = gr.Textbox(label="Manual Tweak (Optional)", lines=1)
                        tts_model_size = gr.Dropdown(label="Size", choices=MODEL_SIZES, value="1.7B", visible=False)
                        gr.Markdown("---")
                        tts_btn = gr.Button("► GENERATE SPEECH", elem_classes="generate-btn")
                        with gr.Accordion("⚙️ SYSTEM PARAMETERS", open=False):
                             with gr.Row():
                                tts_seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                                tts_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                                tts_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                    with gr.Column(scale=1, elem_classes="dark-panel"):
                        gr.Markdown("### // OUTPUT TERMINAL //", elem_classes="group-header")
                        tts_audio_out = gr.Audio(label="Audio Waveform", type="filepath")
                        tts_status = gr.Textbox(label="System Log", interactive=False, elem_classes="log-box", lines=12)
                        with gr.Accordion("📝 DATA STREAMS", open=False):
                             with gr.Row():
                                t_srt1 = gr.File(label="Original"); t_srt2 = gr.File(label="Readable")
                             with gr.Row():
                                t_srt3 = gr.File(label="Word-level"); t_srt4 = gr.File(label="Shorts/Reels")
                tts_btn.click(generate_custom_voice, inputs=[tts_text, tts_language, tts_speaker, tts_emotion, tts_instruct, tts_seed, tts_model_size, tts_rem_silence, tts_make_subs], outputs=[tts_audio_out, tts_status, t_srt1, t_srt2, t_srt3, t_srt4])
              
    return demo
 
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(share,debug):
    demo = build_ui()
    demo.queue().launch(share=share, debug=debug)

if __name__ == "__main__":
    main()
