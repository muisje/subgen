from detect_language import detect_most_probable_language
from language_code import LanguageCode
from datetime import datetime
import os
import json
import xml.etree.ElementTree as ET
import threading
import sys
import time
import queue
import logging
import gc
import random
from typing import Union, Any
import numpy as np
import stable_whisper
from stable_whisper import Segment
import requests
import av
import ffmpeg
import whisper
import ast
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import faster_whisper
import io
from stable_whisper.text_output import segment2srtblock, sec2vtt
import traceback
from typing import Any, List
from fastapi import FastAPI, File, UploadFile, Query, Header, Body, Form, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
from dataclasses import dataclass
import signal
import pickle
from unique_queue import UniqueLifoQueue
from requests.exceptions import RequestException
from name_subtitle import SubtitleTagType, FileWriteBehavior, name_subtitle
from config import load_subtitle_tag_config

subgen_version = '2025.01.28'


def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')
    
plextoken = os.getenv('PLEXTOKEN', 'token here')
plexserver = os.getenv('PLEXSERVER', 'http://192.168.1.111:32400')
jellyfintoken = os.getenv('JELLYFINTOKEN', 'token here')
jellyfinserver = os.getenv('JELLYFINSERVER', 'http://192.168.1.111:8096')
whisper_model = os.getenv('WHISPER_MODEL', 'medium')
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 2))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')
procaddedmedia = convert_to_bool(os.getenv('PROCADDEDMEDIA', True))
procmediaonplay = convert_to_bool(os.getenv('PROCMEDIAONPLAY', True))
namesublang = os.getenv('NAMESUBLANG', '')
webhookport = int(os.getenv('WEBHOOKPORT', 9000))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', True))
use_path_mapping = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from = os.getenv('PATH_MAPPING_FROM', r'/tv')
path_mapping_to = os.getenv('PATH_MAPPING_TO', r'/Volumes/TV')
model_location = os.getenv('MODEL_PATH', './models')
monitor = convert_to_bool(os.getenv('MONITOR', False))
transcribe_folders = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe')
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type = os.getenv('COMPUTE_TYPE', 'auto')
append = convert_to_bool(os.getenv('APPEND', False))
reload_script_on_change = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
lrc_for_audio_files = convert_to_bool(os.getenv('LRC_FOR_AUDIO_FILES', True))
custom_regroup = os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1')
use_model_prompt = os.getenv('USE_MODEL_PROMPT', False)
custom_model_prompt = os.getenv('CUSTOM_MODEL_PROMPT', "")
detect_language_length = int(os.getenv('DETECT_LANGUAGE_LENGTH', 30))
detect_language_start_offset = int(os.getenv('DETECT_LANGUAGE_START_OFFSET', int(0)))
skipifexternalsub = convert_to_bool(os.getenv('SKIPIFEXTERNALSUB', False))
skip_if_to_transcribe_sub_already_exist = convert_to_bool(os.getenv('SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST', True))
skipifinternalsublang = LanguageCode.from_iso_639_2(os.getenv('SKIPIFINTERNALSUBLANG', ''))
skip_lang_codes_list = (
    [LanguageCode.from_iso_639_2(code) for code in os.getenv("SKIP_LANG_CODES", "").split("|")]
        if os.getenv('SKIP_LANG_CODES')
    else []
)
force_detected_language_to = LanguageCode.from_iso_639_2(os.getenv('FORCE_DETECTED_LANGUAGE_TO', ''))
preferred_audio_languages = ( 
    [LanguageCode.from_iso_639_2(code) for code in os.getenv('PREFERRED_AUDIO_LANGUAGES', 'eng').split("|")]
    if os.getenv('PREFERRED_AUDIO_LANGUAGES')
    else []
) # in order of preferrence
limit_to_preferred_audio_languages = convert_to_bool(os.getenv('LIMIT_TO_PREFERRED_AUDIO_LANGUAGE', False))
skip_if_audio_track_is_in_list = (
    [LanguageCode.from_iso_639_2(code) for code in os.getenv('SKIP_IF_AUDIO_TRACK_IS', '').split("|")]
    if os.getenv('SKIP_IF_AUDIO_TRACK_IS')
    else []
)
subtitle_language_naming_type = os.getenv('SUBTITLE_LANGUAGE_NAMING_TYPE', 'ISO_639_2_B')
only_skip_if_subgen_subtitle = convert_to_bool(os.getenv('ONLY_SKIP_IF_SUBGEN_SUBTITLE', False))
skip_unknown_language = convert_to_bool(os.getenv('SKIP_UNKNOWN_LANGUAGE', False))
skip_if_language_is_not_set_but_subtitles_exist = convert_to_bool(os.getenv('SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST', False))
skip_if_language_is_not_set_but_subtitles_exist_in_prefered_language = convert_to_bool(os.getenv('SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST_IN_PREFERRED_LANGUAGE', False))
should_whiser_detect_audio_language = convert_to_bool(os.getenv('SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE', False))
ignore_folders = [folder.strip().lower() for folder in os.getenv('IGNORE_FOLDERS', '').split("|")]
ignore_files = [file.strip().lower() for file in os.getenv('IGNORE_FILES', '').split("|")]
should_write_detected_language = convert_to_bool(os.getenv('SHOULD_WRITE_DETECTED_LANGUAGE', False))
jellyfin_stream_subtile = convert_to_bool(os.getenv('JELLYFIN_STREAM_SUBTITLE', True))
word_highlight_color = os.getenv('WORD_HIGHLIGHT_COLOR', 'FFFF99')
skip_if_preferred_audio_language_sub_already_exist = convert_to_bool(os.getenv('SKIP_IF_PREFERRED_AUDIO_LANGUAGE_SUB_ALREADY_EXIST', False))
detect_language_in_filename = convert_to_bool(os.getenv('DETECT_LANGUAGE_IN_FILENAME', False))
assume_no_language_in_subtitle_is_audio_language = convert_to_bool(os.getenv('ASSUME_NO_LANGUAGE_IN_SUBTITLE_IS_AUDIO_LANGUAGE', False))
assume_default_in_subtitle_is_audio_language = convert_to_bool(os.getenv('ASSUME_DEFAULT_IN_SUBTITLE_IS_AUDIO_LANGUAGE', False))
skip_list_file_name = os.getenv('SKIP_LIST_FILE_NAME', None)
should_stream_subtitle = convert_to_bool(os.getenv('SHOULD_STREAM_SUBTITLE', False))
segment_duration = int(os.getenv('SEGMENT_DURATION', 60 * 3))
transcribe_offset_seconds = int(os.getenv('TRANSCRIBE_OFFSET_SECONDS', 0))
use_webhooks = convert_to_bool(os.getenv('USE_WEBHOOKS', True))
shut_down_timeout_seconds = int(os.getenv('SHUT_DOWN_TIMEOUT_SECONDS', 0))
use_task_queue_file = convert_to_bool(os.getenv('USE_TASK_QUEUE_FILE', False))
jellyseer_transcribe_keyword = os.getenv('JELLYSEER_TRANSCRIBE_KEYWORD', None)
jellyseer_translate_keyword = os.getenv('JELLYSEER_TRANSLATE_KEYWORD', None)
subtitle_tags = None
subtitle_tag_delimiter = os.getenv('SUBTITLE_TAG_DELIMITER', '.')

if not subtitle_tag_delimiter or subtitle_tag_delimiter == "":
    subtitle_tag_delimiter = ' '



#TODO validate config on errors

shutdown_event = threading.Event()



def force_exit():
    logging.info("Shutdown timeout reached. Forcing exit.")
    try:
        sys.exit(1)
    except SystemExit as e:
        print(f"SystemExit caught with code: {e.code}")
        raise  # Re-raise to allow the program to terminate properly
    
def immediate_exit():
    logging.info("Forcing immediate exit with os._exit(1).")
    os._exit(1)  # This forces an immediate termination without cleanup

def log_active_threads():
    """Log all active threads except the main thread."""
    active_threads = [t for t in threading.enumerate() if t is not threading.main_thread()]
    if active_threads:
        logging.info(f"Active threads ({len(active_threads)}): {[t.name for t in active_threads]}")
        logging.info(f"Current thread: {threading.current_thread().name}")
    else:
        logging.info("No active threads.")    
    
def are_threads_active():
    """Check if there are any non-daemon threads still running."""
    # Filter out the main thread
    active_threads = [t for t in threading.enumerate() if t is not threading.main_thread()]
    return any(t.is_alive() for t in active_threads)

    
# Function to handle signals
def signal_handler(signum, _):
    #TODO figure out a away how to shutdown a transcription
    if not shutdown_event.is_set():
        logging.info(f"Signal {signum} received. Stopping...")
        shutdown_event.set()
        save_queue()
        
        seconds_waited = 0 
        if shut_down_timeout_seconds > 0:
            while seconds_waited < shut_down_timeout_seconds:
                logging.info(f"Will force shutdown in {shut_down_timeout_seconds - seconds_waited} seconds.")
                if not are_threads_active():
                    logging.info("No threads are active. Forcing shutdown.")
                    force_exit()
                else:
                    logging.info("Threads are still active. Waiting...")
                    log_active_threads()
                time.sleep(1)
                seconds_waited += 1
            force_exit()
        elif shut_down_timeout_seconds == 0:
            logging.info("shut_down_timeout_seconds is 0. Forcing shutdown without waiting.")
            force_exit()
        else:
            logging.info("shut_down_timeout_seconds is less than 0. Will not shutdown on timeout")
    else:
        logging.info("Received a signal again. Already shutting down. Now forcing exit.")
        immediate_exit()

def load_skip_list(filename):
    if filename is None or filename == "":
        return []
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file.readlines()]
    return []

def write_to_skip_list(filename, skipped_file):
    if not filename or filename == "":
        return
    with open(filename, 'a') as file:
        file.write(f"{skipped_file}\n")

files_to_skip_list = load_skip_list(skip_list_file_name)


try:
    kwargs = ast.literal_eval(os.getenv('SUBGEN_KWARGS', '{}') or '{}')
except ValueError:
    kwargs = {}
    logging.info("kwargs (SUBGEN_KWARGS) is an invalid dictionary, defaulting to empty '{}'")
    
if transcribe_device == "gpu":
    transcribe_device = "cuda"
        

VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", 
    ".3gp", ".ogv", ".vob", ".rm", ".rmvb", ".ts", ".m4v", ".f4v", ".svq3", 
    ".asf", ".m2ts", ".divx", ".xvid"
)

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma", ".alac", ".m4a", ".opus", 
    ".aiff", ".aif", ".pcm", ".ra", ".ram", ".mid", ".midi", ".ape", ".wv", 
    ".amr", ".vox", ".tak", ".spx", '.m4b'
)


app = FastAPI()
model = None

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"
last_print_time = None

task_queue = UniqueLifoQueue() #LIFO queue to prioritize new tasks, should probably use a priority queue. So that when a language task finished it will be transcribed next.

TASK_QUE_FILE = "task_queue.pkl"

finished_processing_paths_event = threading.Event()

task_queue_lock = threading.Lock()

# Load the queue data from file if it exists
def load_queue():
    if use_task_queue_file:
        if os.path.exists(TASK_QUE_FILE):
            # Check if the file is empty before trying to load
            if os.path.getsize(TASK_QUE_FILE) > 0:
                try:
                    with open(TASK_QUE_FILE, "rb") as f:
                        data = pickle.load(f)
                        # Ensure data is iterable before trying to put items in the queue
                        if isinstance(data, list):  
                            for item in data:
                                task_queue.put(item)
                        else:
                            logging.warning(f"Unexpected data type in file: {type(data)}")
                    logging.info("Loaded task queue from file.")
                except EOFError:
                    logging.error("Error: The task queue file is empty or corrupted.")
                except Exception as e:
                    logging.error(f"Error loading task queue: {e}")
            else:
                logging.warning(f"The task queue file {TASK_QUE_FILE} is empty.")
        else:
            logging.info("No existing task queue file found. Starting fresh.")
        
# Save the queue data to a file
def save_queue():
    if use_task_queue_file:
        data = []
        while not task_queue.empty():
            data.append(task_queue.get())  # Retrieve all items
        with open(TASK_QUE_FILE, "wb") as f:
            pickle.dump(data, f)
        logging.info("Task queue saved to file.")

def transcription_worker():
    logging.debug(f"Starting transcription worker {threading.current_thread().name}")
    should_work = True
    stop_working_when_no_tasks = not (monitor or use_webhooks)
    
    list_task_queue = True
    while should_work:
        if shutdown_event.is_set():
            break
        
        # Check if the queue is empty without a lock
        if not task_queue.empty():
            with task_queue_lock: # needs lock because we are directly accesing the queue
                if list_task_queue:
                    logging.debug("Task queue length: %d", task_queue.qsize())
                    logging.info("The tasks in the queue are:")
                    tasks = list(task_queue.queue)  # Convert the queue to a list
                    total = len(tasks)
                    for num, task in zip(range(total, 0, -1), tasks):
                        task_type = task.get('type', task.get('transcribe_or_translate', '???'))
                        logging.info(f"[{num}]  [{task_type}]  {os.path.basename(task.get('path', 'No path'))}")
        
        try:
            # Try to get a task without blocking
            task = task_queue.get_nowait()
            logging.info(f"[{threading.current_thread().name}] Processing task: {task}")
            
            if 'Bazarr-' in task['path']:
                logging.info(f"Task {task['path']} is being handled by ASR.")
            if "type" in task and task["type"] == "detect_language":
                detect_language_task(task['path'], task.get('skip_skip_check', False))
                task_queue.task_done()  # Mark task as done if processing succeeded
            else:
                gen_subtitles(task['path'], task['transcribe_or_translate'], task['force_language']) #TODO on translate subtitle should be in the english
                task_queue.task_done()
                
        except queue.Empty:
            # If the queue is empty, check if we should stop
            if stop_working_when_no_tasks:
                if finished_processing_paths_event.is_set():
                    should_work = False
                    logging.info(f"No more tasks in the queue, stopping transcription {threading.current_thread().name} worker...")
                    continue
            time.sleep(1)  # Wait for a sec before checking again
                
        except Exception as e:
            logging.error(f"Error processing task: {e}")

    logging.info(f"Transcription worker {threading.current_thread().name} stopped.")

def start_transcription_workers():
    threads = []
    for i in range(concurrent_transcriptions):
        thread_name = f"transcription_worker[{i}]"
        thread = threading.Thread(target=transcription_worker, daemon=True, name=thread_name)
        thread.start()
        threads.append(thread)
    return threads
    


# Define a filter class to hide common logging we don't want to see
class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        # Define the patterns to search for
        patterns = [
            "Compression ratio threshold is not met",
            "Processing segment at",
            "Log probability threshold is",
            "Reset prompt",
            "Attempting to release",
            "released on ",
            "Attempting to acquire",
            "acquired on",
            "header parsing failed",
            "timescale not set",
            "misdetection possible",
            "srt was added",
            "doesn't have any audio to transcribe",
            "Calling on_"
        ]
        # Return False if any of the patterns are found, True otherwise
        return not any(pattern in record.getMessage() for pattern in patterns)

# Configure logging
if debug:
    level = logging.DEBUG
    logging.basicConfig(stream=sys.stderr, level=level, format="%(asctime)s %(levelname)s: %(message)s")
else:
    level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(level)  # Set the logger level

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

#This forces a flush to print progress correctly
def progress(seek, total):
    sys.stdout.flush()
    sys.stderr.flush()
    
    
    if(docker_status) == 'Docker':
        global last_print_time
        # Get the current time
        current_time = time.time()
    
        # Check if 5 seconds have passed since the last print
        if last_print_time is None or (current_time - last_print_time) >= 5:
            # Update the last print time
            last_print_time = current_time
            # Log the message
            logging.info("Force Update...")

TIME_OFFSET = 5

def appendLine(result):
    if append:
        lastSegment = result.segments[-1]
        date_time_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        appended_text = f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {date_time_str}"
        
        # Create a new segment with the updated information
        newSegment = Segment(
            start=lastSegment.start + TIME_OFFSET,
            end=lastSegment.end + TIME_OFFSET,
            text=appended_text,
            words=[],  # Empty list for words
            id=lastSegment.id + 1
        )
        
        # Append the new segment to the result's segments
        result.segments.append(newSegment)

@app.get("/plex")
@app.get("/webhook")
@app.get("/jellyfin")
@app.get("/asr")
@app.get("/emby")
@app.get("/detect-language")
@app.get("/tautulli")
@app.get("/jellyseerr")
def handle_get_request(request: Request):
    logging.warning(f"Invalid request at: {request.url} via GET method.")
    return {"You accessed this request incorrectly via a GET request.  See https://github.com/McCloudS/subgen for proper configuration"}

@app.get("/")
def webui():
    return {"The webui for configuration was removed on 1 October 2024, please configure via environment variables or in your Docker settings."}

@app.get("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    logging.info("Received shutdown web request, Shutting down...")
    return {"message": "Shutting down..."}

@app.get("/status")
def status():
    return {"version" : f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"}

@app.post("/tautulli")
def receive_tautulli_webhook(
        source: Union[str, None] = Header(None),
        event: str = Body(None),
        file: str = Body(None),
):
    if source == "Tautulli":
        logging.debug(f"Tautulli event detected is: {event}")
        if((event == "added" and procaddedmedia) or (event == "played" and procmediaonplay)):
            fullpath = file
            logging.debug("Path of file: " + fullpath)

            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
    else:
        return {
            "message": "This doesn't appear to be a properly configured Tautulli webhook, please review the instructions again!"}

    return ""


@app.post("/plex")
def receive_plex_webhook(
        user_agent: Union[str] = Header(None),
        payload: Union[str] = Form(),
):
    try:
        plex_json = json.loads(payload)
        logging.debug(f"Raw response: {payload}")

        if "PlexMediaServer" not in user_agent:
            return {"message": "This doesn't appear to be a properly configured Plex webhook, please review the instructions again"}

        event = plex_json["event"]
        logging.debug(f"Plex event detected is: {event}")

        if (event == "library.new" and procaddedmedia) or (event == "media.play" and procmediaonplay):
            fullpath = get_plex_file_name(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
            logging.debug("Path of file: " + fullpath)

            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
            refresh_plex_metadata(plex_json['Metadata']['ratingKey'], plexserver, plextoken)
            logging.info(f"Metadata for item {plex_json['Metadata']['ratingKey']} refreshed successfully.")
    except Exception as e:
        logging.error(f"Failed to process Plex webhook: {e}")

    return ""

#perhaps expand this with the filenames/ titles and if external or internal and the indexes
#also have something like a source like jellyseerr maybe so it can notify back that it has succedeed 
@dataclass
class MediaInfo:
    path: str
    audio_languages: List[LanguageCode] 
    subtitle_languages: List[LanguageCode] 

# Define the payload model
class JellyseerWebhookPayload(BaseModel):
    notification_type: str
    subject: str
    message: str
    media: Optional[Dict[str, Any]] = None
    request: Optional[Dict[str, Any]] = None
    issue: Optional[Dict[str, Any]] = None
    comment: Optional[Dict[str, Any]] = None
    extra: Optional[list] = []

@app.post("/jellyseerr")
async def receive_jellyseerr_webhook(payload: Optional[JellyseerWebhookPayload] = None):
    # If payload is empty or None (i.e., test notification), return a test connection response
    if payload.notification_type == "TEST_NOTIFICATION":
        logging.info("Test notification of jellyseerr received!")
        return {"status": "ok", "message": "Test connection successful."}
    
    
    # If there's a valid payload, process it
    notification_type = payload.notification_type
    subject = payload.subject
    message = payload.message
    media_type = payload.media.get("media_type") if payload.media else None
    media_status = payload.media.get("status") if payload.media else None
    tmdb_id = payload.media.get("tmdbId") if payload.media else None
    tmdb_id = None if tmdb_id == "" else tmdb_id
    tvdb_id = payload.media.get("tvdbId") if payload.media else None
    tvdb_id = None if tvdb_id == "" else tvdb_id
    issue_id = payload.issue.get("issue_id") if payload.issue else None
    issue_type = payload.issue.get("issue_type") if payload.issue else None
    issue_status = payload.issue.get("issue_status") if payload.issue else None
    reported_by_username = payload.issue.get("reportedBy_username") if payload.issue else None
    
    
    if not notification_type == "ISSUE_CREATED" or not issue_type == "SUBTITLES":
        error_message = f"Invalid notification type or issue type: notification_type={notification_type}, issue_type={issue_type}. Expected ISSUE_CREATED and SUBTITLES."
        logging.warning(error_message)
        # Raise an HTTPException with 400 status code
        raise HTTPException(
            status_code=400,  # Bad Request
            detail=error_message
        )
        
    if not (media_status == "AVAILABLE" or media_status == "PARTIALLY_AVAILABLE"):
        error_message = f"Invalid media_status: media_status={media_status}. Expected AVAILABLE or PARTIALLY_AVAILABLE"
        logging.warning(error_message)
        # Raise an HTTPException with 400 status code
        raise HTTPException(
            status_code=400,  # Bad Request
            detail=error_message
        )
        
    should_transcribe_or_translate = transcribe_or_translate
        
    if jellyseer_transcribe_keyword and jellyseer_transcribe_keyword in message:
        should_transcribe_or_translate = "transcribe"
    elif jellyseer_translate_keyword and jellyseer_translate_keyword in message:
        should_transcribe_or_translate = "translate"
    # should_transcribe_or_translate = "translate"
            

    # Should get is avalable too

    # Log the information (or replace with your logic)
    logging.debug(f"Notification Type: {notification_type}")
    logging.debug(f"Subject: {subject}")
    logging.debug(f"Message: {message}")
    logging.debug(f"Media status: {media_status}")
    logging.debug(f"Media Type: {media_type}, TMDB ID: {tmdb_id}, TVDB ID: {tvdb_id}")
    logging.debug(f"Issue ID: {issue_id}, Type: {issue_type}, Status: {issue_status}")
    logging.debug(f"Reported by: {reported_by_username}")
    logging.debug(f"Extra: {payload.extra}")
    
    
    is_movie = media_type == "movie"
    is_series = media_type == "tv"

   
    #TODO handle the case where there are multiple versions of the same movie or episode (when one episode is requested)
    
    if is_movie:
        movie = get_movie_from_jellyfin(subject, tmdb_id, jellyfinserver, jellyfintoken)
        if movie:
            logging.info(f"Found movie: {movie}")
            force_language = LanguageCode.NONE
            if len(movie.audio_languages) == 1:
                force_language = movie.audio_languages[0]
            else:
                force_language = next((language for language in movie.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
            gen_subtitles_queue(path_mapping(movie.path), transcribe_or_translate, force_language, True)
        else:
            logging.warning(f"Did not find movie for {subject}")
        return
    elif is_series:
        if len(payload.extra) > 0: # Empty if all seasons for tv
            season_nr = next((int(item['value']) for item in payload.extra if item['name'] == 'Affected Season'), None) 
            episode_nr = next((int(item['value']) for item in payload.extra if item['name'] == 'Affected Episode'), None)   # None if all episodes for season
             
            # if series check if one episode or one season or all seasons
            
            if season_nr and episode_nr:
                # Single episode
                episode = get_series_episode_path_from_jellyfin(subject, tmdb_id, tvdb_id, season_nr, episode_nr, jellyfinserver, jellyfintoken)
                if episode:
                    logging.info(f"Found episode: {episode}")
                    force_language = LanguageCode.NONE
                    if len(episode.audio_languages) == 1:
                        force_language = episode.audio_languages[0]
                    else:
                        force_language = next((language for language in episode.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
                    gen_subtitles_queue(path_mapping(episode.path), should_transcribe_or_translate, force_language, True)
                else:
                    logging.warning(f"Did not find episode {episode_nr} of season {season_nr} for {subject}")
            
            elif season_nr:
                # Full season
                episodes = get_series_season_episodes_paths_from_jellyfin(subject, season_nr, tmdb_id, tvdb_id, jellyfinserver, jellyfintoken)
                if episodes:
                    logging.info(f"Found {len(episodes)} episodes for season {season_nr} of {subject}")
                    for episode in episodes:
                        logging.info(f"Handling episode: {episode}")
                        force_language = LanguageCode.NONE
                        if len(episode.audio_languages) == 1:
                            force_language = episode.audio_languages[0]
                        else:
                            force_language = next((language for language in episode.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
                        gen_subtitles_queue(path_mapping(episode.path), should_transcribe_or_translate, force_language, True)
                else:
                    logging.warning(f"Did not find any episodes for season {season_nr} of {subject}")
            else:
                logging.warning(f"Expected at least Affected Season in payload extra when extra is not empty. Instead got this: {payload.extra}")
        
        else:
            # Full series
            episodes = get_series_episodes_paths_from_jellyfin(subject, tmdb_id, tvdb_id, jellyfinserver, jellyfintoken)
            if episodes:
                logging.info(f"Found {len(episodes)} episodes for {subject}")
                for episode in episodes:
                    logging.info(f"Handling episode: {episode}")
                    force_language = LanguageCode.NONE
                    if len(episode.audio_languages) == 1:
                        force_language = episode.audio_languages[0]
                    else:
                        force_language = next((language for language in episode.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
                    gen_subtitles_queue(path_mapping(episode.path), should_transcribe_or_translate, force_language, True)
            else:
                logging.warning(f"Did not find any episodes for {subject}")
                

    
    # can get this from jellyfin api
    
    # Other usefull info maybe
    # "MovieCount": 0,
    # "SeriesCount": 0,
    # "ProgramCount": 0,
    # "EpisodeCount": 0,
    # "EpisodeTitle": "string",    

    logging.info(f"Finished processing jellyseerr webhook for {subject}")

    return {"status": "success", "message": "Webhook processed successfully."}




def _extract_media_info_from_source(source: dict) -> Optional[MediaInfo]:
    """Helper function to extract MediaInfo from a single media source."""
    if 'Path' not in source:
        return None
        
    path = source['Path']
    media_streams = source.get("MediaStreams", [])
    
    audio_langs = [
        LanguageCode.from_iso_639_2(stream.get("Language"))
        for stream in media_streams if stream.get("Type") == "Audio"
    ]
    
    subtitle_langs = [
        LanguageCode.from_iso_639_2(stream.get("Language"))
        for stream in media_streams if stream.get("Type") == "Subtitle"
    ]
    
    return MediaInfo(
        path=path,
        audio_languages=audio_langs,
        subtitle_languages=subtitle_langs
    )

def _extract_media_info_from_item(item: dict) -> Optional[MediaInfo]:
    """Helper function to extract MediaInfo from an item's media sources."""
    media_sources = item.get("MediaSources", [])
    
    # Try each media source
    for source in media_sources:
        media_info = _extract_media_info_from_source(source)
        if media_info:
            return media_info
    
    # If no media sources worked, try direct path
    path = item.get("Path")
    if path:
        return MediaInfo(
            path=path,
            audio_languages=[],  # No language info available
            subtitle_languages=[]
        )
    
    return None

def _base_jellyfin_search(search_term: str, include_item_type: str = "Episode", jellyfinserver: str = "http://localhost:8096", 
                         jellyfintoken: str = "your_token_here", limit: int = 1000) -> Optional[List[dict]]:
    """Base function for Jellyfin API searches that handles common setup and error handling."""
    base_url = f"{jellyfinserver}/Items"
    
    params = {
        "limit": limit,
        "searchterm": search_term,
        "fields": ["Name", "ProviderIds", "MediaSources", "Path"],
        "IncludeItemTypes": [include_item_type],
        "Filters": ["IsNotFolder"] if include_item_type in ["Episode", "Movie"] else ["IsFolder"],
        "isMissing": False,
        "enableImages": False
    }
    
    #interesting fields: MediaSourceCount, ChildCount, OriginalTitle, ParentId, SeriesPresentationUniqueKey, ExternalSeriesId, ItemCounts, 
    # For helping whisper with context: Taglines, Tags, People
    
    
    if include_item_type == "Movie":
        params["isMovie"] = True
        params["hasTmdbId"] = True

    headers = {
        "X-Emby-Token": jellyfintoken
    }

    # Debug logging
    for key, value in params.items():
        logging.debug(f"{key: <15}: {value}")

    response = None
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an exception for HTTP errors
    except requests.exceptions.Timeout: # waited more than 10 sec
        logging.warning("The request timed out. Please try again later.")
        return None
    except RequestException as e:
        logging.warning(f"Request error: {e}")
        return None
    except OSError as e:
        if e.errno == 113:  # Check for "No route to host" error
            logging.warning("No route to host, unable to connect.")
            return None
    except Exception as e:  # Catch any other unexpected errors
        logging.error(f"Unexpected error occurred: {e}")
        return None
    
    if not response:
        logging.warning("No response from Jellyfin at {base_url} with params: {params} and headers: {headers}")
        return None
        
    if response.status_code != 200:
        logging.warning(f"Error: {response.status_code}, {response.text}")
        return None
        
    data = response.json()
    items = data.get("Items", [])
    
    if len(items) < 1:
        logging.warning("No results for search from jellyfin.")
        return None
        
    return items



#Used to compare string from jellyfin and jellyseerr
def compare_strings_normalized(s1, s2):
    # Normalize both strings to Unicode NFC form
    s1_normalized = unicodedata.normalize("NFKC", s1)
    s2_normalized = unicodedata.normalize("NFKC", s2)
    
    return s1_normalized == s2_normalized

def get_movie_from_jellyfin(movie_name: str, tmdb_id: str, 
                           jellyfinserver: str = "http://localhost:8096", 
                           jellyfintoken: str = "your_token_here") -> Optional[MediaInfo]:
    """Get movie MediaInfo from Jellyfin."""
    items = _base_jellyfin_search(movie_name, include_item_type="Movie", jellyfinserver=jellyfinserver, 
                                 jellyfintoken=jellyfintoken, limit=5)
    if not items:
        return None

    for item in items:
        item_name = item.get('Name', 'Movie without a name')
        provider_ids = item.get('ProviderIds', {})
        item_tmdb_id = provider_ids.get("Tmdb")

        # Check TMDB ID match
        if tmdb_id and item_tmdb_id:
            if item_tmdb_id != tmdb_id:
                logging.info(f"tmdbid does not match {tmdb_id} == {item_tmdb_id}")
                continue
        else:
            if  not compare_strings_normalized(item_name, strip_year_from_title(movie_name)):
                # logging.info(f"Name does not match {item_name} == {strip_year_from_title(movie_name)}")
                continue

        media_info = _extract_media_info_from_item(item)
        if media_info:
            return media_info
        
        logging.warning(f"Item {item_name} does not have valid media info. {item}")
    
    return None

def get_series_episodes_paths_from_jellyfin(series_name: str, tmdb_id: str, tvdb_id: str,
                                          jellyfinserver: str = "http://localhost:8096", 
                                          jellyfintoken: str = "your_token_here") -> List[MediaInfo]:
    """Get all episodes from Jellyfin for a series."""
    series_id = get_series_id_from_jellyfin(series_name, tmdb_id, tvdb_id, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not series_id:
        logging.warning(f"Series {series_name} not found in Jellyfin.")
        return []
    
    items = _base_jellyfin_search(series_name, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not items:
        return []
        
    results = []
    for item in items:
        item_series_id = item.get("SeriesId", None)
        
        if not item_series_id == series_id:
            continue


        media_info = _extract_media_info_from_item(item)
        if media_info:
            results.append(media_info)
        else:
            logging.warning(f"Item {item.get('Name')} does not have valid media info. {item}")
            
    return results

def get_series_id_from_jellyfin(series_name: str, tmdb_id: str, tvdb_id: str,
                                jellyfinserver: str = "http://localhost:8096", 
                                jellyfintoken: str = "your_token_here") -> Optional[str]:
    """Get series ID from Jellyfin."""
    
    
    items = _base_jellyfin_search(series_name, include_item_type="Series", jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken, limit=10)
    if not items:
        return None

    for item in items:
        item_series_name = item.get('SeriesName', 'Series without a name')
        provider_ids = item.get('ProviderIds', {})
        item_tmdb_id = provider_ids.get("Tmdb")
        item_tvdb_id = provider_ids.get("Tvdb")

        # Check TMDB ID match
        if tmdb_id and item_tmdb_id:
            if item_tmdb_id == tmdb_id:
                series_id = item.get("Id")
                if series_id:
                    return series_id
                else:
                    logging.warning(f"Item {item_series_name} does not have valid ID.")
        elif tvdb_id and item_tvdb_id:
            if item_tmdb_id == tmdb_id:
                series_id = item.get("Id")
                if series_id:
                    return series_id
                else:
                    logging.warning(f"Item {item_series_name} does not have valid ID.")
        
        elif not compare_strings_normalized(strip_year_from_title(item_series_name), strip_year_from_title(series_name)):
            # if series name doesn't match, skip
            logging.info(f"Series name does not match {item_series_name} == {strip_year_from_title(series_name)}")
            continue

        return item.get("Id")
        
    return None

def get_series_season_episodes_paths_from_jellyfin(series_name: str, season_nr: int, tmdb_id: str, tvdb_id: str,
                                                 jellyfinserver: str = "http://localhost:8096", 
                                                 jellyfintoken: str = "your_token_here") -> List[MediaInfo]:
    """Get all episode MediaInfo objects for a specific season of a series."""
    series_id = get_series_id_from_jellyfin(series_name, tmdb_id, tvdb_id, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not series_id:
        logging.warning(f"Series {series_name} not found in Jellyfin.")
        return []
    
    items = _base_jellyfin_search(series_name, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not items:
        return []
        
    results = []
    for item in items:
        item_season_nr = item.get("ParentIndexNumber", None)
        item_series_id = item.get("SeriesId", None)
        
        
        if not item_series_id == series_id:
            continue
        
            
        if not (item_season_nr == season_nr):
            continue

        media_info = _extract_media_info_from_item(item)
        if media_info:
            results.append(media_info)
        else:
            logging.warning(f"Item {item.get('Name')} does not have valid media info. {item}")
            
    return results

def get_series_episode_path_from_jellyfin(series_name: str,  tmdb_id: str, tvdb_id: str, season_nr: int, episode_nr: int,
                                        jellyfinserver: str = "http://localhost:8096", 
                                        jellyfintoken: str = "your_token_here") -> Optional[MediaInfo]:
    """Get the MediaInfo for a specific episode of a series."""
    series_id = get_series_id_from_jellyfin(series_name, tmdb_id, tvdb_id, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not series_id:
        logging.warning(f"Series {series_name} not found in Jellyfin.")
        return []
    
    items = _base_jellyfin_search(series_name, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not items:
        return None
        
    for item in items:
        item_episode_nr = item.get("IndexNumber")
        item_season_nr = item.get("ParentIndexNumber")
        item_series_id = item.get("SeriesId", None)
        
        if not item_series_id == series_id:
            continue
        

        if not (item_episode_nr == episode_nr and item_season_nr == season_nr):
            continue

        media_info = _extract_media_info_from_item(item)
        if media_info:
            return media_info
            
        logging.warning(f"Item {item.get('Name')} does not have valid media info. {item}")
    
    return None


def strip_year_from_title(title):
    # Find the position of the last " (" and remove the substring from there
    if title.endswith(")") and title[-5:-1].isdigit():
        open_paren_pos = title.rfind(" (")
        if open_paren_pos != -1:
            return title[:open_paren_pos]
    return title


@app.post("/jellyfin")
async def receive_jellyfin_webhook(
        user_agent: str = Header(None),
        NotificationType: str = Body(None),
        # file: str = Body(None),
        #Name
        #Audio_0_Language":"spa"
        Audio_0_Language: str = Body(None),
        Name : str = Body(None),
        StreamSubtitle: bool = Body(False),
        Id: str = Body(None), #Session Id
        ItemId: str = Body(None),
        # request: Request = None,  # To capture the raw body
):
    # Log the received data
    logging.info(f"User Agent: {user_agent}")
    logging.info(f"Notification Type: {NotificationType}")
    logging.info(f"Movie: {Name}")
    logging.info(f"Item ID: {ItemId}")
    logging.info(f"Session ID: {Id}")
    logging.info(f"Should stream subtitle: {StreamSubtitle}")
    logging.info(f"Audio_0_Language: {Audio_0_Language}")
    
    #Maybe name, genre, overview, tagline could be usefull for the prompt
    
    if "Jellyfin-Server" in user_agent:
        logging.debug(f"Jellyfin event detected is: {NotificationType}")
        
        file_path = path_mapping(get_jellyfin_file_name(ItemId, jellyfinserver, jellyfintoken))
        jellyfin_stream_subtile = True
        procmediaonplay = True
        if NotificationType == "PlaybackStart" and procmediaonplay and jellyfin_stream_subtile:
            logging.info("Starting Jellyfin subtitle stream")
            stream_subtitle_to_jellyfin(file_path, jellyfin_movie_id=ItemId, jellyfin_session_id=Id, language=LanguageCode.from_iso_639_2(Audio_0_Language), segment_duration=60 * 3)

        elif (NotificationType == "ItemAdded" and procaddedmedia) or (NotificationType == "PlaybackStart" and procmediaonplay):
            logging.debug(f"Adding item from Jellyfin to the queue: {file_path}")
            gen_subtitles_queue(path_mapping(file_path), transcribe_or_translate)
            try:
                refresh_jellyfin_metadata(ItemId, jellyfinserver, jellyfintoken)
                logging.info(f"Metadata for item {ItemId} refreshed successfully.")
            except Exception as e:
                logging.error(f"Failed to refresh metadata for item {ItemId}: {e}")
        else:
            logging.info(f"Skipping Jellyfin event: {NotificationType}")
    else:
        return {
            "message": "This doesn't appear to be a properly configured Jellyfin webhook, please review the instructions again!"}

    return ""


@app.post("/emby")
def receive_emby_webhook(
        user_agent: Union[str, None] = Header(None),
        data: Union[str, None] = Form(None),
):
    logging.debug("Raw response: %s", data)

    if not data:
        return ""

    data_dict = json.loads(data)
    event = data_dict['Event']
    logging.debug("Emby event detected is: " + event)

    # Check if it's a notification test event
    if event == "system.notificationtest":
        logging.info("Emby test message received!")
        return {"message": "Notification test received successfully!"}

    if (event == "library.new" and procaddedmedia) or (event == "playback.start" and procmediaonplay):
        fullpath = data_dict['Item']['Path']
        logging.debug("Path of file: " + fullpath)
        gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)

    return ""
    
@app.post("/batch")
def batch(
        directory: Union[str, None] = Query(default=None),
        forceLanguage: Union[str, None] = Query(default=None)
):
    transcribe_existing(directory, LanguageCode.from_string(forceLanguage))
    
# idea and some code for asr and detect language from https://github.com/ahmetoner/whisper-asr-webservice
@app.post("//asr")
@app.post("/asr")
async def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    initial_prompt: Union[str, None] = Query(default=None),  # Not used by Bazarr
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),  # Not used by Bazarr/always False
    output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps: bool = Query(default=False, description="Word-level timestamps"),  # Not used by Bazarr
):
    try:
        logging.info(f"Transcribing file '{video_file}' from Bazarr/ASR webhook" if video_file else "Transcribing file from Bazarr/ASR webhook")
        
        result = None
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))

        if force_detected_language_to:
            language = force_detected_language_to.from_iso_639_1()
            logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")

        start_time = time.time()
        start_model()

        task_id = {'path': f"Bazarr-asr-{random_name}"}
        task_queue.put(task_id)

        args = {}
        args['progress_callback'] = progress

        if not encode:
            args['audio'] = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000
        else:
            args['audio'] = audio_file.file.read()

        if custom_regroup:
            args['regroup'] = custom_regroup

        args.update(kwargs)

        result = model.transcribe_stable(task=task, language=language, **args)
        appendLine(result)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(
            f"Transcription of '{video_file}' from Bazarr complete, it took {minutes} minutes and {seconds} seconds to complete." if video_file 
            else f"Transcription complete, it took {minutes} minutes and {seconds} seconds to complete.")
    
    except Exception as e:
        logging.error(
            f"Error processing or transcribing Bazarr file: {video_file} -- Exception: {e}" if video_file
            else f"Error processing or transcribing Bazarr file Exception: {e}"
        )
    
    finally:
        await audio_file.close()
        task_queue.task_done()
        delete_model()
    
    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath=None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Transcribed using stable-ts from Subgen!',
            }
        )
    else:
        return
@app.post("//detect-language")
@app.post("/detect-language")
async def detect_language(
        audio_file: UploadFile = File(...),
        #encode: bool = Query(default=True, description="Encode audio first through ffmpeg") # This is always false from Bazarr
        detect_lang_length: int = Query(default=30, description="Detect language on the first X seconds of the file")
):    
    detected_language = LanguageCode.NONE
    language_code = 'und'
    if force_detected_language_to:
            logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}\n Returning without detection")
            return {"detected_language": force_detected_language_to.to_name(), "language_code": force_detected_language_to.to_iso_639_1()}
    if int(detect_lang_length) != 30:
        global detect_language_length 
        detect_language_length = detect_lang_length
    if int(detect_language_length) != 30:
        logging.info(f"Detect language is set to detect on the first {detect_language_length} seconds of the audio.")
    try:
        start_model()
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))
        
        task_id = { 'path': f"Bazarr-detect-language-{random_name}" }     
        task_queue.put(task_id)
        args = {}
        #sample_rate = next(stream.rate for stream in av.open(audio_file.file).streams if stream.type == 'audio')
        audio_file.file.seek(0)
        args['progress_callback'] = progress
        args['input_sr'] = 16000
        args['audio'] = whisper.pad_or_trim(np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0, args['input_sr'] * int(detect_language_length))

        args.update(kwargs)
        detected_language = LanguageCode.from_name(model.transcribe_stable(**args).language)
        logging.debug(f"Detected language: {detected_language.to_name()}")
        # reverse lookup of language -> code, ex: "english" -> "en", "nynorsk" -> "nn", ...
        language_code = detected_language.to_iso_639_1()
        logging.debug(f"Language Code: {language_code}")

    except Exception as e:
        logging.info(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
        
    finally:
        await audio_file.close()
        task_queue.task_done()
        delete_model()

        return {"detected_language": detected_language.to_name(), "language_code": language_code}

def detect_language_task(path, skip_skip_check = False):
    detected_language = LanguageCode.NONE
    global detect_language_length 

    logger.info(f"[Detect language task] Detecting language of file: {path} on the first {detect_language_length} seconds of the file")

    try:
        start_model()
        logging.info("[Detect language task] model started.")
        minimum_language_probability = 0.8
        
        #TODO make this an env variable
        should_write_detected_language = False
        
        # maybe check first if there are multiple audio tracks.
        audio_tracks = extract_nparray_audio_tracks(path, duration=detect_language_length, offset=detect_language_start_offset)
        number_of_tracks = len(audio_tracks)
        logging.info(f"[Detect language task] {number_of_tracks} Audio tracks extracted.") 
        if number_of_tracks == 0:
            logger.warning(f"[Detect language task] File {path} doesn't have any audio to transcribe")
            return
        elif number_of_tracks == 1:
            audio, index = audio_tracks[0]
            try:
                #TODO fix this not returning language
                language = detect_most_probable_language(model, audio, seconds=detect_language_length, probability_threshold=minimum_language_probability)
                
                if language:
                    detected_language = LanguageCode.from_iso_639_1(language)
                    if detected_language == LanguageCode.NONE:
                        logger.warning(f"[Detect language task] Language detection for stream index {index} returned {language} as an unknown language. Skipping.")
                        return
                    if should_write_detected_language:
                        logging.info(f"[Detect language task] Writing audio language: {detected_language.to_name()} to file: {path}")
                        set_audio_language(path, detected_language, index)
                else:
                    logger.warning(f"[Detect language task] Language detection for stream index {index} returned no language. Skipping.")
                    return
                
            except Exception as e:
                print("An error occurred during language detection for stream index %d:" % index, str(e))
                logging.warning(traceback.format_exc())
        else: 
            #Multiple audio tracks detected
            language_tracks = []
            for audio, index in audio_tracks:
                try:
                    language = detect_most_probable_language(model, audio, seconds=detect_language_length, probability_threshold=minimum_language_probability)
                    if language:
                        track_language = LanguageCode.from_iso_639_1(language)
                        if track_language == LanguageCode.NONE:
                            logger.warning(f"[Detect language task] Language detection for stream index {index} returned {track_language} as an unknown language. Skipping.")
                            continue
                        if should_write_detected_language:
                            logging.info(f"[Detect language task] Writing audio language: {track_language.to_name()} to file: {path}")
                            # set_audio_language(path, detected_language, index)
                        language_tracks.append((index, track_language))
                    else:
                        logger.warning(f"[Detect language task] Language detection for stream index {index} returned no language. Skipping.")
                        continue
                    
                except Exception as e:
                    print("An error occurred during language detection for stream index %d:" % index, str(e))
            if len(language_tracks) > 0:
                for preferred_audio_language in preferred_audio_languages:
                    for index, language in language_tracks:
                        if language == preferred_audio_language:
                            detected_language = language
                if not detected_language:
                    detected_language = language_tracks[0][1]
            
                if should_write_detected_language:
                    logging.info(f"[Detect language task] Writing multiple audio languages: {language_tracks} to file: {path}")
                    #TODO fix this
                    # set_audio_track_languages(path, language_tracks)

        if  detected_language:
            logging.info(f"[Detect language task] Detected language of file: {os.path.basename(path)} is: {detected_language.to_name()}. Will add it to the gen subtitles queue")
            gen_subtitles_queue(path, transcribe_or_translate, force_language=detected_language, skip_skip_check=skip_skip_check)
    
    except Exception as e:
        logging.error(f"[Detect language task] Error detectign language of file with whisper: {e}")
        


    
    finally:
        logging.info(f"[Detect language task] Detected language of file: {path} is completed.")
        delete_model()
 
        #maybe modify the file to contain detected language so we won't trigger this again
        return

    
    
def write_audio_language(path, language: LanguageCode):
    #write to audio track instead
    """Write metadata without keeping a temporary file."""
    base, ext = os.path.splitext(path)
    temp_path = f"{base}.tmp{ext}" #TODO FIND OUT WHY ..EXT
    (
        ffmpeg
        .input(path)
        .output(temp_path, **{"metadata": f"language={language.to_iso_639_2_t()}"}, codec="copy")
        .overwrite_output()
        .run(quiet=True)
    )
    os.replace(temp_path, path)
    
def set_audio_track_languages(file_path, language_tracks):
    """
    Sets the language metadata for all audio tracks in a file.

    Args:
        file_path (str): Path to the media file.
    """
    #TODO fix this
    # Prepare temporary file for output
    base, ext = os.path.splitext(file_path)
    temp_path = f"{base}.tmp{ext}"
    
    # Prepare metadata arguments for each audio track
    metadata_args = {}
    for index, language_code in language_tracks:
        metadata_args[f"metadata:s:a:{index}"] = f"language={language_code.to_iso_639_2_t()}"
    
    # Set language metadata for all audio streams
    ffmpeg.input(file_path)\
        .output(temp_path,
                codec="copy",  # Copy streams without re-encoding
                **metadata_args)\
        .overwrite_output()\
        .run(quiet=True)
    
    # Replace the original file with the modified file
    os.replace(temp_path, file_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
def set_audio_language(file_path, language_code: LanguageCode, index=0):
    """
    Sets the language metadata of the audio track in a file.

    Args:
        file_path (str): Path to the media file.
        language_code (LanguageCode)
    """
    #TODO fix this
    try:
        
        # Prepare temporary file for output
        base, ext = os.path.splitext(file_path)
        temp_path = f"{base}.tmp{ext}"
    
        
        # Set language metadata for the first audio stream
        ffmpeg.input(file_path)\
            .output(temp_path,
                    codec="copy",  # Copy streams without re-encoding
                    **{f"metadata:s:a:{index}": f"language={language_code.to_iso_639_2_t()}"})\
            .overwrite_output()\
            .run(quiet=True)
        
        # Replace the original file with the modified file
        os.replace(temp_path, file_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"Audio language metadata set to '{language_code}' successfully.")
    
    except ffmpeg.Error as e:
        print(f"Error setting audio language: {e.stderr.decode('utf-8')}")
    except KeyError:
        print("Invalid metadata or stream information.")
    
def set_video_language(file_path, language_code: LanguageCode):
    """
    Sets the language metadata of a video track in a file.
    Args:
        language_code (str): ISO 639-2 language code (e.g., 'eng', 'spa').
    """
    try:
        base, ext = os.path.splitext(file_path)
        temp_path = f"{base}.tmp{ext}"
        ffmpeg.input(file_path)\
            .output(temp_path, 
                    codec="copy",  # Copy streams without re-encoding
                    **{"metadata:s:v:0": f"language={language_code.to_iso_639_2_t()}"})\
            .overwrite_output()\
            .run(quiet=True)
        # Clean up the temporary file if it wasn't replaced
        os.replace(temp_path, file_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except ffmpeg.Error as e:
        print(f"Error setting video language: {e.stderr.decode('utf-8')}")

def get_video_language(input_path):
    """
    Reads the language metadata of a video stream in a file.

    Args:
        input_path (str): Path to the input video file.

    Returns:
        str: The ISO 639-2 language code for the video stream or None if no language is set.
    """
    try:
        # Get metadata for the specified stream
        probe = ffmpeg.probe(input_path, v='error', select_streams='v:0', show_entries='stream_tags=language')
        language = None
        if 'tags' in probe['streams'][0]:
            language = probe['streams'][0]['tags'].get('language', None)
        
        if language:
            return LanguageCode.from_iso_639_2(language)
        else:
            logging.debug("No language metadata found for video stream.")
            return LanguageCode.NONE
    except ffmpeg.Error as e:
        logging.error(f"Error reading video language: {e.stderr.decode('utf-8')}")
        return LanguageCode.NONE

def extract_audio_segment_to_memory(input_file, start_time, duration):
    """
    Extract a segment of audio from input_file, starting at start_time for duration seconds.
    
    :param input_file: Path to the input audio file
    :param start_time: Start time in seconds (e.g., 60 for 1 minute)
    :param duration: Duration in seconds (e.g., 30 for 30 seconds)
    :return: BytesIO object containing the audio segment
    """
    try:
        # Run FFmpeg to extract the desired segment
        out, _ = (
            ffmpeg
            .input(input_file, ss=start_time, t=duration)  # Start time and duration
            .output('pipe:1', format='wav', ac=1, ar=16000)  # Output to pipe as WAV acodec='pcm_s16le'
            .run(capture_stdout=True, capture_stderr=True)
        )
        return  np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        print("Error occurred:", e.stderr.decode())
        return None

def start_model():
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)


def delete_model():
    gc.collect()
    if clear_vram_on_complete and task_queue.qsize() == 0:
        global model
        logging.debug("Queue is empty, clearing/releasing VRAM")
        model = None

def isAudioFileExtension(file_extension):
    return file_extension.casefold() in \
        AUDIO_EXTENSIONS

def write_lrc(result, file_path):
    with open(file_path, "w") as file:
        for segment in result.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            fraction = int((segment.start - int(segment.start)) * 100)
            file.write(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}] {segment.text}\n")
            
def get_updated_subtitle_tags(subtitle_tags, language = None):
    current_subtitle_tags = []
    for tag in subtitle_tags:
        updated_tag = tag
        if isinstance(tag, SubtitleTagType.BaseTag):
            if tag == SubtitleTagType.LANGUAGE:
                if language:
                    updated_tag = SubtitleTagType.LANGUAGE(language=language, subtitle_language_naming_type=subtitle_language_naming_type)
                elif tag._stored_arguments.get("language", LanguageCode.NONE) is not LanguageCode.NONE:
                    updated_tag = tag
                else:
                    updated_tag = None
            elif tag == SubtitleTagType.SETTING:
                if tag._stored_arguments.get("setting_name") is not None:
                    setting_name = tag._stored_arguments.get("setting_name").lower()
                    value = locals().get(setting_name, None)
                    if not value:
                        value = globals().get(setting_name, None)
                    if value:
                        updated_tag = SubtitleTagType.SETTING(setting_name=setting_name, value=value, rename=tag._stored_arguments.get("rename", None))
                    else:
                        logging.warning("Setting %s not found, skipping" % setting_name)
                        updated_tag = None
                else:
                    logging.warning("Setting name not set, skipping")
                    updated_tag = None
        else:
            if tag is not None:
                if not isinstance(tag, str):
                    logging.warning(f"Unknown tag type: {tag.__class__}")
        if updated_tag:        
            current_subtitle_tags.append(updated_tag)
    return current_subtitle_tags

def gen_subtitles(file_path: str, transcription_type: str, force_language : LanguageCode = LanguageCode.NONE) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: str - The path to the video file.
        transcription_type: str - The type of transcription or translation to perform.
        force_language: str - The language to force for transcription or translation. Default is None.
    """

    try:
        logging.info(f"Preparing to transcribe file: {os.path.basename(file_path)} in {force_language if force_language else 'Unkown Language'}")

        start_time = time.time()
        start_model()
        
        # Check if the file is an audio file before trying to extract audio 
        file_name, file_extension = os.path.splitext(file_path)
        is_audio_file = isAudioFileExtension(file_extension)
        
        data = file_path
        
        
        if not is_audio_file:
            # Extract audio from the file if it has multiple audio tracks
            #TODO maybe make this return the language of the audio track too
            extracted_audio_file = handle_multiple_audio_tracks(file_path, force_language)
            if extracted_audio_file:
                logging.debug(f"Extracted {force_language} audio from {file_path}")
                data = extracted_audio_file.read()
        
   
        args = {}

        args['progress_callback'] = progress
            
        if custom_regroup:
            args['regroup'] = custom_regroup
        
            
        args.update(kwargs)
        
        # transcription_prompt = "This audio contains a dialogue from a movie. Please transcribe the spoken dialogue clearly and include any significant non-verbal sounds like music, laughter, sound effects, or background noises that may be important to understanding the context. Ensure that the transcription is complete and provides all meaningful audio content, such as identifying who is speaking, as well as key sounds or music cues."
        transcription_prompt = ""
        if use_model_prompt and custom_model_prompt != "":
            initial_prompt = custom_model_prompt

        #     #TODO STREAMING MODE WITH JELLYFIN INTEGRATION, 
        #     #TODO OPTIONALLY ADD MESSAGE AFTER CHUNK IN SUBTITLE THAT IT IS STILL GENERATING. AND DELETE THAT MESSAGE WHEN THE NEXT CHUNK IS READY, BUT DO NOT PUT THE MESSSAGE WHEN IT DID ALL CHUNKS
        #     #TODO  Could not initialize NNPACK! Reason: Unsupported hardware.
        #     #TODO Support writing to ass file as well. And have more subtile word highlighting

        
        #Updating subtitle tags to match for this subtitle

        current_subtitle_tags = get_updated_subtitle_tags(subtitle_tags, force_language)
        
        srt_vtt_word_formatting = ""
        if word_level_highlight and word_highlight_color:
            srt_vtt_word_formatting = (f'<font color="#{word_highlight_color}">', '</font>')
         
        if should_stream_subtitle:
            logging.info(f"Transcribing in chunks (streaming subtitle): {file_path}")
            stream_subtitle(file_path, current_subtitle_tags, write_intro=append, srt_vtt_word_formatting=srt_vtt_word_formatting, language=force_language, segment_duration=segment_duration, transcription_prompt=transcription_prompt, transcription_type=transcription_type, **args)
        
        else:
            #Normal transcribe    
            
            logging.info(f"Starting transcription of {file_name} in {force_language if force_language else 'Unkown Language'}")
            result = model.transcribe_stable(data, language=force_language.to_iso_639_1(), initial_prompt=transcription_prompt, task=transcription_type, **args)
            logging.info(f"Finished transcription of {file_name}")
            
            #TODO remove prompt from subtitle
    
            appendLine(result)
            
            if transcription_prompt:
                # Remove the first segment which is the transcription prompt
                #TODO better handle this in case of long prompt
                result.remove_segment(result[0])
                
            # If it is an audio file, write the LRC file
            if is_audio_file and lrc_for_audio_files:
                # todouse denoiser="demucs" and vad=True for music
                write_lrc(result, file_name + '.lrc')
                
                # maybe add support for this https://en.wikipedia.org/wiki/LRC_(file_format)
                # A2 extension (Enhanced LRC format)
            else:
                if not force_language:
                    force_language = LanguageCode.from_iso_639_1(result.language)
                
                subtitle_file_name = name_subtitle(file_path, FileWriteBehavior.UNIQUE, tags=current_subtitle_tags)
                if subtitle_file_name:
                    result.to_srt_vtt(subtitle_file_name, word_level=word_level_highlight, tag=srt_vtt_word_formatting)
                    logging.info(f"Subtitle file written to: {subtitle_file_name}")
                else:
                    logging.warning(f"Subtitle file not written for {file_path}")
            #TODO maybe multiple output formats?

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(
            f"Transcription of {os.path.basename(file_path)} is completed, it took {minutes} minutes and {seconds} seconds to complete.")

    except Exception as e:
        logging.warning(f"Error processing or transcribing {file_path} in {force_language}: {e}")
        
        logging.warning(traceback.format_exc())
        

    finally:
        if skip_list_file_name:
            write_to_skip_list(skip_list_file_name, file_path)
        delete_model()

def handle_multiple_audio_tracks(file_path: str, language: LanguageCode | None = None) -> io.BytesIO | None:
    """
    Handles the possibility of a media file having multiple audio tracks.
    
    If the media file has multiple audio tracks, it will extract the audio track of the selected language. Otherwise, it will extract the first audio track.
    
    Parameters:
    file_path (str): The path to the media file.
    language (LanguageCode | None): The language of the audio track to search for. If None, it will extract the first audio track.
    
    Returns:
    io.BytesIO  | None: The audio or None if no audio track was extracted.
    """
    audio_bytes = None
    audio_tracks = get_audio_tracks(file_path)

    if len(audio_tracks) > 1:
        logging.debug(f"Handling multiple audio tracks from {file_path} and planning to extract audio track of language {language}")
        logging.debug(
            "Audio tracks:\n"
            + "\n".join([f"  - {track['index']}: {track['codec']} {track['language']} {('default' if track['default'] else '')}" for track in audio_tracks])
        )

        if language is not None:
            audio_track = get_audio_track_by_language(audio_tracks, language)
        if audio_track is None:
            #TODO check which language this track is
            audio_track = audio_tracks[0]
        
        audio_bytes = extract_audio_track_to_memory(file_path, audio_track["index"])
        if audio_bytes is None:
            logging.error(f"Failed to extract audio track {audio_track['index']} from {file_path}")
            return None
    return audio_bytes

#todo get_audio_track_index_by_language(file_path, language)
def get_audio_track_index_by_language(file_path, language):
    audio_tracks = get_audio_tracks(file_path)
    for track in audio_tracks:
        if track['language'] == language:
            return track['index']
    return audio_tracks[0]['index'] 

def extract_audio_tracks(video_path):
    probe = ffmpeg.probe(video_path)
    audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
    audio_tracks = []
    for i, stream in enumerate(audio_streams):
        audio_data = extract_audio_track_to_memory(video_path, stream['index'])
        if audio_data is not None:
            audio_tracks.append((audio_data, stream['index']))
    return audio_tracks

def extract_nparray_audio_tracks(video_path, offset=0, duration=None):
    start_time = time.time()
    audio_streams = []
    try:
        probe = ffmpeg.probe(video_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
    except ffmpeg.Error as e:
        print("An error occurred while probing the video file:", e.stderr.decode())
        return []
    
    audio_tracks = []
    for i, stream in enumerate(audio_streams):
        audio_data = extract_audio_track_to_nparray(video_path, stream['index'], offset, duration)
        if audio_data is not None:
            audio_tracks.append((audio_data, stream['index']))
    end_time = time.time()
    print(f"Extracting audio tracks took {end_time - start_time:.2f} seconds")
    return audio_tracks

def extract_audio_track_to_nparray(input_video_path, track_index, offset=0, duration=None) -> np.ndarray | None:
    """
    Extract a specific audio track from a video file to memory using FFmpeg.

    Args:
        input_video_path (str): The path to the video file.
        track_index (int): The index of the audio track to extract. If None, skip extraction.
        offset (int): The beginning offset of the audio track in seconds. Default is 0.
        duration (int): The total duration of the audio track in seconds. Default is None (extract until the end).

    Returns:
        np.ndarray | None: The audio data as a numpy array, or None if extraction failed.
    """
    if track_index is None:
        # logging.warning(f"Skipping audio track extraction for {input_video_path} because track index is None")
        return None

    try:
        # Use FFmpeg to extract the specific audio track and output to memory
        ffmpeg_input = ffmpeg.input(input_video_path, ss=offset)
        if duration is not None:
            ffmpeg_input = ffmpeg_input.output(
                "pipe:",  # Direct output to a pipe
                map=f"0:{track_index}",  # Select the specific audio track
                format="wav",             # Output format
                ac=1,                     # Mono audio (optional)
                ar=16000,                 # Sample rate 16 kHz (recommended for speech models)
                t=duration,               # Set the duration of the audio track
                loglevel="quiet"
            )
        else:
            ffmpeg_input = ffmpeg_input.output(
                "pipe:",  # Direct output to a pipe
                map=f"0:{track_index}",  # Select the specific audio track
                format="wav",             # Output format
                ac=1,                     # Mono audio (optional)
                ar=16000,                 # Sample rate 16 kHz (recommended for speech models)
                loglevel="quiet"
            )
        out, _ = ffmpeg_input.run(capture_stdout=True, capture_stderr=True)  # Capture output in memory
        # https://github.com/openai/whisper/blob/25639fc/whisper/audio.py#L25-L62
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    except ffmpeg.Error as e:
        print("An error occurred while extracting audio track:", e.stderr.decode())
        return None

def extract_audio_track_to_memory(input_video_path, track_index) -> io.BytesIO | None:
    """
    Extract a specific audio track from a video file to memory using FFmpeg.

    Args:
        input_video_path (str): The path to the video file.
        track_index (int): The index of the audio track to extract. If None, skip extraction.

    Returns:
        io.BytesIO | None: The audio data as a BytesIO object, or None if extraction failed.
    """
    if track_index is None:
        logging.warning(f"Skipping audio track extraction for {input_video_path} because track index is None")
        return None

    try:
        # Use FFmpeg to extract the specific audio track and output to memory
        out, _ = (
            ffmpeg.input(input_video_path)
            .output(
                "pipe:",  # Direct output to a pipe
                map=f"0:{track_index}",  # Select the specific audio track
                format="wav",             # Output format
                ac=1,                     # Mono audio (optional)
                ar=16000,                 # Sample rate 16 kHz (recommended for speech models)
                loglevel="quiet"
            )
            .run(capture_stdout=True, capture_stderr=True)  # Capture output in memory
        )
        # Return the audio data as a BytesIO object
        #TODO do not return io bytes but a numpy array
        logging.info(f"Extracted audio track {track_index} from {input_video_path}")
        return io.BytesIO(out)

    except ffmpeg.Error as e:
        print("An error occurred:", e.stderr.decode())
        return None
    
def transcribe_chunk(chunk_audio, language: LanguageCode, transcription_prompt = "", transcription_type = "transcribe", **args):
    # # Convert in-memory audio chunk to a format stable-ts can handle (e.g., wav)
    # audio_chunk_io = io.BytesIO()
    # audio_chunk_io.write(chunk_audio)  # Write the chunk data to the in-memory buffer
    # audio_chunk_io.seek(0)  # Reset the pointer to the beginning
    
    # Transcribe using the model
    
    # transcription = model.transcribe_stable(audio_chunk_io.read(), language=language.to_iso_639_1(), initial_prompt="", task=transcription_type, vad=True, suppress_silence=True, **args) # , beam_size=5
    transcription = model.transcribe_stable(chunk_audio, language=language.to_iso_639_1(), initial_prompt=transcription_prompt, task=transcription_type, **args) # , beam_size=5
    # if transcription_prompt:
    #     # Remove the first segment which is the transcription prompt
    #     transcription.remove_segment(transcription[0])
                
    return transcription  # Return the transcription object

def split_audio_in_memory(input_file, chunk_duration_ms, audio_track_index):
    # Use ffmpeg to read the input audio file into memory
    input_stream = ffmpeg.input(input_file)
    
    # Get the total duration of the audio in seconds
    probe = ffmpeg.probe(input_file, v='error', select_streams='a', show_entries='format=duration')
    total_duration = float(probe['format']['duration'])

    chunks = []
    for start_seconds in range(0, int(total_duration), int(chunk_duration_ms / 1000)):  # Convert ms to seconds
        try:
            out, _ = (
                input_stream
                .output('pipe:1', ss=start_seconds, t=chunk_duration_ms / 1000, format='wav', map=f'0:{audio_track_index}', ac=1, ar=16000)  
                .run(capture_stdout=True, capture_stderr=True)
            )
            chunks.append(out)
        except ffmpeg.Error as e:
            logging.error(f"Error processing chunk starting at {start_seconds}s: {e.stderr.decode()}")
            break
    
    return chunks

def stream_subtitle(input_file, subtitle_tags, srt_vtt_word_formatting = "", write_intro = True, language: LanguageCode = LanguageCode.NONE, segment_duration: int = 60, transcription_prompt: str = "", transcription_type: str = "transcribe", **args):
    subtitle_file = name_subtitle(input_file, tags=subtitle_tags)
    
    if subtitle_file is None:
        logging.warning(f"Subtitle file could not be written for {input_file}")
        return
    
    index = 0
    
    if write_intro:
        with open(subtitle_file, "w", encoding="utf-8") as f:
                starting_segment = segment2srtblock({
                    "start": 0,
                    "end": 5,
                    "text": "Generating subtitles...\n"
                    }, index)
                f.write(starting_segment)
        index += 1
        

    probe = ffmpeg.probe(input_file, v='error', select_streams='a', show_entries='format=duration')
    total_duration = float(probe['format']['duration'])

    
    start_times = range(transcribe_offset_seconds, int(total_duration), segment_duration)
    total_segments = len(start_times)
    logging.info(f"Transcribing audio in {len(start_times)} chunks of {segment_duration} seconds")
    # Transcribe audio in chunks. 
    # start_model()
    gen_start_time = time.time()
    for segment_index, start_time in enumerate(start_times, start=1): 
        audio_segment = extract_audio_segment_to_memory (input_file, start_time, segment_duration)
        if any(audio_segment):
            logging.info(f"Transcribing audio segment from {sec2vtt(start_time)} to {sec2vtt(start_time + segment_duration)}...")
            transcription_result = transcribe_chunk(audio_segment, language=language, transcription_prompt=transcription_prompt, transcription_type=transcription_type, **args)
            transcription_result.reassign_ids(start=index) # not sure what only_segments: bool = False
            transcription_result.offset_time(start_time)
            index += len(transcription_result.segments)
            logging.info("Transcription for audio segment complete.")
            with open(subtitle_file, "a", encoding="utf-8") as f:
                f.write(transcription_result.to_srt_vtt(word_level=word_level_highlight, tag = srt_vtt_word_formatting))
                logging.info(f"Wrote transcription for audio segment to {subtitle_file}. from {start_time} to {start_time + segment_duration}")
                        # Calculate and log progress
                        
                        
                        # Calculate timing
            elapsed_time = time.time() - gen_start_time
            average_time_per_segment = elapsed_time / segment_index
            estimated_total_time = average_time_per_segment * total_segments
            estimated_time_remaining = estimated_total_time - elapsed_time
            percentage = (segment_index / total_segments) * 100
            # Calculate processing speed (audio seconds per real-time second)
            total_audio_processed = segment_index * segment_duration
            processing_speed = total_audio_processed / elapsed_time  # seconds of audio per second of real time

            logging.info(
                f"Progress: {percentage:.2f}% ({segment_index}/{total_segments}) completed. "
                f"Elapsed time: {sec2vtt(elapsed_time)}, Estimated remaining time: {sec2vtt(estimated_time_remaining)}, "
                f"Total estimated time: {sec2vtt(estimated_total_time)}. Speed: {processing_speed:.2f}x (audio seconds/second)."
            )
        else:
            logging.warning(f"Audio segment from {start_time} to {start_time + segment_duration} not extracted.")
    logging.info(f"Transcription complete. Wrote everything to {subtitle_file}. It took {sec2vtt(elapsed_time)}")
    # delete_model()
            
            

def get_audio_track_by_language(audio_tracks, language):
    """
    Returns the first audio track with the given language.
    
    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
        language (str): The language of the audio track to search for.
    
    Returns:
        dict: The first audio track with the given language, or None if no match is found.
    """
    for track in audio_tracks:
        if track['language'] == language:
            return track
    return None

def choose_transcribe_language(file_path, forced_language: LanguageCode):
    """
    Determines the language to be used for transcription based on the provided
    file path and language preferences.

    Args:
        file_path: The path to the file for which the audio tracks are analyzed.
        forced_language: The language to force for transcription if specified.

    Returns:
        The language code to be used for transcription. It prioritizes the
        `forced_language`, then the environment variable `force_detected_language_to`,
        then the preferred audio language if available, and finally the default
        language of the audio tracks. Returns None if no language preference is
        determined.
    """
    
    logger.debug(f"choose_transcribe_language({file_path}, {forced_language})")
    
    if force_detected_language_to:
        logger.debug(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")
        return force_detected_language_to
    
    if forced_language:
        logger.debug(f"Language already is set: {forced_language}")   
        return forced_language

    audio_tracks = get_audio_tracks(file_path)
    
    found_track_in_language = find_language_audio_track(audio_tracks, preferred_audio_languages)
    if found_track_in_language:
        language = found_track_in_language
        if language:
            logger.debug(f"Preferred language found: {language}")
            return language
    
    default_language = find_default_audio_track_language(audio_tracks)
    if default_language:
        logger.debug(f"Default language found: {default_language}")
        return default_language
    
    if detect_language_in_filename:
        language = find_language_in_filename(file_path)
        if language:
            logger.debug(f"Language detected in filename: {language}")
            return language
    
    # container_language = get_container_language(file_path)
    # if container_language:
    #     logger.debug(f"Container language found: {container_language}")
    #     return container_language
    
    # video_language = get_video_language(file_path)
    # if video_language:
    #     logger.debug(f"Video language found: {video_language}")
    #     return video_language

    return LanguageCode.NONE 

def get_container_language(video_path):
    """Extract general language metadata from a video file."""
    # Use ffprobe to get metadata
    probe = ffmpeg.probe(video_path)
    
    # Extract format-level tags (container-level metadata)
    format_tags = probe.get('format', {}).get('tags', {})
    
    # Look for language or related metadata
    language = LanguageCode.from_iso_639_2(format_tags.get('language', 'unknown'))  # Default to 'unknown'
    return language
    
def get_audio_tracks(video_file):
    """
    Extracts information about the audio tracks in a file.

    Returns:
        List of dictionaries with information about each audio track.
        Each dictionary has the following keys:
            index (int): The stream index of the audio track.
            codec (str): The name of the audio codec.
            channels (int): The number of audio channels.
            language (LanguageCode): The language of the audio track.
            title (str): The title of the audio track.
            default (bool): Whether the audio track is the default for the file.
            forced (bool): Whether the audio track is forced.
            original (bool): Whether the audio track is the original.
            commentary (bool): Whether the audio track is a commentary.

    Example:
        >>> get_audio_tracks("french_movie_with_english_dub.mp4")
        [
            {
                "index": 0,
                "codec": "dts",
                "channels": 6,
                "language": LanguageCode.FRENCH,
                "title": "French",
                "default": True,
                "forced": False,
                "original": True,
                "commentary": False
            },
            {
                "index": 1,
                "codec": "aac",
                "channels": 2,
                "language":  LanguageCode.ENGLISH,
                "title": "English",
                "default": False,
                "forced": False,
                "original": False,
                "commentary": False
            }
        ]

    Raises:
        ffmpeg.Error: If FFmpeg fails to probe the file.
    """
    try:
        # Probe the file to get audio stream metadata
        probe = ffmpeg.probe(video_file, select_streams='a')
        audio_streams = probe.get('streams', [])
        
        # Extract information for each audio track
        audio_tracks = []
        for stream in audio_streams:
            audio_track = {
                "index": int(stream.get("index", None)),
                "codec": stream.get("codec_name", "Unknown"),
                "channels": int(stream.get("channels", None)),
                "language": LanguageCode.from_iso_639_2(stream.get("tags", {}).get("language", "und")),
                "title": stream.get("tags", {}).get("title", "None"),
                "default": stream.get("disposition", {}).get("default", 0) == 1,
                "forced": stream.get("disposition", {}).get("forced", 0) == 1,
                "original": stream.get("disposition", {}).get("original", 0) == 1,
                "commentary": "commentary" in stream.get("tags", {}).get("title", "").lower()
            }
            audio_tracks.append(audio_track)    
        return audio_tracks

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"An error occurred while reading audio track information: {str(e)}")
        return []

def find_language_audio_track(audio_tracks, find_languages):
    """
    Checks if an audio track with any of the given languages is present in the list of audio tracks.
    Returns the first language from `find_languages` that matches.
    
    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
        find_languages (list): A list  language codes to search for.
    
    Returns:
        str or None: The first language found from `find_languages`, or None if no match is found.
    """
    for language in find_languages:
        for track in audio_tracks:
            if track['language'] == language:
                return language
    return None
def find_default_audio_track_language(audio_tracks):    
    """
    Finds the language of the default audio track in the given list of audio tracks.

    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
            Must contain the key "default" which is a boolean indicating if the track is the default track.

    Returns:
        str: The ISO 639-2 code of the language of the default audio track, or None if no default track was found.
    """
    for track in audio_tracks:
        if track['default'] is True:
            return track['language']
    return None
    
    
def gen_subtitles_queue(file_path: str, transcription_type: str, force_language: LanguageCode = LanguageCode.NONE, skip_skip_check = False) -> None:
    global task_queue
    
    logging.debug(f"gen_subtitles_queue({file_path}, {transcription_type}, {force_language}, {skip_skip_check})")
    
    if not is_valid_path(file_path):
        logging.warning(f"{file_path} is not a valid path")
        return
        
    
    if not has_audio(file_path):
        logging.warning(f"{file_path} doesn't have any audio to transcribe!")
        return
    
    
    if not skip_skip_check and have_to_skip_before_choosing_language(file_path, force_language):
        logging.info(f"Skipping: {os.path.basename(file_path)}")
        if skip_list_file_name:
            write_to_skip_list(skip_list_file_name, file_path)
        return

    
    force_language = choose_transcribe_language(file_path, force_language)
    
    # check if we would like to detect audio language in case of no audio language specified. Will return here again with specified language from whisper. #TODO Probably should make a function from what's down here and just call that instead.
    if not force_language and should_whiser_detect_audio_language:
        # make a detect language task
        task_id = { 'path': file_path, 'type': "detect_language", 'skip_skip_check': skip_skip_check}
        task_queue.put(task_id)
        logging.debug(f"task_queue.put(task_id)({file_path}, detect_language)")
        return
    
    
    if not skip_skip_check and have_to_skip(file_path, force_language):
        logging.debug(f"Skipping: {os.path.basename(file_path)}")
        if skip_list_file_name:
            write_to_skip_list(skip_list_file_name,file_path)
        return
    
    task = {
        'path': file_path,
        'transcribe_or_translate': transcription_type,
        'force_language': force_language
    }
    task_queue.put(task)
    logging.info(f"Added to queue: [{transcription_type}] {force_language} {os.path.basename(file_path)}")

def have_to_skip_before_choosing_language(file_path: str, force_language: LanguageCode) -> bool:
    
    if force_language and limit_to_preferred_audio_languages:
        if force_language not in preferred_audio_languages:
            logging.debug(f"Skipping {file_path} because {force_language} is not in the preferred languages")
            return True
    
    if skip_if_preferred_audio_language_sub_already_exist and check_if_preferred_audio_language_sub_already_exist(file_path):
        return True
    
    
    # Check if subtitles in the specified internal language(s) should skip processing
    # TODO should remove this without breaking backwards compatibility
    if skipifinternalsublang and has_subtitle_language(file_path, skipifinternalsublang):
        logging.debug(f"{file_path} has internal subtitles matching skip condition, skipping.")
        return True

    # Check if external subtitles exist for the specified language
    # Probably not use LanguageCode for this, but just check with strings, to be able to skip with custom named languages. 
    if LanguageCode.is_valid_language(namesublang):
        if skipifexternalsub and has_subtitle_language(file_path, LanguageCode.from_string(namesublang)):
            logging.debug(f"{file_path} has external subtitles in {namesublang}, skipping.")
            return True

    # Skip if any language in the skip list is detected in existing subtitles
    existing_sub_langs = get_subtitle_languages(file_path)
    if any(lang in skip_lang_codes_list for lang in existing_sub_langs):
        logging.debug(f"Languages in skip list {skip_lang_codes_list} detected in {file_path}, skipping.")
        return True
    
    #TODO check here if subtitles exist for no language/ default probably
    if has_subtitle_language(file_path):
        #TODO maybe use another function for this to check for subtitles without knowing the target language
        logging.debug(f"{file_path} already has subtitles, skipping.")
        return True
    return False
    

def check_if_preferred_audio_language_sub_already_exist(file_path: str) -> bool:
    
    for language in preferred_audio_languages:
        if has_subtitle_language(file_path, language):
            logging.debug(f"{file_path} already has subtitles in preferred audio language {language}, skipping.")
            return True
        
    return False

def have_to_skip(file_path: str, transcribe_language: LanguageCode ) -> bool:
    """
    Determines whether subtitle generation should be skipped for a given file.

    Args:
        file_path: The path to the file to check for existing subtitles.
        transcribe_language: The language intended for transcription.

    Returns:
        True if subtitle generation should be skipped; otherwise, False.
    """
        
    if not transcribe_language:
        if skip_unknown_language:
            logging.debug(f"{file_path} has unknown language, skipping.")
            return True
        if has_subtitle_language(file_path):
            #TODO maybe use another function for this to check for subtitles without knowing the target language
            logging.debug(f"{file_path} already has subtitles, skipping.")
            return True
        
        audio_langs = get_audio_languages(file_path)
        if any(language in preferred_audio_languages for language in audio_langs):
            logging.debug(f"Preferred audio language {preferred_audio_languages} detected in {file_path}.")
            # maybe not skip if subtitle exist in preferred audio language, but not in another preferred audio language if the file has multiple audio tracks matching the preferred audio languages
        else:
            if limit_to_preferred_audio_languages:
                
                logging.debug(f"Only non-preferred audio language {audio_langs} detected in {file_path}, skipping.")
                return True
            if any(lang in skip_if_audio_track_is_in_list for lang in audio_langs):
                logging.debug(f"Audio language in skip list {skip_if_audio_track_is_in_list} detected in {file_path}, skipping.")
                return True
        
    else:
        # Check on these conditions only if the transcribe language is not NONE
    
        #if transcribing only certain languages there's no need to check for other conditions on skipping if the to transcribe language is not in the list
        if limit_to_preferred_audio_languages and transcribe_language not in preferred_audio_languages:
            logging.debug(f"{file_path} is not in preferred audio languages, skipping.")
            return True
    
        # Check if subtitles in the desired transcription language already exist
        if skip_if_to_transcribe_sub_already_exist and has_subtitle_language(file_path, transcribe_language):
            logging.debug(f"{file_path} already has subtitles in {transcribe_language}, skipping.")
            return True

        # if transcribe_language is something, we don't need to check the audio languages
        if transcribe_language not in preferred_audio_languages and limit_to_preferred_audio_languages:
            return True
        if transcribe_language in skip_if_audio_track_is_in_list:
            return True
        logging.debug(f"Audio language {transcribe_language} is not a reason to skip in {file_path}.")
        

    # If none of the conditions matched, do not skip
    logging.debug(f"Not skipping {file_path}. No conditions met to skip.")
    return False

def get_string_after_year(text: str) -> str:
    DELIMITERS = {'.': '.', '(': ')', ' ': ' '}
    YEAR_LENGTH = 4
    
    # Loop through the string and check for a 4-digit year
    for i in range(1, len(text) - YEAR_LENGTH - 1):
        # Check if current substring is a 4-digit number
        if text[i:i+YEAR_LENGTH].isdigit():
            start_delim = text[i-1]  # Get the delimiter before the year
            
            # Check if the start delimiter is in the dictionary and matches the end delimiter
            if start_delim in DELIMITERS and text[i + YEAR_LENGTH] == DELIMITERS[start_delim]:
                # Return everything after the year and its matching delimiter
                return text[i + YEAR_LENGTH + 1:]

    return ""

def find_language_in_filename(path):
    # maybe strip the title from it to not get a false positive for this on german: Pinocho de Guillermo del Toro (2022) [BluRay Rip][AC3 5.1 Castellano][www.nucleohd.com].avi
    #so to only check on this part [BluRay Rip][AC3 5.1 Castellano][www.nucleohd.com]
    # so it will return spanish instead of german
    string = os.path.splitext(os.path.basename(path))[0]
    
    # checking after the year in the filename because there usually is the Language defined if it is there
    # TODO handle multiple languages
    # TODO handle this format seriename.S01E04.Eng.Fre.Ger.Ita.Por.Spa.1080p.something.mkv
    return find_language_in_string(get_string_after_year(string))

def find_language_in_string(string):
    for part in split_words(string):
        if len(part) < 3:
            # Avoid false postives 
            continue
        if LanguageCode.is_valid_language(part):
            return LanguageCode.from_string(part)
    return LanguageCode.NONE

def has_word_in_string(string, word):
    lower_word = word.lower()
    for part in split_words(string):
        if part.lower() == lower_word:
            return True
    return False

def split_words(string):
    parts = []
    current_part = ""

    for char in string:
        if char.isalpha():  # If the character is a letter
            current_part += char
        else:
            if current_part:  # If there is a current part to add
                parts.append(current_part)
                current_part = ""

    # Add the last part if any
    if current_part:
        parts.append(current_part)

    return parts


def get_subtitle_languages(video_path): #TODO make it check subtitle files too
    """
    :param video_path: Path to the video file
    :return: List of language codes for each subtitle stream
    """
    languages = []

    languages = get_subtitle_languages_in_file(video_path)
    languages.extend(get_subtitle_languages_in_file(video_path))
    
    return languages

def get_subtitle_languages_in_file(video_path):
    languages = []

    # Open the video file
    with av.open(video_path) as container:
        # Iterate through each audio stream
        for stream in container.streams.subtitles:
            # Access the metadata for each audio stream
            if 'language' in stream.metadata:
                language_string = stream.metadata['language']
                lang_code = LanguageCode.from_iso_639_2(language_string)
                if lang_code:
                    languages.append(lang_code)
                else:
                    logging.warning(f"found {language_string} subtitle in file which is not a valid language code in {video_path}.")
    return languages

def get_file_name_without_extension(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    return file_name

def get_audio_languages(video_path):
    """
    Extract language codes from each audio stream in the video file.

    :param video_path: Path to the video file
    :return: List of language codes for each audio stream
    """
    audio_tracks = get_audio_tracks(video_path)
    return [track['language'] for track in audio_tracks]    

def has_subtitle_language(video_file, target_language: LanguageCode = LanguageCode.NONE):
    """
    Determines if a subtitle file with the target language is available for a specified video file.

    This function checks both within the video file and in its associated folder for subtitles
    matching the specified language.

    Args:
        video_file: The path to the video file.
        target_language: The language of the subtitle file to search for.

    Returns:
        bool: True if a subtitle file with the target language is found, False otherwise.
    """
    if  has_subtitle_of_language_in_folder(video_file, target_language):
        return True
    if  has_subtitle_language_in_file(video_file, target_language):
        return True
    return False

def has_subtitle_language_in_file(video_file, target_language: LanguageCode):
    """
    Checks if a video file contains subtitles with a specific language.

    Args:
        video_file: The path to the video file.
        target_language: The language of the subtitle file to search for.

    Returns:
        bool: True if a subtitle file with the target language is found, False otherwise.
    """
    
    #TODO check if tag is default is set for subtitle and implement this: assume_default_in_subtitle_is_audio_language
    
    if only_skip_if_subgen_subtitle:
        return False
    if (target_language == LanguageCode.NONE and (not skip_if_language_is_not_set_but_subtitles_exist or not skip_if_language_is_not_set_but_subtitles_exist_in_prefered_language)): # skip if language is not set or we are only interested in subgen subtitles which are not internal, only external
        return False
    try:
        with av.open(video_file) as container:
            subtitle_streams = (stream for stream in container.streams if stream.type == 'subtitle')
            
            if not any(subtitle_streams):
                logging.debug("No subtitles found in the video.")
                return False
            
            if skip_if_language_is_not_set_but_subtitles_exist and target_language == LanguageCode.NONE and any(subtitle_streams):
                logging.debug("Language is not set but internal subtitles exist.")
                return True
            for subtitle_stream in subtitle_streams:
                if 'language' in subtitle_stream.metadata:
                    subtitle_language = LanguageCode.from_iso_639_2(subtitle_stream.metadata.get('language'))
                    if subtitle_language:
                        if target_language == LanguageCode.NONE:
                            if skip_if_language_is_not_set_but_subtitles_exist_in_prefered_language and subtitle_language in preferred_audio_languages:
                                logging.debug(f"Subtitles in preferred language '{subtitle_language}' found in the video.")
                                return True
                        elif subtitle_language == target_language:
                            logging.debug(f"Subtitles in '{target_language}' language found in the video.")
                            return True
                        
                    else:
                        logging.warning(f"Subtitles without unsupported language '{subtitle_stream.metadata.get('language')}' as {subtitle_language} found in the video. Of file {video_file}")
                elif assume_no_language_in_subtitle_is_audio_language:
                    logging.debug(f"Subtitles without language found in the video. Assuming they are in the same language as the audio. Of file {video_file}")
                    return True
                    
            logging.debug(f"No subtitles in '{target_language}' language found in the video.")
            return False
    except av.AVError as e:
        logging.error(f"An error occurred while opening {os.path.basename(video_file)} with pyav: {str(e)}") # TODO: figure out why this throws (empty) errors
        logging.error(traceback.format_exc())
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while opening {os.path.basename(video_file)}: {str(e)}") # TODO: figure out why this throws (empty) errors
        logging.error(traceback.format_exc())
        return False

SUBTITLE_EXTENSIONS = ['.srt', '.vtt', '.sub', '.ass', '.ssa', '.idx', '.sbv', '.pgs', '.ttml', '.lrc']

def has_subtitle_of_language_in_folder(video_file, target_language: LanguageCode, recursion = True):
    """Checks if the given folder has a subtitle file with the given language.

    Args:
        video_file: The path of the video file.
        target_language: The language of the subtitle file that we are looking for.
        recursion: If True, search in subfolders of the given folder. If False,
            only search in the given folder.

    Returns:
        True if a subtitle file with the given language is found in the folder,
            False otherwise.
    """
    # logging.info(f" ??? has_subtitle_of_language_in_folder({video_file}, {target_language}, {recursion})")
    
    # just get the name of the movie e.g. movie.2025.remastered
    video_file_stripped = os.path.splitext(os.path.split(video_file)[1])[0]
    folder_path = os.path.dirname(video_file)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            root, ext = os.path.splitext(file_name)
            if root.startswith(video_file_stripped) and ext.lower() in SUBTITLE_EXTENSIONS:
                
                identifier_string =  root[len(video_file_stripped):]
                # logging.info(f"!!! {video_file} identifier_string: {identifier_string}")
                
                word_do_not_skip_condition = not only_skip_if_subgen_subtitle
                if only_skip_if_subgen_subtitle:
                    word_do_not_skip_condition = has_word_in_string(identifier_string, "subgen")
                
                # Only check this if word_do_not_skip_condition is True. If only_skip_if_subgen_subtitle is False then word_do_not_skip_condition is True
                if word_do_not_skip_condition:
                    if skip_if_language_is_not_set_but_subtitles_exist and not target_language:
                        return True
                    else:
                        if assume_default_in_subtitle_is_audio_language and has_word_in_string(identifier_string, "default"):
                            return True
                        
                        subtitle_language =find_language_in_string(identifier_string)
                        if subtitle_language:
                            if subtitle_language == target_language:
                                return True
                            elif skip_if_language_is_not_set_but_subtitles_exist_in_prefered_language and subtitle_language in preferred_audio_languages:
                                return True
                        elif assume_no_language_in_subtitle_is_audio_language:
                            return True
                            

                
        elif os.path.isdir(file_path) and recursion: 
            # Looking in the subfolders of the video for subtitles
            if has_subtitle_of_language_in_folder(os.path.join(file_path, os.path.split(video_file)[1]) , target_language, False):
                # If the language is found in the subfolders, return True
                return True
    return False


def get_subtitle_languages_in_folder(video_file):
    languages = []
    video_file_stripped = os.path.splitext(os.path.split(video_file)[1])[0]
    folder_path = os.path.dirname(video_file)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            root, ext = os.path.splitext(file_name)
            if root.startswith(video_file_stripped) and ext.lower() in SUBTITLE_EXTENSIONS:
                
                identifier_string =  root[len(video_file_stripped):]
                subtitle_language = find_language_in_string(identifier_string)
                if subtitle_language:
                    languages.append(subtitle_language)
    return languages
                
    

def get_plex_file_name(itemid: str, server_ip: str, plex_token: str) -> str:
    """Gets the full path to a file from the Plex server.

    Args:
        itemid: The ID of the item in the Plex library.
        server_ip: The IP address of the Plex server.
        plex_token: The Plex token.

    Returns:
        The full path to the file.
    """

    url = f"{server_ip}/library/metadata/{itemid}"

    headers = {
        "X-Plex-Token": plex_token,
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        fullpath = root.find(".//Part").attrib['file']
        return fullpath
    else:
        raise Exception(f"Error: {response.status_code}")

def refresh_plex_metadata(itemid: str, server_ip: str, plex_token: str) -> None:
    """
    Refreshes the metadata of a Plex library item.
    
    Args:
        itemid: The ID of the item in the Plex library whose metadata needs to be refreshed.
        server_ip: The IP address of the Plex server.
        plex_token: The Plex token used for authentication.
        
    Raises:
        Exception: If the server does not respond with a successful status code.
    """

    # Plex API endpoint to refresh metadata for a specific item
    url = f"{server_ip}/library/metadata/{itemid}/refresh"

    # Headers to include the Plex token for authentication
    headers = {
        "X-Plex-Token": plex_token,
    }

    # Sending the PUT request to refresh metadata
    response = requests.put(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        logging.info("Metadata refresh initiated successfully.")
    else:
        raise Exception(f"Error refreshing metadata: {response.status_code}")

def refresh_jellyfin_metadata(itemid: str, server_ip: str, jellyfin_token: str) -> None:
    """
    Refreshes the metadata of a Jellyfin library item.
    
    Args:
        itemid: The ID of the item in the Jellyfin library whose metadata needs to be refreshed.
        server_ip: The IP address of the Jellyfin server.
        jellyfin_token: The Jellyfin token used for authentication.
        
    Raises:
        Exception: If the server does not respond with a successful status code.
    """

    # Jellyfin API endpoint to refresh metadata for a specific item
    url = f"{server_ip}/Items/{itemid}/Refresh"

    # Headers to include the Jellyfin token for authentication
    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_token}",
    }
    
    # Query parameters
    params = {
        "metadataRefreshMode": "FullRefresh",
    }

    # # Cheap way to get the admin user id, and save it for later use.
    # users = json.loads(requests.get(f"{server_ip}/Users", headers=headers).content)
    # jellyfin_admin = get_jellyfin_admin(users)

    # response = requests.get(f"{server_ip}/Users/{jellyfin_admin}/Items/{itemid}/Refresh", headers=headers)

    # Sending the PUT request to refresh metadata
    response = requests.post(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 204:
        logging.info("Metadata refresh queued successfully.")
    else:
        raise Exception(f"Error refreshing metadata: {response.status_code}")


def get_jellyfin_file_name(item_id: str, jellyfin_url: str, jellyfin_token: str) -> str:
    """Gets the full path to a file from the Jellyfin server.

    Args:
        jellyfin_url: The URL of the Jellyfin server.
        jellyfin_token: The Jellyfin token.
        item_id: The ID of the item in the Jellyfin library.

    Returns:
        The full path to the file.
    """

    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_token}",
    }

    # Cheap way to get the admin user id, and save it for later use.
    users = json.loads(requests.get(f"{jellyfin_url}/Users", headers=headers).content)
    jellyfin_admin = get_jellyfin_admin(users)

    response = requests.get(f"{jellyfin_url}/Users/{jellyfin_admin}/Items/{item_id}", headers=headers)

    if response.status_code == 200:
        file_name = json.loads(response.content)['Path']
        return file_name
    else:
        raise Exception(f"Error: {response.status_code}")

def get_jellyfin_admin(users):
    for user in users:
        if user["Policy"]["IsAdministrator"]:
            return user["Id"]
            
    raise Exception("Unable to find administrator user in Jellyfin")



def has_audio(file_path, open_file = True, check_extensions = True):
    try:
        # if not is_valid_path(file_path):
        #     return False
        if check_extensions:
            if not (has_video_extension(file_path) or  has_audio_extension(file_path)):
                # logging.debug(f"{file_path} is an not a video or audio file, skipping processing. skipping processing")
                return False

        if open_file:
            with av.open(file_path) as container:
                # Check for an audio stream and ensure it has a valid codec
                for stream in container.streams:
                    if stream.type == 'audio':
                        # Check if the stream has a codec and if it is valid
                        if stream.codec_context and stream.codec_context.name != 'none':
                            return True
                        else:
                            logging.debug(f"Unsupported or missing codec for audio stream in {file_path}")
                return False
    except FileNotFoundError:
        logging.warning("The file was not found.")
    except PermissionError:
        logging.warning("You don't have permission to access this file.")
    except OSError as e:
        logging.warning(f"An unexpected OS error occurred: {e}")
    except av.InvalidDataError as e:
        logging.warning(f"Invalid data error with {file_path}: {e}")
    except av.AVError as e:
        logging.warning(f"Error processing file with {file_path}: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error of type {type(e).__name__}: {e}")
    return False

def is_valid_path(file_path):
    # Check if the path is a file
    if not os.path.isfile(file_path):
        # If it's not a file, check if it's a directory
        if not os.path.isdir(file_path):
            logging.warning(f"{file_path} is neither a file nor a directory. Are your volumes correct?")
            if not os.access(file_path, os.R_OK):
                logging.warning(f"{file_path} is not readable. Check the file permissions.")
            if not os.path.exists(file_path):
                logging.warning(f"{file_path} does not exist. Check the path.")
            return False
        else:
            logging.debug(f"{file_path} is a directory, skipping processing as a file.")
            return False
    else:
        return True    

def has_video_extension(file_name):
    file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension
    return file_extension in VIDEO_EXTENSIONS

def has_audio_extension(file_name):
    file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension
    return file_extension in AUDIO_EXTENSIONS


def path_mapping(fullpath):
    if use_path_mapping:
        logging.debug("Updated path: " + fullpath.replace(path_mapping_from, path_mapping_to))
        return fullpath.replace(path_mapping_from, path_mapping_to)
    return fullpath

if monitor:
    # Define a handler class that will process new files
    class NewFileHandler(FileSystemEventHandler):
        def __init__(self, observed_path):
            self.observed_path = observed_path  # Store the path observed by the observer
            super().__init__()
            
        def create_subtitle(self, file_path):
            logging.info(f"[Monitor] File: {file_path} was added to the queue.")
            gen_subtitles_queue(file_path, transcribe_or_translate)
        def on_created(self, event):
            logging.debug(f"[Monitor] File created: {event.src_path}, handling it.")
            if self.will_handle(event):
                self.create_subtitle(path_mapping(event.src_path))
        def on_modified(self, event):
            logging.debug(f"[Monitor] File modified: {event.src_path}, ignoring it.")
            # Let's not do this, because it might trigger something that's already being processed
            
            
        def will_handle(self, event):
            if not event.is_directory:
                file_path = event.src_path
                file_name = os.path.basename(file_path)
                # Exclude files in ignore_files based on file name (only the file name, not the full path)
                if any(ignore_folders):
                    event_folder = os.path.dirname(event.src_path)  # Get the directory of the created/modified file
                    if self._is_subfolder(event_folder) and self._is_ignored_folder(event_folder):
                        logging.info(f"[Monitor] Excluding folder: {event_folder} due to ignore_folders criteria.")
                        return False
                if any(ignore_files):
                    if any(keyword in file_name.lower() for keyword in ignore_files):
                        logging.info(f"[Monitor] Excluding file: {file_path} due to ignore_files criteria.")
                        return False
                
                if has_audio(file_path):
                    return True
            return False
        
        def _is_subfolder(self, event_folder):
            # This method checks if event_folder is a subfolder of the observed path
            return event_folder.startswith(self.observed_path)

        def _is_ignored_folder(self, event_folder):
            # This method checks if any folder in the path is in the ignore_folders list
            event_folders = event_folder.split(os.sep)  # Split into individual folder names
            for folder in event_folders:
                if folder.lower() in ignore_folders:
                    return True
            return False
                    

def transcribe_existing(transcribe_folders, forceLanguage : LanguageCode | None = None, n_threads = 4):
    folders = transcribe_folders.split("|")
    logging.info("Starting to search folders to see if we need to create subtitles.")
    logging.debug(f"The folders are: {folders}")
    
    if not folders:
        logging.warning("No folders provided. Skipping.")
        return None
    
    # First collect ALL paths from ALL folders
    all_paths = []
    for path in folders:
        path = path_mapping(path)
        if os.path.exists(path):
            logging.info(f"Collecting paths from: {path}")
            paths = collect_all_paths(path, set(ignore_folders))
            if not paths:
                logging.warning(f"No valid paths found in {path}.")
                continue
            all_paths.extend(paths)
        else:
            logging.warning(f"Path {path} does not exist.")
    
    # TODO maybe add a check here to see if path has files in them or not. if not warn the user that the path is empty and theres nothing to do there
    
    if not all_paths:
        logging.warning("No valid paths to process")
        return None

    

    shuffle_paths = True
    if shuffle_paths:
        random.shuffle(all_paths)

    # Now distribute the paths to the threads
    distributed_paths = [[] for _ in range(n_threads)]
    for i, file_path in enumerate(all_paths):
        distributed_paths[i % n_threads].append(file_path)
    thread = threading.Thread(target=process_paths_in_groups, args=(distributed_paths,), name="process_groups_threads_owner", daemon=True)
    thread.start()
    return thread

def collect_all_paths(root_path: str, ignore_folders: set) -> list[str]:
    """Collect all folder paths from a root path."""
    paths = []
    try:
        # Handle single file case
        if os.path.isfile(root_path):
            return [root_path]
            
        # Add the root path itself
        paths.append(root_path)
        
        # Collect all subfolder paths
        for root, dirs, _ in os.walk(root_path):
            # Filter out ignored folders
            if ignore_folders:
                dirs[:] = [d for d in dirs if d.lower() not in ignore_folders]
                
            # Add full path of each subfolder
            paths.extend(os.path.join(root, d) for d in dirs)
                
    except Exception as e:
        logging.error(f"Error collecting paths from {root_path}: {str(e)}")
    
    return paths

def is_valid_audio_file(file_path, check_extensions = True):
    # Check if the file exists
    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist.")
        return False
    # Check if it's a file
    if not os.path.isfile(file_path):
        logging.warning(f"{file_path} is not a file.")
        return False
    return has_audio(file_path, check_extensions = check_extensions)

def is_ignored_file(file_name, ignore_files):
    """Check if the file name matches any ignore pattern."""
    return ignore_files and any(keyword in file_name.lower() for keyword in ignore_files)

def process_file(file_path, ignore_files):
    """Check if this file is a candidate to have subtitles created for it. If so, add it to the queue."""
    file_name = os.path.basename(file_path)
    
    if not (has_video_extension(file_path) or  has_audio_extension(file_path)):
        return
    
    if is_ignored_file(file_name, ignore_files):
        logger.debug(f"Excluding file: {file_path} due to ignore_files criteria.")
        return
    
    #Maybe check first if if has an video/audio file exension before checking this list and then check if it has audio with the av probe. Not sure what would be better
    
    if file_path in files_to_skip_list:
        logger.debug(f"Excluding file: {file_path} due to skip list.")
        return

    if is_valid_audio_file(file_path, check_extensions = False):
        logging.debug(f"Processing {file_path} in thread {threading.current_thread().name}")
        gen_subtitles_queue(path_mapping(file_path), transcribe_or_translate)
    else:
        logging.debug(f"Not an audio file: {file_path}")

def process_folder(file_path, ignore_files):
    """Process all files in the folder."""
    for file_name in os.listdir(file_path):
        current_file_path = os.path.join(file_path, file_name)
        if os.path.isdir(current_file_path):
            #We are not interested in folders, just files
            continue
        process_file(current_file_path, ignore_files)

def process_paths(file_paths, ignore_files=ignore_files):
    """This is a thread that processes a list of paths. These are all folders unless TRANSCRIBE_FOLDERS has a value of a file. If it is a folder it will only check for the files in that folder, but not in the subfolders, because that subfolder will be a different path in the list (if not excluded)"""
    for file_path in file_paths:
        if os.path.isdir(file_path):
            logging.debug(f"Processing folder: {file_path}")
            process_folder(file_path, ignore_files)
        elif os.path.isfile(file_path):
            logging.debug(f"Processing file: {file_path}")
            process_file(file_path, ignore_files)
        elif not os.access(file_path, os.R_OK):
                logging.warning(f"{file_path} is not readable. Check the file permissions.")
        elif not os.path.exists(file_path):
                logging.warning(f"{file_path} does not exist. Check the path.")
        else:
            logging.warning(f"Invalid path: {file_path}")
    
def process_paths_in_groups(distributed_paths):
    # Start a thread for each group
    threads = []

    logging.debug(f"Starting {len(distributed_paths)} groups of threads to process paths.")
    for i, group in enumerate(distributed_paths):
        thread_name = f"process_group-{i+1}"
        thread = threading.Thread(target=process_paths, args=(group,), name=thread_name)
        threads.append(thread)
        thread.start()
        logging.debug(f"Started thread for group {i+1} with {len(group)} files.")
    
    #wait for these traids to finish
    for thread in threads:
        thread.join()   
    
    logging.debug("All threads have finished processing paths.")
    finished_processing_paths_event.set()  # Signal that this group of threads has finished

        
def monitor_folders(paths):
     # Set up the observer to watch for new files
    observer = Observer()
    for path in  paths.split("|"):
        logging.debug(f"Monitoring in: {path}")
        if os.path.isdir(path):
            handler = NewFileHandler(path)
            observer.schedule(handler, path, recursive=True)
    observer.start()
    logging.info("Finished searching and queueing files for transcription. Now watching for new files.")

# Uvicorn server runner
def run_uvicorn():
    logging.info("Starting webhook server")
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), reload=reload_script_on_change, use_colors=True)

if __name__ == "__main__":
    import uvicorn
    logging.info(f"Subgen v{subgen_version}-MUIJSE")

    
    logging.info("Starting Subgen!")
    logging.info(f"Transcriptions are limited to running {str(concurrent_transcriptions)} at a time")
    logging.info(f"Running {str(whisper_threads)} threads per transcription")
    logging.info(f"Using {transcribe_device} to encode")
    logging.info(f"Using faster-whisper {whisper_model.split('.')[0]}")
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

    load_queue()
    
    add_files_to_queue_threads = None
    
    if transcribe_folders:
        logging.info("Transcibing folders: {transcribe_folders}")
        add_files_to_queue_threads = transcribe_existing(transcribe_folders)
        
        if monitor:
            logging.info("Starting to monitor folders for new files.")
            monitor_folders(transcribe_folders)
    
    logging.info("Starting transcription workers")
    transcription_threads = start_transcription_workers()
    
    server_thread = None
    if use_webhooks:
            # Run uvicorn in a separate thread
        server_thread = threading.Thread(target=run_uvicorn, daemon=True, name="uvicorn_server")
        server_thread.start()

    subtitle_tags = load_subtitle_tag_config(os.getenv('SUBTITLE_TAGS', ''), whisper_model=whisper_model.split('.')[0], subtitle_language_naming_type=subtitle_language_naming_type, language=namesublang if namesublang else (LanguageCode.ENGLISH if transcribe_or_translate == "translate" else force_detected_language_to))


    logging.info("Setup complete!")
    
    current_subtitle_tags = get_updated_subtitle_tags(subtitle_tags)
    example_subtitle_name = name_subtitle("Example Movie (2025)", tags = current_subtitle_tags, subtitle_tag_delimiter=subtitle_tag_delimiter)
    logging.info(f"example subtitle file from subgen: {example_subtitle_name}")
    
    # Join all threads 
    if add_files_to_queue_threads:
        add_files_to_queue_threads.join()
    
    for thread in transcription_threads:
        thread.join()
        logging.debug(f"Joined thread: {thread.name}")
        
    if server_thread:
        server_thread.join()
        logging.debug("Joined server thread")
        
    logging.debug("All transcription threads have finished")
    logging.debug("All threads have finished")
    
    # program will end here if monitor or webhooks is not enabled and all threads have finished. This means that all desired files should have been transcribed.
    logging.info("All done!")
    sys.exit(0)
