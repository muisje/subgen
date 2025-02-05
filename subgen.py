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
from typing import Union, List, Tuple, Optional, Any, Dict
from fastapi import FastAPI, File, UploadFile, Query, Header, Body, Form, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import dataclass
import signal
import pickle
from unique_queue import UniqueLifoQueue
from requests.exceptions import RequestException
from name_subtitle import SubtitleTagType, FileWriteBehavior, name_subtitle
from config import load_subtitle_tag_config
import unicodedata
from subtitle_event import SubtitleEventHandler, SubtitleEventConfig
from enum import Enum
from load_env_variables import load_env_variables
from task_tracker import TaskTracker
from uuid import UUID
from overwrite_video_metadata import over_write_audio_language_metadata

subgen_version = '2025.01.28'


load_env_variables()

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
webhookport = int(os.getenv('WEBHOOKPORT', 9000))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', True))
use_path_mapping = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from = os.getenv('PATH_MAPPING_FROM', r'/tv')
path_mapping_to = os.getenv('PATH_MAPPING_TO', r'/Volumes/TV')
model_location = os.getenv('MODEL_PATH', './models')
monitor = convert_to_bool(os.getenv('MONITOR', False))
transcribe_folders = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_existing_in_transcribe_folders = convert_to_bool(os.getenv('TRANSCRIBE_EXISTING_IN_TRANSCRIBE_FOLDERS', True))
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
force_detected_language_to = LanguageCode.from_iso_639_2(os.getenv('FORCE_DETECTED_LANGUAGE_TO', ''))
preferred_audio_languages = ( 
    [LanguageCode.from_iso_639_2(code) for code in os.getenv('PREFERRED_AUDIO_LANGUAGES', 'eng').split("|")]
    if os.getenv('PREFERRED_AUDIO_LANGUAGES')
    else []
) # in order of preferrence
subtitle_language_naming_type = os.getenv('SUBTITLE_LANGUAGE_NAMING_TYPE', 'ISO_639_2_B')
should_whiser_detect_audio_language = convert_to_bool(os.getenv('SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE', False))
ignore_folders = [folder.strip().lower() for folder in os.getenv('IGNORE_FOLDERS', '').split("|")]
ignore_files = [file.strip().lower() for file in os.getenv('IGNORE_FILES', '').split("|")]
should_write_detected_language = convert_to_bool(os.getenv('SHOULD_WRITE_DETECTED_LANGUAGE', False))
jellyfin_stream_subtile = convert_to_bool(os.getenv('JELLYFIN_STREAM_SUBTITLE', True))
word_highlight_color = os.getenv('WORD_HIGHLIGHT_COLOR', 'FFFF99')
detect_language_in_filename = convert_to_bool(os.getenv('DETECT_LANGUAGE_IN_FILENAME', False))
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
jellyseerr_api_key = os.getenv('JELLYSEERR_API_KEY', None)
jellyseerr_base_url = os.getenv('JELLYSEERR_BASE_URL', None)
do_not_transcribe = os.getenv('DO_NOT_TRANSCRIBE', False)

only_skip_if_subgen_subtitle = convert_to_bool(os.getenv('ONLY_SKIP_IF_SUBGEN_SUBTITLE', False))

skip_unknown_language = convert_to_bool(os.getenv('SKIP_UNKNOWN_LANGUAGE', False))
skip_if_language_is_not_set_but_subtitles_exist = convert_to_bool(os.getenv('SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST', False))
skip_if_language_is_not_set_but_subtitles_exist_in_prefered_language = convert_to_bool(os.getenv('SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST_IN_PREFERRED_LANGUAGE', False))
skip_if_preferred_audio_language_sub_already_exist = convert_to_bool(os.getenv('SKIP_IF_PREFERRED_AUDIO_LANGUAGE_SUB_ALREADY_EXIST', False))
skip_if_any_subtitles_exist = convert_to_bool(os.getenv('SKIP_IF_SUBTITLES_EXIST', False))
skip_if_to_transcribe_sub_already_exist = convert_to_bool(os.getenv('SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST', True))

limit_to_preferred_audio_languages = convert_to_bool(os.getenv('LIMIT_TO_PREFERRED_AUDIO_LANGUAGE', False))
skip_if_audio_track_is_in_list = (
    [LanguageCode.from_iso_639_2(code) for code in os.getenv('SKIP_IF_AUDIO_TRACK_IS', '').split("|")]
    if os.getenv('SKIP_IF_AUDIO_TRACK_IS')
    else []
)
do_not_transcribe_audio_languages = (
    [LanguageCode.from_iso_639_2(code) for code in os.getenv('DO_NOT_TRANSCRIBE_AUDIO_LANGUAGES', '').split("|")]
    if os.getenv('DO_NOT_TRANSCRIBE_AUDIO_LANGUAGES')
    else []
)

#TODO Do not use these and put a warning when using them
assume_no_language_in_subtitle_is_audio_language = convert_to_bool(os.getenv('ASSUME_NO_LANGUAGE_IN_SUBTITLE_IS_AUDIO_LANGUAGE', False))
assume_default_in_subtitle_is_audio_language = convert_to_bool(os.getenv('ASSUME_DEFAULT_IN_SUBTITLE_IS_AUDIO_LANGUAGE', False))
skipifexternalsub = convert_to_bool(os.getenv('SKIPIFEXTERNALSUB', False))
skipifinternalsublang = LanguageCode.from_iso_639_2(os.getenv('SKIPIFINTERNALSUBLANG', ''))
namesublang = os.getenv('NAMESUBLANG', '')
skip_lang_codes_list = (
    [LanguageCode.from_iso_639_2(code) for code in os.getenv("SKIP_LANG_CODES", "").split("|")]
        if os.getenv('SKIP_LANG_CODES')
    else []
)

if not subtitle_tag_delimiter or subtitle_tag_delimiter == "":
    subtitle_tag_delimiter = ' '



#TODO validate config on errors

shutdown_event = threading.Event()


model_lock = threading.Lock() #To prevent loading the multiple times at the same time which causes an error



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

SUBTITLE_EXTENSIONS = ['.srt', '.vtt', '.sub', '.ass', '.ssa', '.idx', '.sbv', '.pgs', '.ttml', '.lrc']    

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
                detect_language_task(task['path'], task.get('skip_skip_check', False), task.get('event_handler', None))
                task_queue.task_done()  # Mark task as done if processing succeeded
            else:
                gen_subtitles(task['path'], task['transcribe_or_translate'], task['force_language'], task.get('event_handler', None))
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
    name: str
    jellyfin_id: str
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
    
    handle_issue_notification_types = ["ISSUE_CREATED", "ISSUE_REOPENED"]
    
    if notification_type not in handle_issue_notification_types or not issue_type == "SUBTITLES":
        error_message = f"Invalid notification type or issue type: notification_type in {handle_issue_notification_types}, issue_type={issue_type}. Expected ISSUE_CREATED and SUBTITLES."
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
    
    if (jellyseer_transcribe_keyword or jellyseer_translate_keyword):  
        if jellyseer_transcribe_keyword and jellyseer_transcribe_keyword in message:
            should_transcribe_or_translate = "transcribe"
        elif jellyseer_translate_keyword and jellyseer_translate_keyword in message:
            should_transcribe_or_translate = "translate"
        else:
            #Did not find required keyword in message
            message = f"Did not find required keyword {jellyseer_transcribe_keyword or jellyseer_translate_keyword} in message: {message}. Subgen will ignore this request"
            logging.debug("[jellyseerr-webhook] {message}")
            jellyseerr_write_comment(issue_id, message, jellyseerr_api_key, jellyseerr_base_url)
            return
     
    
    jellyseer_force_language = LanguageCode.NONE
    
    for word in split_words(message):
        if len(word) > 2:
            if LanguageCode.is_valid_language(word):
                jellyseer_force_language = LanguageCode.from_string(word)
                break
        
    
        
    #TODO maybe make the default behaviour to do not skip skip check and only skip skip check when a keyword is present like "Force"
    initial_message = ""
            
    if jellyseerr_base_url and jellyseerr_api_key:
        if notification_type == "ISSUE_REOPENED":
            initial_message = f"Issue reopened by {reported_by_username}, will handle it again"
        else:
            initial_message = f"Received new request to {should_transcribe_or_translate} subtitles from {reported_by_username}"
            
        if jellyseer_force_language != LanguageCode.NONE:
            initial_message += f" in {jellyseer_force_language}\n\n"
        else:
            initial_message += "\n\n"
            
        logging.debug("[jellyseerr-webhook] {message}")
        # jellyseerr_write_comment(issue_id, message, jellyseerr_api_key, jellyseerr_base_url)
    else:
        logging.warning("[jellyseerr] API key or base url not set. Will not write comments.")
        
    
    
    initial_comment_id = jellyseerr_overwrite_comment(issue_id, initial_message, jellyseerr_api_key, jellyseerr_base_url).get("comment_id")
    

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
    
    task_tracker = TaskTracker()
    task_tracker_id = task_tracker.track_new_tasks()
    
    
    send_feedback_to_jellyseerr = jellyseerr_api_key and jellyseerr_base_url
    
    #TODO send response when not fidning movie or series and ask for a tmdbid, tvdbid, imdbid, jellyfin_id or alternative title or fallback to folder filenamef
    #TODO handle the case where there are multiple versions of the same movie or episode (when one episode is requested)
    
    items_added_to_queue = [] #just a list to collect everything to print it in one message once all items are added to the queue of this request. Then one comment will be written with all the items in this list.
    
    if is_movie:
        movie = get_movie_from_jellyfin(subject, tmdb_id, jellyfinserver, jellyfintoken)
        if movie:
            logging.info(f"Found movie: {movie}")
            if jellyseer_force_language != LanguageCode.NONE:
                if len(movie.audio_languages) == 1:
                    jellyseer_force_language = movie.audio_languages[0]
                else:
                    jellyseer_force_language = next((language for language in movie.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
            is_added_to_queue = gen_subtitles_queue_jellyseerr(movie, transcribe_or_translate, issue_id, force_language=jellyseer_force_language)
            if is_added_to_queue:
                items_added_to_queue.append(f"{movie.name} in {jellyseer_force_language} language")
        else:
            logging.warning(f"Did not find movie for {subject} on jellyfin.")
            if send_feedback_to_jellyseerr:
                jellyseerr_write_comment(issue_id, "Did not find movie.", jellyseerr_api_key, jellyseerr_base_url)
            
        return
    elif is_series:
        if len(payload.extra) > 0: # Empty if all seasons for tv
            season_nr = next((int(item['value']) for item in payload.extra if item['name'] == 'Affected Season'), None) 
            episode_nr = next((int(item['value']) for item in payload.extra if item['name'] == 'Affected Episode'), None)   # None if all episodes for season
             
            # if series check if one episode or one season or all seasons
            
            if season_nr and episode_nr:
                # Single episodegen_subtitles_queue_jellyseerr
                episode = get_series_episode_path_from_jellyfin(subject, tmdb_id, tvdb_id, season_nr, episode_nr, jellyfinserver, jellyfintoken)
                if episode:
                    logging.info(f"Found episode: {episode}")
                    if jellyseer_force_language == LanguageCode.NONE:
                        if len(episode.audio_languages) == 1:
                            jellyseer_force_language = episode.audio_languages[0]
                        else:
                            jellyseer_force_language = next((language for language in episode.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
                    is_added_to_queue = gen_subtitles_queue_jellyseerr(episode, should_transcribe_or_translate, issue_id, force_language=jellyseer_force_language)
                    if is_added_to_queue:
                        items_added_to_queue.append(f"{episode.name} in {jellyseer_force_language} language")
                else:
                    logging.warning(f"Did not find episode {episode_nr} of season {season_nr} for {subject} on jellyfin.")
                    if send_feedback_to_jellyseerr:
                        jellyseerr_write_comment(issue_id, f"Did not find episode {episode_nr} of season {season_nr}", jellyseerr_api_key, jellyseerr_base_url)
            
            elif season_nr:
                # Full season
                episodes = get_series_season_episodes_paths_from_jellyfin(subject, season_nr, tmdb_id, tvdb_id, jellyfinserver, jellyfintoken)
                if episodes:
                    logging.info(f"Found {len(episodes)} episodes for season {season_nr} of {subject}")
                    for episode in episodes:
                        logging.info(f"Handling episode: {episode}")
                        if jellyseer_force_language == LanguageCode.NONE:
                            if len(episode.audio_languages) == 1:
                                jellyseer_force_language = episode.audio_languages[0]
                            else:
                                jellyseer_force_language = next((language for language in episode.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
                        is_added_to_queue = gen_subtitles_queue_jellyseerr(episode, should_transcribe_or_translate, issue_id, task_tracker_id=task_tracker_id, force_language=jellyseer_force_language)
                        if is_added_to_queue:
                            items_added_to_queue.append(f"{episode.name} in {jellyseer_force_language} language")
                else:
                    logging.warning(f"Did not find any episodes for season {season_nr} of {subject} on jellyfin.")
                    if send_feedback_to_jellyseerr:
                        jellyseerr_write_comment(issue_id, f"Did not find any episodes for season {season_nr}", jellyseerr_api_key, jellyseerr_base_url)
            else:
                logging.warning(f"Expected at least Affected Season in payload extra when extra is not empty. Instead got this: {payload.extra}")
                if send_feedback_to_jellyseerr:
                    jellyseerr_write_comment(issue_id, "Expected at least Affected Season in payload extra when extra is not empty.", jellyseerr_api_key, jellyseerr_base_url)
        
        else:
            # Full series
            episodes = get_series_episodes_paths_from_jellyfin(subject, tmdb_id, tvdb_id, jellyfinserver, jellyfintoken)
            if episodes:
                logging.info(f"Found {len(episodes)} episodes for {subject}")
                for episode in episodes:
                    logging.info(f"Handling episode: {episode}")
                    if jellyseer_force_language == LanguageCode.NONE:
                        if len(episode.audio_languages) == 1:
                            jellyseer_force_language = episode.audio_languages[0]
                        else:
                            jellyseer_force_language = next((language for language in episode.audio_languages if language in preferred_audio_languages), LanguageCode.NONE)
                    is_added_to_queue = gen_subtitles_queue_jellyseerr(episode, should_transcribe_or_translate, issue_id, task_tracker_id=task_tracker_id, force_language=jellyseer_force_language)
                    if is_added_to_queue:
                        items_added_to_queue.append(f"{episode.name} in {jellyseer_force_language} language")
            else:
                logging.warning(f"Did not find any episodes for {subject}")
                if send_feedback_to_jellyseerr:
                    jellyseerr_write_comment(issue_id, "Did not find any episodes on jellyfin", jellyseerr_api_key, jellyseerr_base_url)
                

    initial_message += f"Added {len(items_added_to_queue)} to queue:\n\no  " + "\n\no  ".join(items_added_to_queue) #TODO r \n is not handled well and comment is cutoff when starting a new line with a non letter character
    #TODO maybe have a list of not added if any
    
    
    print(initial_message)
    
    jellyseerr_overwrite_comment(issue_id, initial_message, jellyseerr_api_key, jellyseerr_base_url, comment_id=initial_comment_id)
    # can get this from jellyfin api
    
    # Other usefull info maybe
    # "MovieCount": 0,
    # "SeriesCount": 0,
    # "ProgramCount": 0,
    # "EpisodeCount": 0,
    # "EpisodeTitle": "string",    


    logging.info(f"Finished processing jellyseerr webhook for {subject}")

    return {"status": "success", "message": "Webhook processed successfully."}
    
def gen_subtitles_queue_jellyseerr(media: MediaInfo, 
                                   transcription_type: str, 
                                   issue_id,
                                   task_tracker_id: Optional[UUID] = None,
                                   force_language: LanguageCode = LanguageCode.NONE,
                                   ) -> None:
    # Configure the event handler
    subtitle_event_handler = SubtitleEventHandler(
        shared_args = {
            'jellyseerr_api_key': jellyseerr_api_key,
            'jellyseerr_base_url': jellyseerr_base_url,
            'jellyseerr_issue_id': issue_id,
            'task_tracker_id': task_tracker_id
        },
        on_start = SubtitleEventConfig(jellyseerr_overwrite_comment, save_shared_state=True),
        # on_update = jellyseerr_write_comment,
        on_detect_language = jellyseerr_write_comment,
        on_detect_language_failed = (
            jellyseerr_write_comment, 
            {'message': "Failed to detect language. Consider specifying the language in the description of \"What's wrong? *\"."},
            ["specific_args", "shared_args"]
        ),
        on_progress = jellyseerr_overwrite_comment,
        on_error = [jellyseerr_write_comment, jellyseerr_mark_resolved],
        on_skip = jellyseerr_write_comment,
        on_complete = [
            complete_task,
            jellyseerr_overwrite_comment_on_complete,
            jellyseerr_mark_resolved,
            (
                refresh_jellyfin_metadata, 
                {'jellyfin_item_id': media.jellyfin_id, 'jellyfin_api_key': jellyfintoken, 'jellyfin_base_url': jellyfinserver}
            )
        ]
    )
    

    is_added_to_a_queue = gen_subtitles_queue(path_mapping(media.path), transcription_type, force_language, True, subtitle_event_handler)
    
    if is_added_to_a_queue:
        if task_tracker_id:
            TaskTracker().add_task(task_tracker_id)
        # jellyseerr_write_comment(issue_id, f"{media.name} in {force_language} has been added to queue", jellyseerr_api_key, jellyseerr_base_url)
        return True
    
    logging.info(f"[jellyseerr] {media.name} in {force_language} has not been added to queue.")
    jellyseerr_write_comment(issue_id, f"{media.name} in {force_language} has not been added to queue", jellyseerr_api_key, jellyseerr_base_url)
    return False

def complete_task(task_tracker_id: Optional[UUID] = None):
    if task_tracker_id:
        TaskTracker().task_finished(task_tracker_id)



def jellyseerr_overwrite_comment_on_complete(jellyseerr_issue_id: int, message: str, jellyseerr_api_key: str, jellyseerr_base_url: str = "localhost:5055", task_tracker_id: Optional[UUID] = None, comment_id: int | None = None):
        edited_message = message
        if task_tracker_id:
            issue_is_resolved = TaskTracker().did_all_tasks_finish(task_tracker_id)
            if issue_is_resolved:
                edited_message = f"{message}\n\nAll subtitles are generated. Will close issue."
            else:
                remaining_task = TaskTracker().tasks_remaining(task_tracker_id)
                edited_message = f"{message}\n\n{remaining_task} subtitles remaining."
        else:
            edited_message = f"{message}\n\n This was the only subtitle of this issue. Will close issue."
        jellyseerr_overwrite_comment(jellyseerr_issue_id, edited_message, jellyseerr_api_key, jellyseerr_base_url, comment_id)


def jellyseerr_write_comment(jellyseerr_issue_id: int, message: str, jellyseerr_api_key: str, jellyseerr_base_url: str = "localhost:5055"):
    """
    Sends a comment to a specific issue in Jellyseerr/Overseerr.

    :param jellyseerr_issue_id: The ID of the issue to comment on.
    :param message: The comment message.
    :param jellyseerr_api_key: The API key for authentication.
    :param jellyseerr_base_url: The base URL of the Jellyseerr/Overseerr server.
    :return: A success message or error details.
    """
    if not (jellyseerr_api_key and jellyseerr_base_url):
        logging.warning("No Jellyseerr/Overseerr API key or base URL provided.")
        return
    
    # logging.debug(f"Sending comment to Jellyseerr/Overseerr: {message}")
    url = f"{jellyseerr_base_url}/api/v1/issue/{jellyseerr_issue_id}/comment"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Api-Key": jellyseerr_api_key
    }
    data = {"message": message}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises an error for 4xx and 5xx status codes
        
        # Log the JSON response content
        response_json = response.json()
        
        logging.debug(f"Wrote comment with id: {response_json['id']}")
        
        logging.info(f"Comment successfully sent to issue {jellyseerr_issue_id}")
        return {"success": True, "message": "Comment sent successfully", "response": response_json}
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending comment to issue {jellyseerr_issue_id}: {str(e)}")
        return {"success": False, "error": str(e)}


def jellyseerr_overwrite_comment(jellyseerr_issue_id: int, message: str, jellyseerr_api_key: str, jellyseerr_base_url: str = "localhost:5055", comment_id: int | None = None):
    """
    Sends a comment to a specific issue in Jellyseerr/Overseerr.

    :param jellyseerr_issue_id: The ID of the issue to comment on.
    :param message: The comment message.
    :param jellyseerr_api_key: The API key for authentication.
    :param jellyseerr_base_url: The base URL of the Jellyseerr/Overseerr server.
    :return: A success message or error details.
    """
    if not (jellyseerr_api_key and jellyseerr_base_url):
        logging.warning("No Jellyseerr/Overseerr API key or base URL provided.")
        return
    
    # logging.debug(f"Sending comment to Jellyseerr/Overseerr: {message}")
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Api-Key": jellyseerr_api_key
    }
    data = {"message": message}

    try:

        if not comment_id:
            url = f"{jellyseerr_base_url}/api/v1/issue/{jellyseerr_issue_id}/comment"
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raises an error for 4xx and 5xx status codes
        
            response_json = response.json()
            # Iterate through the comments to find the matching message
            for comment in response_json['comments']:
                if comment['message'] == message:
                    comment_id = comment['id']
                    break
            else:
                logging.warning(f"No message found matching '{message}'")
            logging.debug(f"Wrote new comment {message} with id: {comment_id}")
        else:
            url = f"{jellyseerr_base_url}/api/v1/issueComment/{comment_id}"
            response = requests.put(url, json=data, headers=headers)
            response.raise_for_status()
            logging.debug(f"Overwrote comment {message} with id: {comment_id}")
        
        
        return {"success": True, "comment_id": comment_id}
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending comment to issue {jellyseerr_issue_id}: {str(e)}")
        return {"success": False, "error": str(e)}


def jellyseerr_mark_resolved(jellyseerr_issue_id: int,
                            jellyseerr_api_key: str, 
                            jellyseerr_base_url: str = "localhost:5055",
                            task_tracker_id: Optional[UUID] = None):
    """
    Marks an issue as resolved in Jellyseerr/Overseerr.

    :param jellyseerr_issue_id: The ID of the issue to mark as resolved.
    :param jellyseerr_api_key: The API key for authentication.
    :param jellyseerr_base_url: The base URL of the Jellyseerr/Overseerr server.
    :return: A success message or error details.
    """
    if task_tracker_id:
        issue_is_resolved = TaskTracker().did_all_tasks_finish(task_tracker_id)
        if not issue_is_resolved:
            logger.debug(f"Not marking issue {jellyseerr_issue_id} as resolved because not all tasks are finished.")
            return {"success": False, "message": "Not all tasks are finished."}
        else:
            logger.debug(f"All tasks finished for issue {jellyseerr_issue_id}, marking as resolved.")
    
    if not (jellyseerr_api_key and jellyseerr_base_url):
        logging.warning("No Jellyseerr/Overseerr API key or base URL provided.")
        return
    logging.debug(f"Marking issue {jellyseerr_issue_id} as resolved in Jellyseerr/Overseerr")
    url = f"{jellyseerr_base_url}/api/v1/issue/{jellyseerr_issue_id}/resolved"
    headers = {
        "accept": "application/json",
        "X-Api-Key": jellyseerr_api_key
    }

    try:
        response = requests.post(url, headers=headers, data="")  # Empty body
        response.raise_for_status()  # Raises an error for 4xx and 5xx status codes
        
        # Log the JSON response content
        response_json = response.json()
        # logging.debug(f"Response JSON: {response_json}")
        
        logging.info(f"Issue {jellyseerr_issue_id} marked as resolved successfully")
        return {"success": True, "message": "Issue marked as resolved", "response": response_json}
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error marking issue {jellyseerr_issue_id} as resolved: {str(e)}")
        return {"success": False, "error": str(e)}

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
    
    # print(json.dumps(source, indent=4))
    
    return MediaInfo(
        name=source.get("Name"),
        jellyfin_id=source.get("Id"),
        path=path,
        audio_languages=audio_langs,
        subtitle_languages=subtitle_langs
    )

def _extract_media_info_from_item(item: dict) -> Optional[MediaInfo]:
    """Helper function to extract MediaInfo from an item's media sources."""
    media_sources = item.get("MediaSources", [])
    
    # Try each media source
    for source in media_sources:
        # print(json.dumps(source, indent=4))
        media_info = _extract_media_info_from_source(source)
        if media_info:
            return media_info
    
    # If no media sources worked, try direct path
    path = item.get("Path")
    if path:
        return MediaInfo(
            name=source.get("Name"),
            jellyfin_id=item.get("Id"),
            path=path,
            audio_languages=[],  # No language info available
            subtitle_languages=[]
        )
    
    return None

def _base_jellyfin_search(search_term: str, 
                          include_item_type: str = "Episode", 
                          jellyfinserver: str = "http://localhost:8096", 
                          jellyfintoken: str = "your_token_here", 
                          limit: int = 333 # Don't set too high else Jellyfin will have an internal server error
                          ) -> Optional[List[dict]]:
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
    
    logger.warning(f"Series {series_name} not found in Jellyfin. Did not match anything on tmdbid: {tmdb_id} or tvdbid: {tvdb_id} or name.")
    return None

def get_series_season_episodes_paths_from_jellyfin(series_name: str, season_nr: int, tmdb_id: str, tvdb_id: str,
                                                 jellyfinserver: str = "http://localhost:8096", 
                                                 jellyfintoken: str = "your_token_here") -> List[MediaInfo]:
    """Get all episode MediaInfo objects for a specific season of a series."""
    series_id = get_series_id_from_jellyfin(series_name, tmdb_id, tvdb_id, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not series_id:
        logging.warning(f"Series {series_name} not found in Jellyfin.")
        return []
    
    #TODO filter with this: parentIndexNumber	integer <int32> / Optional filter by parent index number.
    
    #TODO Maybe use parentId	 string <uuid> specify this to localize the search to a specific item or folder. Omit to use the root.
    # parent id would be the season id so TODO get the season id

    items = _base_jellyfin_search(series_name, jellyfinserver=jellyfinserver, jellyfintoken=jellyfintoken)
    if not items:
        logging.warning(f"No episodes found for season {season_nr} of {series_name} with series_id {series_id}")  
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
    
      #TODO filter with this: parentIndexNumber	integer <int32> / Optional filter by parent index number
      #TODO filter also with this: indexNumber
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
        if (NotificationType == "ItemAdded" and procaddedmedia) or (NotificationType == "PlaybackStart" and procmediaonplay):
            logging.debug(f"Adding item from Jellyfin to the queue: {file_path}")
            #TODO refresh jellyfin metadata after transcribing
            gen_subtitles_queue(path_mapping(file_path), transcribe_or_translate)
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

def detect_language_task(path, skip_skip_check = False, event_handler: Optional[SubtitleEventHandler] = None):
    detected_language = LanguageCode.NONE
    global detect_language_length 

    logger.info(f"[Detect language task] Detecting language of file: {path} on the first {detect_language_length} seconds of the file")

    try:
        start_model()
        logging.info("[Detect language task] model started.")
        minimum_language_probability = 0.8
        
        
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
                        succes = over_write_audio_language_metadata(path, {0: detected_language})
                        if succes:
                            logging.info(f"[Detect language task] Wrote audio language metadata: {detected_language} to file: {path}")
                        else:
                            logging.warning(f"[Detect language task] Failed to write audio language metadata to file: {path}")

                    if detected_language in do_not_transcribe_audio_languages:
                        logger.debug(f"[Detect language task] Detected language for stream index {index} is {detected_language.to_name()} which is in do_not_transcribe_audio_languages. Skipping.")
                        detected_language = LanguageCode.NONE
                    elif limit_to_preferred_audio_languages and detected_language not in preferred_audio_languages:
                        logger.debug(f"[Detect language task] Detected language for stream index {index} is {detected_language.to_name()} which is not in preferred_audio_languages. Skipping.")
                        detected_language = LanguageCode.NONE
                    
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
                        language_tracks.append((index, track_language))
                    else:
                        logger.warning(f"[Detect language task] Language detection for stream index {index} returned no language. Skipping.")
                        continue
                    
                except Exception as e:
                    print("An error occurred during language detection for stream index %d:" % index, str(e))
            if len(language_tracks) > 0:
                #First try to find the first preferred audio language if it exists
                for preferred_audio_language in preferred_audio_languages:
                    for index, language in language_tracks:
                        if track_language in do_not_transcribe_audio_languages:
                            logger.debug(f"[Detect language task] Detected language for stream index {index} is {track_language.to_name()} which is in the do_not_transcribe_audio_languages list.")
                            continue
                        if language == preferred_audio_language:
                            detected_language = language
                
                # Else take the first acceptable language
                if not detected_language and not limit_to_preferred_audio_languages:
                    for index, language in language_tracks:
                        if language in do_not_transcribe_audio_languages:
                            continue
                        detected_language = language
                        break
                
                # Could be that it detected multiple languages, but none were acceptable to transcribe
                
                
                if should_write_detected_language:
                    logging.info(f"[Detect language task] Writing {len(language_tracks)} audio languages: {language_tracks} to file: {path}")
                    
                    language_tracks_reindexed = {}
                    for index, (_, language) in enumerate(language_tracks):
                        language_tracks_reindexed[index] = language
                    
                    succes = over_write_audio_language_metadata(path, language_tracks_reindexed)
                    
                    if succes:
                        logging.info(f"[Detect language task] Wrote {len(language_tracks)} audio languages metadata: {language_tracks} to file: {path}")
                    else:
                        logging.warning(f"[Detect language task] Failed to write audio languages metadata to file: {path}")
        if  detected_language:
            logging.info(f"[Detect language task] Detected language of file: {os.path.basename(path)} is: {detected_language.to_name()}. Will add it to the gen subtitles queue")
            if event_handler: 
                event_handler.on_detect_language(message=f"Detected language: {detected_language.to_name()}")
            else:
                logging.warning("SubtitleEventHandler is None")
            gen_subtitles_queue(path, transcribe_or_translate, force_language=detected_language, skip_skip_check=skip_skip_check, event_handler=event_handler, has_whisper_detected_language=True)
        else:
            if event_handler:
                event_handler.on_detect_language_failed()
            logging.warning(f"[Detect language task] Did not detect language of file: {os.path.basename(path)}. Will not add it to the gen subtitles queue.")
            
    except Exception as e:
        logging.error(f"[Detect language task] Error detecting language of file with whisper: {e}")
        


    
    finally:
        logging.info(f"[Detect language task] Detected language of file: {path} is completed.")
        delete_model()
 
        #maybe modify the file to contain detected language so we won't trigger this again
        return

    
    

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
        # metadata_args[f"metadata:s:a:{index}"] = f"title={language_code.to_name()} - Audio Track"
    
    # Set language metadata for all audio streams
    ffmpeg.input(file_path)\
        .output(temp_path,
                codec="copy",  # Copy streams without re-encoding
                map='0',  # Include all streams from input
                **metadata_args)\
        .overwrite_output()\
        .run(quiet=True)
    
    # Replace the original file with the modified file
    os.replace(temp_path, file_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    

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
    with model_lock:
        global model
        if model is None:
            logging.debug("Model was purged, need to re-create")
            model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)


def delete_model():
    gc.collect()
    with model_lock:
        global model
        if clear_vram_on_complete and task_queue.qsize() == 0 and model is not None:
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

def gen_subtitles(file_path: str, 
                  transcription_type: str, 
                  force_language : LanguageCode = LanguageCode.NONE, 
                  event_handler: Optional[SubtitleEventHandler] = None,
                  ) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: str - The path to the video file.
        transcription_type: str - The type of transcription or translation to perform.
        force_language: str - The language to force for transcription or translation. Default is None.
    """

    try:
        logging.info(f"Preparing to transcribe file: {os.path.basename(file_path)} in {force_language if force_language else 'Unkown Language'}")

        if event_handler:
            event_handler.on_start(message=f"Started to {transcription_type}: {os.path.basename(file_path)} in {force_language if force_language else 'Unkown Language'}")
        else:
            logging.warning("SubtitleEventHandler is None")
        
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
            transcription_prompt = custom_model_prompt

        #     #TODO STREAMING MODE WITH JELLYFIN INTEGRATION, 
        #     #TODO OPTIONALLY ADD MESSAGE AFTER CHUNK IN SUBTITLE THAT IT IS STILL GENERATING. AND DELETE THAT MESSAGE WHEN THE NEXT CHUNK IS READY, BUT DO NOT PUT THE MESSSAGE WHEN IT DID ALL CHUNKS
        #     #TODO  Could not initialize NNPACK! Reason: Unsupported hardware.
        #     #TODO Support writing to ass file as well. And have more subtile word highlighting

        
        #Updating subtitle tags to match for this subtitle

        current_subtitle_tags = get_updated_subtitle_tags(subtitle_tags, force_language)
        
        srt_vtt_word_formatting = ""
        if word_level_highlight and word_highlight_color:
            srt_vtt_word_formatting = (f'<font color="#{word_highlight_color}">', '</font>')
         
        subtitle_file_name = None
         
        if should_stream_subtitle:
            logging.info(f"Transcribing in chunks (streaming subtitle): {file_path}")
            subtitle_file_name = stream_subtitle(file_path, 
                                                 current_subtitle_tags, 
                                                 write_intro=append, 
                                                 srt_vtt_word_formatting=srt_vtt_word_formatting, 
                                                 language=force_language, 
                                                 segment_duration=segment_duration, 
                                                 transcription_prompt=transcription_prompt, 
                                                 transcription_type=transcription_type,
                                                 event_handler=event_handler,
                                                 **args)
        
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
                #TODO us optionally denoiser="demucs" and vad=True for music
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
            #TODO maybe multiple output formats?

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        
        if subtitle_file_name:
            message = f"Wrote {os.path.basename(subtitle_file_name)}\n\n Tt took {minutes} minutes and {seconds} seconds to complete."
            logging.info(message)
            # Execute the on_complete action if provided
            if event_handler:
                event_handler.on_complete(message=message) #
        else:
            message = f"Couldn't write subtitle file, it took {minutes} minutes and {seconds} seconds to complete."
            logging.warning(message)
            if event_handler:
                event_handler.on_error(message=message)
                

    except Exception as e:
        logging.warning(f"Error processing or transcribing {file_path} in {force_language}: {e}")
        
        logging.warning(traceback.format_exc())
        if event_handler:
            event_handler.on_error(message=f"Error {transcription_type} {file_path} in {force_language}: {e}")
        else:
            logging.warning("SubtitleEventHandler is None")
        

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

    transcription = model.transcribe_stable(chunk_audio, language=language.to_iso_639_1(), initial_prompt=transcription_prompt, task=transcription_type, **args) # , beam_size=5
                
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

def stream_subtitle(input_file, 
                    subtitle_tags, 
                    srt_vtt_word_formatting = "", 
                    write_intro = True, 
                    language: LanguageCode = LanguageCode.NONE, 
                    segment_duration: int = 60, 
                    transcription_prompt: str = "", 
                    transcription_type: str = "transcribe",
                    event_handler: Optional[SubtitleEventHandler] = None,
                    **args):
    subtitle_file = name_subtitle(input_file, tags=subtitle_tags)
    
    if subtitle_file is None:
        logging.warning(f"Subtitle file could not be written for {input_file}")
        return None
    
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
            transcribed_until = transcribe_offset_seconds + start_time + segment_duration 

            message = f"Subtitle: {os.path.basename(subtitle_file)}\n\n" \
                    f"Progress: {percentage:.2f}% ({segment_index}/{total_segments}) completed. \n" \
                    f"Elapsed time: {sec2vtt(elapsed_time)}. \nEstimated remaining time: {sec2vtt(estimated_time_remaining)}, \n" \
                    f"Total estimated time: {sec2vtt(estimated_total_time)}. \nSpeed: {processing_speed:.2f}x (audio seconds/second).\n\n" \
                    f"Subtitle written until {sec2vtt(transcribed_until)} of {sec2vtt(total_duration)}"
            
            logging.info(message)
            if event_handler:
                event_handler.on_progress(message=message)
        else:
            logging.warning(f"Audio segment from {start_time} to {start_time + segment_duration} not extracted.")
    logging.info(f"Transcription complete. Wrote everything to {subtitle_file}. It took {sec2vtt(elapsed_time)}")
    # delete_model()
    return subtitle_file
            
            

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
    
    # logger.debug(f"choose_transcribe_language({file_path}, {forced_language})")
    
    if force_detected_language_to:
        logger.debug(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")
        return force_detected_language_to
    
    if forced_language:
        logger.debug(f"Language already is set: {forced_language}")   
        return forced_language

    audio_tracks = get_audio_tracks(file_path)
    
    filtered_audio_tracks = list(filter(lambda x: x not in do_not_transcribe_audio_languages, audio_tracks))
    
    
    if preferred_audio_languages:
        found_track_in_language = find_language_audio_track(filtered_audio_tracks, preferred_audio_languages)
        if found_track_in_language:
            language = found_track_in_language
            if language:
                logger.debug(f"Preferred language found: {language}")
                return language
    
    default_language = find_default_audio_track_language(filtered_audio_tracks)
    if default_language:
        logger.debug(f"Default language found: {default_language}")
        if default_language not in do_not_transcribe_audio_languages:
            return default_language
        else:
            logger.debug(f"Default language is in do_not_transcribe_audio_languages: {default_language}")
    
    if detect_language_in_filename:
        language = find_language_in_filename(file_path)
        if language:
            logger.debug(f"Language detected in filename: {language}")
            
            if language not in do_not_transcribe_audio_languages:
                return language
            else:
                logger.debug(f"Language detected in filename is in do_not_transcribe_audio_languages: {language}")
    
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

class SkipReason(Enum):
    # File related
    UNVALID_PATH = "File path is not valid"
    
    # Audio related
    NO_AUDIO = "File has no audio to transcribe"
    UNKNOWN_LANGUAGE = "Audio language is unknown"
    NON_PREFERRED_AUDIO_LANGUAGE = "Audio language is not in preferred languages"
    AUDIO_LANGUAGE_IN_SKIP_LIST = "Audio language is in skip list"
    
    # Subtitle related
    PREFERRED_LANGUAGE_SUB_EXISTS = "Subtitles already exist in preferred language"
    INTERNAL_SUB_EXISTS = "Internal subtitles already exist"
    EXTERNAL_SUB_EXISTS = "External subtitles already exist"
    TARGET_LANGUAGE_SUB_EXISTS = "Subtitles already exist in target language"
    SKIP_LISTED_LANGUAGE_SUB_EXISTS = "Subtitles exist in a skip-listed language"
    GENERIC_SUB_EXISTS = "Subtitles already exist"
    
    # Language preferences
    LANGUAGE_NOT_PREFERRED = "Determined language is not in preferred languages"
    
def gen_subtitles_queue(file_path: str, 
                        transcription_type: str, 
                        force_language: LanguageCode = LanguageCode.NONE, 
                        skip_skip_check = False, 
                        event_handler: Optional[SubtitleEventHandler] = None,
                        has_whisper_detected_language = False
                        ) -> None:
    global task_queue
    
    if not has_whisper_detected_language:
        #Only do this once
        
        if not is_valid_path(file_path):
            message = f"Skipping {file_path}. {SkipReason.UNVALID_PATH.value}."
            logging.warning(message)
            if event_handler:
                event_handler.on_error(message=message)
            return False
            
        
        if not has_audio(file_path):
            message= f"Skipping {os.path.basename(file_path)}. {SkipReason.NO_AUDIO.value}."
            logging.warning(message)
            if event_handler:
                event_handler.on_error(message=message)
            return False
        
        
        if not skip_skip_check:
            # Check if file should be skipped before language selection.
            skip_reason = have_to_skip_before_choosing_language(file_path, force_language)
            if skip_reason:
                message = f"Skipping {os.path.basename(file_path)}. {skip_reason.value}."
                logging.debug(message)
                if skip_list_file_name:
                    write_to_skip_list(skip_list_file_name, file_path)
                if event_handler:
                    event_handler.on_skip(message=message)
                return False

        
        force_language = choose_transcribe_language(file_path, force_language)
        
        # check if we would like to detect audio language in case of no audio language specified. Will return here again with specified language from whisper. #TODO Probably should make a function from what's down here and just call that instead.
        if not force_language and should_whiser_detect_audio_language:
            # make a detect language task
            task_id = { 
                       'path': file_path, 
                       'type': "detect_language", 
                       'skip_skip_check': skip_skip_check, 
                       'event_handler': event_handler,
                    #    'task_tracker_id': task_tracker_id
                       }
            is_added_to_queue = task_queue.put(task_id)
            # logging.debug(f"task_queue.put(task_id)({file_path}, detect_language, skip_skip_check={skip_skip_check})")
            logging.debug(
                f"Language is not determined while deciding if it has to be skipped or not. Will add it to the queue for language detection.\n" \
                f"Added to queue: [detect_language] {os.path.basename(file_path)}"
                )
            return is_added_to_queue
    
    
    # Ending it here if we only want to detect languages. Use it just write them to the metadata.
    if do_not_transcribe:
        return
    
    if not skip_skip_check:
        # Check if the file should be skipped after language selection
        skip_reason = have_to_skip(file_path, force_language)
        if skip_reason:
            message = f"Skipping {os.path.basename(file_path)}. {skip_reason.value}."
            logging.debug(message)
            if skip_list_file_name:
                write_to_skip_list(skip_list_file_name,file_path)
            if event_handler:
                event_handler.on_skip(message=message)
            return False
    
    task = {
        'path': file_path,
        'transcribe_or_translate': transcription_type,
        'force_language': force_language,
        'event_handler': event_handler,
        # 'task_tracker_id': task_tracker_id
    }
    is_added_to_queue = task_queue.put(task)
    
    if is_added_to_queue:
        logging.info(f"Added to queue: [{transcription_type}] {os.path.basename(file_path)} in {force_language}")
        
    return is_added_to_queue

def have_to_skip_before_choosing_language(file_path: str, force_language: LanguageCode) -> Optional[SkipReason]:
    """
    Checks if a file should be skipped before language selection process.
    
    Args:
        file_path: Path to the media file
        force_language: Forced language code if specified
        
    Returns:
        SkipReason if file should be skipped, None otherwise
    """
    # Check forced language against preferences
    if force_language and limit_to_preferred_audio_languages:
        if force_language not in preferred_audio_languages:
            return SkipReason.LANGUAGE_NOT_PREFERRED
    
    # Check for existing subtitles
    
    subtitle_exists, subtitle_langauges = check_subtitles(file_path, LanguageCode.NONE, require_subgen=only_skip_if_subgen_subtitle)
    
    if not subtitle_exists:
        return None
    elif skip_if_any_subtitles_exist:
        return SkipReason.GENERIC_SUB_EXISTS
    elif skip_if_preferred_audio_language_sub_already_exist:
            if any(language in preferred_audio_languages for language in subtitle_langauges):
                return SkipReason.PREFERRED_LANGUAGE_SUB_EXISTS
        
    
    #Removed skipifexternalsub, namesublang, skipifinternalsublang, skip_lang_codes_list
    #TODO implement assume_default_in_subtitle_is_audio_language, assume_no_language_in_subtitle_is_audio_language
    

    return None

def have_to_skip(file_path: str, transcribe_language: LanguageCode) -> Optional[SkipReason]:
    """
    Determines whether subtitle generation should be skipped for a given file
    after language selection.

    Args:
        file_path: The path to the file to check for existing subtitles
        transcribe_language: The language intended for transcription

    Returns:
        SkipReason if subtitle generation should be skipped; otherwise None
    """
    if not transcribe_language:
        if skip_unknown_language:
            return SkipReason.UNKNOWN_LANGUAGE
            
        
        subtitles_exist, subtitle_languages = check_subtitles(file_path, LanguageCode.NONE, require_subgen=only_skip_if_subgen_subtitle)
        
        if not subtitles_exist:
            # Won't skip if no subtitle is found
            return None
        else:
            if skip_if_language_is_not_set_but_subtitles_exist:
                return SkipReason.GENERIC_SUB_EXISTS
        
        
        if skip_if_language_is_not_set_but_subtitles_exist_in_prefered_language:
            if any(language in preferred_audio_languages for language in subtitle_languages):
                return SkipReason.PREFERRED_LANGUAGE_SUB_EXISTS
            
                
        # Check for a reason based on the audio languages
        audio_langs = get_audio_languages(file_path)
        
        if not any(language in preferred_audio_languages for language in audio_langs):
            if limit_to_preferred_audio_languages:
                return SkipReason.NON_PREFERRED_AUDIO_LANGUAGE
                
            if any(lang in skip_if_audio_track_is_in_list for lang in audio_langs):
                return SkipReason.AUDIO_LANGUAGE_IN_SKIP_LIST
    else:
        # Handle cases where transcription language is specified
        if limit_to_preferred_audio_languages and transcribe_language not in preferred_audio_languages:
            return SkipReason.NON_PREFERRED_AUDIO_LANGUAGE
    
        # Check for existing subtitles in target language
        if skip_if_to_transcribe_sub_already_exist:
            subtitles_exist, subtitle_language = check_subtitles(file_path, transcribe_language, require_subgen=only_skip_if_subgen_subtitle)
            
            if not subtitles_exist:
                # Do not skip if no subtitle is found
                return None
            # else the subtitle already exists so we can skip it
            return SkipReason.TARGET_LANGUAGE_SUB_EXISTS

        # Additional language preference checks
        if transcribe_language not in preferred_audio_languages and limit_to_preferred_audio_languages:
            return SkipReason.NON_PREFERRED_AUDIO_LANGUAGE
            
        if transcribe_language in skip_if_audio_track_is_in_list:
            return SkipReason.AUDIO_LANGUAGE_IN_SKIP_LIST

    return None


def check_subtitles(video_file: str, target_language: Union[LanguageCode, List[LanguageCode]], require_subgen: bool = False) -> Tuple[bool, Optional[List[LanguageCode]]]:
    """
    Combined function to check for both internal and external subtitles in a video file.
    Returns whether subtitles exist, their languages if found, and if they're subgen subtitles (for external subtitles).
    
    Args:
        video_file: The path to the video file
        target_language: The language or list of languages to check for, or NONE to check for any
        require_subgen: Whether to require subgen subtitles (only applies to external subtitles)
        
    Returns:
        tuple[bool, Optional[List[LanguageCode]]]: 
            (subtitles_exist, detected_languages)
    """
    detected_languages = []

    # Normalize target_language to a list
    if not isinstance(target_language, list):
        target_language = [target_language] if target_language != LanguageCode.NONE else []

    if not require_subgen:
        # Don't check internal subtitles if we require subgen subtitles
        # Check internal subtitles first
        internal_exists, internal_languages = check_internal_subtitles(video_file, target_language)
        if internal_exists:
            detected_languages.extend(internal_languages)

    # If no internal subtitles, check external subtitles
    external_exists, external_languages = check_external_subtitles(video_file, target_language, require_subgen)
    if external_exists:
        detected_languages.extend(external_languages)

    # No subtitles found
    return bool(detected_languages), detected_languages if detected_languages else None

def check_external_subtitles(video_file: str, target_language: List[LanguageCode], require_subgen: bool = False, recursion = True) -> Tuple[bool, Optional[List[LanguageCode]]]:
    """
    Core function to check for external subtitle files.
    Returns whether subtitles exist, their languages if found, and if they're subgen subtitles.
    
    Args:
        video_file: The path to the video file
        target_language: The list of languages to check for, or empty list to check for any
        require_subgen: Whether to require subgen subtitles
        
    Returns:
        tuple[bool, Optional[List[LanguageCode]]]: (subtitles_exist, detected_languages)
    """
    video_file_stripped = os.path.splitext(os.path.split(video_file)[1])[0]
    folder_path = os.path.dirname(video_file)
    detected_languages = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            root, ext = os.path.splitext(file_name)
            if root.startswith(video_file_stripped) and ext.lower() in SUBTITLE_EXTENSIONS:
                identifier_string = root[len(video_file_stripped):]
                
                is_subgen = has_word_in_string(identifier_string, "subgen")
                is_default = has_word_in_string(identifier_string, "default")
                
                if require_subgen and not is_subgen:
                    continue
                
                if is_default:
                    detected_languages.append(None)
                    continue
                    
                subtitle_language = find_language_in_string(identifier_string)
                if subtitle_language:
                    if not target_language or subtitle_language in target_language:
                        detected_languages.append(subtitle_language)
                    else:
                        detected_languages.append(None)
                        
        elif os.path.isdir(file_path) and recursion:
            external_exists, external_languages = check_external_subtitles(file_path, target_language, require_subgen, False)
            if external_exists:
                detected_languages.extend(external_languages)
                    
    return bool(detected_languages), detected_languages if detected_languages else None

def check_internal_subtitles(video_file: str, target_language: List[LanguageCode]) -> Tuple[bool, Optional[List[LanguageCode]]]:
    """
    Core function to check for internal subtitles in a video file.
    Returns whether subtitles exist and their languages if found.
    
    Args:
        video_file: The path to the video file
        target_language: The list of languages to check for, or empty list to check for any
        
    Returns:
        tuple[bool, Optional[List[LanguageCode]]]: (subtitles_exist, detected_languages)
    """
    detected_languages = []

    try:
        with av.open(video_file) as container:
            subtitle_streams = [s for s in container.streams if s.type == 'subtitle']
            
            if not subtitle_streams:
                return False, None
                
            for subtitle_stream in subtitle_streams:
                if 'language' in subtitle_stream.metadata:
                    subtitle_language = LanguageCode.from_iso_639_2(subtitle_stream.metadata.get('language'))
                    if subtitle_language:
                        if not target_language or subtitle_language in target_language:
                            detected_languages.append(subtitle_language)
                    else:
                        logging.warning(f"Unsupported subtitle language code: {subtitle_stream.metadata.get('language')} in {video_file}")
                else:
                    # Subtitle exists but has no language tag
                    detected_languages.append(None)
                    
            return bool(detected_languages), detected_languages if detected_languages else None
            
    except Exception as e:
        logging.error(f"Error checking internal subtitles in {os.path.basename(video_file)}: {str(e)}")
        logging.error(traceback.format_exc())
        return False, None

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


def get_audio_languages(video_path):
    """
    Extract language codes from each audio stream in the video file.

    :param video_path: Path to the video file
    :return: List of language codes for each audio stream
    """
    audio_tracks = get_audio_tracks(video_path)
    return [track['language'] for track in audio_tracks]    

            
    

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

def refresh_jellyfin_metadata(jellyfin_item_id: str, jellyfin_base_url: str, jellyfin_api_key: str) -> None:
    """
    Refreshes the metadata of a Jellyfin library item.
    
    Args:
        jellyfin_item_id: The ID of the item in the Jellyfin library whose metadata needs to be refreshed.
        jellyfin_base_url: The IP address of the Jellyfin server.
        jellyfin_api_key: The Jellyfin token used for authentication.
        
    Raises:
        Exception: If the server does not respond with a successful status code.
    """

    # Jellyfin API endpoint to refresh metadata for a specific item
    url = f"{jellyfin_base_url}/Items/{jellyfin_item_id}/Refresh"

    # Headers to include the Jellyfin token for authentication
    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_api_key}",
    }
    
    # Query parameters
    params = {
        "metadataRefreshMode": "FullRefresh",
    }


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
            is_added_to_queue = gen_subtitles_queue(file_path, transcribe_or_translate)
            if is_added_to_queue:
                logging.info(f"[Monitor] File: {file_path} was added to the queue.")
            else:
                logging.debug(f"[Monitor] File: {file_path} was not added to the queue.")
        def on_created(self, event):
            if self.will_handle(event):
                logging.debug(f"[Monitor] Handling [File created]: {event.src_path}")
                self.create_subtitle(event.src_path) #Removed pathmapping
                
        def on_modified(self, event):
            logging.debug(f"[Monitor] Ignoring [File modified]: {event.src_path}")
            # Let's not do this, because it might trigger something that's already being processed
            
            
        def will_handle(self, event):
            if not event.is_directory:
                file_path = event.src_path
                file_name = os.path.basename(file_path)
                # Exclude files in ignore_files based on file name (only the file name, not the full path)
                if any(ignore_folders):
                    event_folder = os.path.dirname(file_path)  # Get the directory of the created/modified file
                    if self._is_subfolder(event_folder) and self._is_ignored_folder(event_folder):
                        logging.debug(f"[Monitor] Excluding folder: {event_folder} due to ignore_folders criteria.")
                        return False
                if any(ignore_files):
                    if any(keyword in file_name.lower() for keyword in ignore_files):
                        logging.debug(f"[Monitor] Excluding file: {file_path} due to ignore_files criteria.")
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
        path = path ##removed path mapping
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
        # logger.debug(f"Excluding file: {file_path} due to ignore_files criteria.")
        return
    
    #Maybe check first if if has an video/audio file exension before checking this list and then check if it has audio with the av probe. Not sure what would be better
    
    if file_path in files_to_skip_list:
        # logger.debug(f"Excluding file: {file_path} due to skip list.")
        return

    if is_valid_audio_file(file_path, check_extensions = False):
        logging.debug(f"Processing {file_path} in thread {threading.current_thread().name}")
        gen_subtitles_queue(file_path, transcribe_or_translate) #Removed pathmapping
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
    
    transcription_threads = []
    if transcribe_folders:
        if not (transcribe_existing_in_transcribe_folders or monitor):
            logging.warning("TRANSCRIBE_FOLDERS is set, but TRANSCRIBE_EXISTING_IN_TRANSCRIBE_FOLDERS or MONITOR is not set to True. TRANSCRIBE_FOLDERS will be ignored.")
        
        if transcribe_existing_in_transcribe_folders:
            logging.info(F"Transcibing folders: {transcribe_folders}")
            add_files_to_queue_threads = transcribe_existing(transcribe_folders)
            
            logging.info("Starting transcription workers")
            transcription_threads = start_transcription_workers()
        
        if monitor:
            logging.info("Starting to monitor folders for new files.")
            monitor_folders(transcribe_folders)
    

    server_thread = None
    if use_webhooks:
            # Run uvicorn in a separate thread
        server_thread = threading.Thread(target=run_uvicorn, daemon=True, name="uvicorn_server")
        server_thread.start()

    subtitle_tags = load_subtitle_tag_config(os.getenv('SUBTITLE_TAGS', ''), whisper_model=whisper_model.split('.')[0], subtitle_language_naming_type=subtitle_language_naming_type, language=namesublang if namesublang else (LanguageCode.ENGLISH if transcribe_or_translate == "translate" else (force_detected_language_to if force_detected_language_to else LanguageCode.ENGLISH)))


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
