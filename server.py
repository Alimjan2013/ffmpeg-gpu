import subprocess


import os
import requests
from flask import Flask, request, jsonify
import ffmpeg
import uuid
import boto3
import io

app = Flask(__name__)
DOWNLOAD_DIR = '/tmp/downloads'
OUTPUT_DIR = '/tmp/outputs'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/convert', methods=['POST'])
def convert():
    # Get R2/S3 credentials and config from environment
    r2_account_id = os.environ.get('R2_ACCOUNT_ID')
    r2_access_key_id = os.environ.get('R2_ACCESS_KEY_ID')
    r2_access_key_secret = os.environ.get('R2_ACCESS_KEY_SECRET')
    r2_bucket = os.environ.get('R2_BUCKET')
    r2_region = os.environ.get('R2_REGION', 'auto')
    # Compose endpoint URL
    r2_endpoint = f'https://{r2_account_id}.r2.cloudflarestorage.com'

    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No url provided'}), 400
    try:
        # Download file
        filename = str(uuid.uuid4())
        input_path = os.path.join(DOWNLOAD_DIR, filename)
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(input_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            return jsonify({'error': f'Failed to download file: {str(e)}'}), 400

        # Check if file was downloaded and is not empty
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            return jsonify({'error': 'Downloaded file is empty or missing'}), 400

        # Get format from request, default to mp4
        out_format = data.get('format', 'mp4').lower()
        if out_format not in ['mp4', 'webm', 'mkv', 'h265']:
            return jsonify({'error': 'Unsupported format'}), 400

        # Set codec and extension based on format (GPU-accelerated for mp4/h265 if possible)
        use_gpu = os.environ.get('USE_GPU', '1') == '1'
        if out_format == 'mp4':
            if use_gpu:
                vcodec = 'h264_nvenc'
            else:
                vcodec = 'libx264'
            acodec = 'aac'
            ext = '.mp4'
            ff_format = 'mp4'
        elif out_format == 'webm':
            vcodec = 'libvpx-vp9'
            acodec = 'libopus'
            ext = '.webm'
            ff_format = 'webm'
        elif out_format == 'mkv':
            vcodec = 'av1_nvenc'
            acodec = 'libopus'
            ext = '.mkv'
            ff_format = 'matroska'
        elif out_format == 'h265':
            if use_gpu:
                vcodec = 'hevc_nvenc'
            else:
                vcodec = 'libx265'
            acodec = 'aac'
            ext = '.mp4'
            ff_format = 'mp4'

        output_path = os.path.join(OUTPUT_DIR, filename + ext)
        try:
            stream = ffmpeg.input(input_path)
            output_kwargs = dict(
                vcodec=vcodec,
                vf='scale=-2:480',
                acodec=acodec,
                audio_bitrate='128k',
                format=ff_format
            )
            # Add GPU-specific options if using GPU
            if use_gpu and vcodec in ['h264_nvenc', 'hevc_nvenc']:
                output_kwargs['preset'] = 'fast'
                output_kwargs['rc'] = 'vbr'
                output_kwargs['gpu'] = 0
            stream = ffmpeg.output(
                stream,
                output_path,
                **output_kwargs
            )
            ffmpeg.run(stream, overwrite_output=True)
        except Exception as e:
            return jsonify({'error': f'ffmpeg conversion failed: {str(e)}'}), 500

        # Upload to R2 via S3 protocol
        try:
            s3 = boto3.client(
                service_name="s3",
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key_id,
                aws_secret_access_key=r2_access_key_secret,
                region_name=r2_region,
            )
            file_key = os.path.basename(output_path)
            with open(output_path, 'rb') as f:
                s3.upload_fileobj(f, r2_bucket, file_key)
            r2_url = f"{r2_endpoint}/{r2_bucket}/{file_key}"
        except Exception as e:
            return jsonify({'error': f'Upload to R2 failed: {str(e)}'}), 500

        return jsonify({'downloaded': input_path, 'output': output_path, 'r2_url': r2_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200

@app.route('/gpu-test', methods=['GET'])
def gpu_test():
    """Test if NVIDIA GPU is available and ffmpeg NVENC encoders are present."""
    # Check nvidia-smi
    try:
        smi_out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, timeout=5).decode()
        gpu_status = 'OK'
    except Exception as e:
        smi_out = str(e)
        gpu_status = 'FAIL'

    # Check ffmpeg NVENC encoders
    try:
        ffmpeg_out = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.STDOUT, timeout=10).decode()
        nvenc_present = any(enc in ffmpeg_out for enc in ['h264_nvenc', 'hevc_nvenc'])
    except Exception as e:
        ffmpeg_out = str(e)
        nvenc_present = False

    return jsonify({
        'gpu_status': gpu_status,
        'nvidia_smi': smi_out,
        'nvenc_in_ffmpeg': nvenc_present,
        'ffmpeg_encoders_sample': '\n'.join([line for line in ffmpeg_out.splitlines() if 'nvenc' in line])
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
import subprocess
import os
import requests
from flask import Flask, request, jsonify
import uuid
import boto3
import shlex
import threading
import logging
import time
from functools import lru_cache

app = Flask(__name__)

# Basic logging configuration (stdout). User can set LOG_LEVEL=DEBUG
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s %(levelname)s %(threadName)s %(message)s',
)
logger = logging.getLogger(__name__)
DOWNLOAD_DIR = '/tmp/downloads'
OUTPUT_DIR = '/tmp/outputs'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def nvenc_available():
    try:
        out = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.STDOUT, timeout=10).decode()
        return any(x in out for x in ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc'])
    except Exception:
        return False


def build_ffmpeg_command(input_path, output_path, params):
    use_gpu = params['use_gpu'] and nvenc_available()
    out_format = params['out_format']
    scale_filter = None
    width = params['width']
    height = params['height']
    if width or height:
        if use_gpu and params['scale_mode'] == 'gpu':
            sw = width if width else -1
            sh = height if height else -1
            scale_filter = f'scale_npp={sw}:{sh}:interp_algo=super'
        else:
            sw = width if width else -2
            sh = height if height else -2
            scale_filter = f'scale={sw}:{sh}'

    if out_format == 'mp4':
        vcodec = 'h264_nvenc' if use_gpu else 'libx264'
        acodec = 'aac'
        container = 'mp4'
        ext = '.mp4'
    elif out_format == 'h265':
        vcodec = 'hevc_nvenc' if use_gpu else 'libx265'
        acodec = 'aac'
        container = 'mp4'
        ext = '.mp4'
    elif out_format == 'webm':
        vcodec = 'libvpx-vp9'
        acodec = 'libopus'
        container = 'webm'
        ext = '.webm'
    elif out_format == 'mkv':
        # Prefer av1_nvenc if GPU, else fall back to libx265
        vcodec = 'av1_nvenc' if use_gpu and 'av1_nvenc' in available_nvenc_codecs() else ('hevc_nvenc' if use_gpu else 'libx265')
        acodec = 'libopus'
        container = 'matroska'
        ext = '.mkv'
    else:
        raise ValueError('Unsupported format')

    cmd = ['ffmpeg', '-hide_banner', '-y']
    if use_gpu:
        cmd += ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    if params['start_time'] is not None:
        cmd += ['-ss', str(params['start_time'])]
    cmd += ['-i', input_path]
    if params['duration'] is not None:
        cmd += ['-t', str(params['duration'])]

    # Video options
    cmd += ['-c:v', vcodec]
    if vcodec.endswith('nvenc'):
        preset = params['preset']
        if preset.startswith('p') or preset in ['slow','medium','fast','hq','ll','llhq','llhp']:
            cmd += ['-preset', preset]
        rc_mode = params['rc_mode']
        if rc_mode:
            if vcodec == 'av1_nvenc':
                # av1_nvenc does not support vbr_hq / cbr_hq; map or drop
                if rc_mode in ['vbr_hq', 'cbr_hq']:
                    rc_mode = rc_mode.split('_')[0]  # vbr_hq -> vbr
                allowed_av1_rc = {'vbr','cbr','constqp'}
                if rc_mode not in allowed_av1_rc:
                    rc_mode = 'vbr'
            cmd += ['-rc', rc_mode]
        if params['bitrate']:
            cmd += ['-b:v', params['bitrate']]
        if params['maxrate']:
            cmd += ['-maxrate', params['maxrate']]
        if params['bufsize']:
            cmd += ['-bufsize', params['bufsize']]
        if params['cq'] is not None and rc_mode in ['vbr','vbr_hq','cbr','cbr_hq']:
            cmd += ['-cq', str(params['cq'])]
        # Lookahead replacement: use -rc-lookahead frames (skip for av1 if build lacks support)
        if vcodec != 'av1_nvenc' and (params.get('look_ahead') or params.get('rc_lookahead') is not None):
            look_val = params.get('rc_lookahead') if params.get('rc_lookahead') is not None else 20
            cmd += ['-rc-lookahead', str(look_val)]
        if params['bframes'] is not None:
            cmd += ['-bf', str(params['bframes'])]
        # Adaptive quantization flags not always supported by av1_nvenc; only apply for h264/hevc
        if vcodec in ['h264_nvenc','hevc_nvenc']:
            if params['temporal_aq']:
                cmd += ['-temporal-aq', '1']
            if params['spatial_aq']:
                cmd += ['-spatial-aq', '1']
            if params['aq_strength'] is not None:
                cmd += ['-aq-strength', str(params['aq_strength'])]
    else:
        if params['crf'] is not None and vcodec in ['libx264','libx265']:
            cmd += ['-crf', str(params['crf'])]
        if params['bitrate']:
            cmd += ['-b:v', params['bitrate']]

    if scale_filter:
        cmd += ['-vf', scale_filter]

    # Audio options
    cmd += ['-c:a', acodec]
    if acodec in ['aac'] and params['audio_bitrate']:
        cmd += ['-b:a', params['audio_bitrate']]
    if acodec == 'libopus' and params['audio_bitrate']:
        cmd += ['-b:a', params['audio_bitrate']]

    # Container format
    cmd += ['-f', container, output_path]
    return cmd, ext, vcodec


def run_ffmpeg(cmd, timeout, stream_output=True):
    """Run ffmpeg command. If stream_output True, log lines in real-time.
    Returns (returncode, combined_output)."""
    start = time.time()
    if not stream_output:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return proc.returncode, proc.stdout.decode(errors='ignore')

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            lines.append(line)
            # Only log relevant progress lines (frame=, size=, speed=, or errors)
            if any(tag in line for tag in ['frame=', 'size=', 'time=', 'speed=', 'error', 'Input #', 'Output #']):
                logger.debug(f"ffmpeg: {line}")
            else:
                logger.trace(line) if hasattr(logger, 'trace') else None
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
        proc.wait()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
    return proc.returncode, '\n'.join(lines)


@lru_cache(maxsize=1)
def available_nvenc_codecs():
    try:
        out = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.STDOUT, timeout=10).decode()
        return {c for c in ['h264_nvenc','hevc_nvenc','av1_nvenc'] if c in out}
    except Exception:
        return set()


@lru_cache(maxsize=1)
def get_s3_client():
    r2_account_id = os.environ.get('R2_ACCOUNT_ID')
    r2_access_key_id = os.environ.get('R2_ACCESS_KEY_ID')
    r2_access_key_secret = os.environ.get('R2_ACCESS_KEY_SECRET')
    r2_region = os.environ.get('R2_REGION', 'auto')
    r2_endpoint = f'https://{r2_account_id}.r2.cloudflarestorage.com'
    return boto3.client(
        service_name='s3',
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_access_key_id,
        aws_secret_access_key=r2_access_key_secret,
        region_name=r2_region,
    )


@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json() or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No url provided'}), 400

    params = {
        'out_format': data.get('format', 'mp4').lower(),
        'use_gpu': (os.environ.get('USE_GPU', '1') == '1') and data.get('use_gpu', True),
        'preset': str(data.get('preset', 'p4')),  # p4 ~ good balance
        'rc_mode': data.get('rc', 'vbr_hq'),
        'bitrate': data.get('bitrate'),  # e.g. '3M'
        'maxrate': data.get('maxrate'),
        'bufsize': data.get('bufsize'),
        'cq': data.get('cq'),  # For NVENC VBR quality (0-51 lower=better)
        'crf': data.get('crf'),  # For CPU codecs
        'width': data.get('width'),
        'height': data.get('height'),
        'scale_mode': data.get('scale_mode', 'gpu'),
        'audio_bitrate': data.get('audio_bitrate', '128k'),
        'start_time': data.get('start'),
        'duration': data.get('duration'),
        'look_ahead': bool(data.get('look_ahead', True)),  # legacy boolean
        'rc_lookahead': data.get('rc_lookahead'),  # explicit frames
        'bframes': data.get('bframes', 3),
        'temporal_aq': bool(data.get('temporal_aq', True)),
        'spatial_aq': bool(data.get('spatial_aq', True)),
        'aq_strength': data.get('aq_strength', 8),
    }

    if params['out_format'] not in ['mp4','webm','mkv','h265']:
        return jsonify({'error': 'Unsupported format'}), 400

    filename = str(uuid.uuid4())
    input_path = os.path.join(DOWNLOAD_DIR, filename)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*512):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        return jsonify({'error': f'Download failed: {e}'}), 400

    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        return jsonify({'error': 'Downloaded file empty'}), 400

    try:
        output_path_base = os.path.join(OUTPUT_DIR, filename)
        cmd, ext, vcodec = build_ffmpeg_command(input_path, output_path_base, params)
        output_path = output_path_base + ext
    except Exception as e:
        return jsonify({'error': f'Parameter processing failed: {e}'}), 400

    timeout_s = data.get('timeout', 3600)
    stream_output = bool(int(str(os.environ.get('STREAM_FFMPEG_LOG', '1'))))
    logger.info('Starting ffmpeg job gpu=%s cmd=%s', params['use_gpu'], ' '.join(shlex.quote(c) for c in cmd))
    try:
        rc, output_text = run_ffmpeg(cmd, timeout=timeout_s, stream_output=stream_output)
        if rc != 0:
            logger.error('ffmpeg failed rc=%s', rc)
            return jsonify({'error': 'ffmpeg failed', 'details': output_text[-4000:], 'command': ' '.join(shlex.quote(c) for c in cmd)}), 500
    except subprocess.TimeoutExpired:
        logger.error('ffmpeg timeout after %ss', timeout_s)
        return jsonify({'error': 'ffmpeg timeout', 'command': ' '.join(shlex.quote(c) for c in cmd)}), 504
    except Exception as e:
        logger.exception('ffmpeg execution error')
        return jsonify({'error': f'ffmpeg execution error: {e}'}), 500

    # Upload
    r2_bucket = os.environ.get('R2_BUCKET')
    r2_account_id = os.environ.get('R2_ACCOUNT_ID')
    r2_endpoint = f'https://{r2_account_id}.r2.cloudflarestorage.com'
    try:
        s3 = get_s3_client()
        file_key = os.path.basename(output_path)
        with open(output_path, 'rb') as f:
            s3.upload_fileobj(f, r2_bucket, file_key)
        r2_url = f"{r2_endpoint}/{r2_bucket}/{file_key}"
    except Exception as e:
        return jsonify({'error': f'Upload failed: {e}'}), 500

    logger.info('Completed job output=%s size=%dB', output_path, os.path.getsize(output_path))
    return jsonify({
        'input': input_path,
        'output': output_path,
        'size_bytes': os.path.getsize(output_path),
        'nvenc': vcodec.endswith('nvenc'),
        'used_gpu': params['use_gpu'] and nvenc_available(),
        'ffmpeg_cmd': ' '.join(shlex.quote(c) for c in cmd),
        'r2_url': r2_url
    }), 200


# ---- Simplified user-facing endpoint ----

SIMPLE_QUALITY_MAP = {
    'fast':  {'preset': 'p3', 'rc': 'vbr',    'cq': 32},
    'balanced': {'preset': 'p4', 'rc': 'vbr_hq', 'cq': 28},
    'hq':   {'preset': 'p5', 'rc': 'vbr_hq', 'cq': 24},
    'max':  {'preset': 'p6', 'rc': 'vbr_hq', 'cq': 20},
}

SIMPLE_RES_WHITELIST = [240, 360, 480, 540, 720, 1080, 1440, 2160]


def parse_simple_payload(data):
    applied = {}
    url = data.get('url')
    if not url:
        raise ValueError('url is required')
    fmt_raw = str(data.get('format', 'mp4')).lower()
    fmt_alias = {'hevc': 'h265', '265': 'h265', 'x265': 'h265', 'mov': 'mp4'}
    out_format = fmt_alias.get(fmt_raw, fmt_raw)
    if out_format not in ['mp4', 'mkv', 'webm', 'h265']:
        raise ValueError('unsupported format')
    # Resolution (allow mis-typed keys)
    resolution = data.get('resolution') or data.get('res') or data.get('height') or data.get('resulvation')
    width = None
    height = None
    if resolution in [None, 'orig', 'original', 'source']:
        applied['resolution'] = 'original'
    else:
        try:
            r_int = int(resolution)
            if r_int not in SIMPLE_RES_WHITELIST:
                # Choose closest
                r_int = min(SIMPLE_RES_WHITELIST, key=lambda v: abs(v - r_int))
            height = r_int
            applied['resolution'] = r_int
        except Exception:
            raise ValueError('invalid resolution value')
    # Quality profile
    quality_key = str(data.get('quality', 'balanced')).lower()
    if quality_key not in SIMPLE_QUALITY_MAP:
        quality_key = 'balanced'
    qconf = SIMPLE_QUALITY_MAP[quality_key]
    applied['quality'] = quality_key
    # GPU toggle
    use_gpu = bool(data.get('gpu', True))
    applied['use_gpu'] = use_gpu
    # Build full params bridging to advanced convert()
    advanced = {
        'format': out_format,
        'use_gpu': use_gpu,
        'preset': qconf['preset'],
        'rc': qconf['rc'],
        'cq': qconf['cq'],
        'width': width,
        'height': height,
        'scale_mode': 'gpu',
        'audio_bitrate': data.get('audio_bitrate', '128k'),
    }
    # Optional clipping
    if 'start' in data:
        advanced['start'] = data['start']
    if 'duration' in data:
        advanced['duration'] = data['duration']
    return url, advanced, applied


def invoke_internal_convert(url, adv):
    # Reuse logic without duplicating code: mimic incoming JSON for /convert
    dummy_payload = {'url': url}
    dummy_payload.update(adv)
    with app.test_request_context(json=dummy_payload):
        return convert()


@app.route('/convert/simple', methods=['POST'])
def convert_simple():
    data = request.get_json() or {}
    try:
        url, adv, applied = parse_simple_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e), 'hint': 'Provide at least url, optionally format, resolution, quality'}), 400
    resp = invoke_internal_convert(url, adv)
    # resp is a Flask response, augment JSON body
    try:
        body, status = resp.get_json(), resp.status_code
        if status == 200 and isinstance(body, dict):
            body['simplified'] = True
            body['defaults_applied'] = applied
            return jsonify(body), status
        return resp
    except Exception:
        return resp

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200

@app.route('/gpu-test', methods=['GET'])
def gpu_test():
    """Test if NVIDIA GPU is available and ffmpeg NVENC encoders are present."""
    # Check nvidia-smi
    try:
        smi_out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, timeout=5).decode()
        gpu_status = 'OK'
    except Exception as e:
        smi_out = str(e)
        gpu_status = 'FAIL'

    # Check ffmpeg NVENC encoders
    try:
        ffmpeg_out = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.STDOUT, timeout=10).decode()
        nvenc_present = any(enc in ffmpeg_out for enc in ['h264_nvenc', 'hevc_nvenc'])
    except Exception as e:
        ffmpeg_out = str(e)
        nvenc_present = False

    return jsonify({
        'gpu_status': gpu_status,
        'nvidia_smi': smi_out,
        'nvenc_in_ffmpeg': nvenc_present,
        'ffmpeg_encoders_sample': '\n'.join([line for line in ffmpeg_out.splitlines() if 'nvenc' in line])
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
