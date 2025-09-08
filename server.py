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
