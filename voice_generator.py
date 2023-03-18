import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import librosa
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
import re
import logging
import sys
from pydub import AudioSegment
from pydub.playback import play
import soundfile

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
logging.getLogger('numba').setLevel(logging.WARNING)

def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text
    
class VoiceGenerator():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
        parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
        parser.add_argument("--share", default=False, help="make link public (used in colab)")
        args = parser.parse_args()
        
        device = "cpu"
        self.hps = utils.get_hparams_from_file(args.config_dir)
        
        self.net_g = SynthesizerTrn(
            len(self.hps.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(device)
        
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(args.model_dir, self.net_g, None)

    def tts_play(self, user, msg):

        text = "[ZH]{0}[ZH]".format(msg)

        length_scale, text = get_label_value(
            text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(
            text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(
            text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')

        stn_tst = get_text(text, self.hps, cleaned=cleaned)
        
        speaker_id = 1
        out_path = 'wav_temp'

        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speaker_id])
            audio = self.net_g.infer(x_tst, 
                                     x_tst_lengths, 
                                     sid=sid, 
                                     noise_scale=noise_scale,
                                     noise_scale_w=noise_scale_w, 
                                     length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
            
        soundfile.write('wav_temp/output.wav', audio, self.hps.data.sampling_rate)
        song = AudioSegment.from_wav("wav_temp/output.wav")
        play(song)

    def tts_danmu(self, user, msg):
        self.tts_play(user, msg)

    def tts_gift(self, user, gift_name, gift_count):
        print(gift_name)

    def log(self, msg):
        self.tts_play("系统", msg)

    def voice_conversion(self, msg):

        length_scale, text = get_label_value(
            text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(
            text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(
            text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')

        stn_tst = get_text(text, self.hps, cleaned=cleaned)

        speaker_id = get_speaker_id('Speaker ID: ')
        out_path = input('Path to save: ')

        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speaker_id])
            audio = self.net_g.infer(x_tst, 
                                     x_tst_lengths, 
                                     sid=sid, 
                                     noise_scale=noise_scale,
                                     noise_scale_w=noise_scale_w, 
                                     length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
            return audio, out_path
