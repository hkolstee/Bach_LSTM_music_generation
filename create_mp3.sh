#!/bin/bash
python3 bach_inference.py #make sure you change the params in this file (model + model params)
python3 create_midi.py
# fluidsynth soundfonts/YamahaC5Grand-v2_4.sf2 -F output/output.mp3 output/output.midi
fluidsynth soundfonts/040_Florestan_String_Qraurtet.sf2 -F output/output.mp3 output/output.midi
