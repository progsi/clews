# repo of discogs-vi-2 needs to be in the same dir as this repo
dvi2_path="../discogs-vi-2/data/dvi2/dataset/"

mkdir -p data/dvi2 data/dvi2fm data/dvi2fm_light

# Copy train.json, test.json, val.json to data/dvi2
cp "${dvi2_path}train.json" "${dvi2_path}test.json" "${dvi2_path}val.json" data/dvi2/

# Copy matched/train.json, matched/test.json, matched/val.json to data/dvi2fm
cp "${dvi2_path}matched/train.json" "${dvi2_path}matched/test.json" "${dvi2_path}matched/val.json" data/dvi2fm/

# Copy matched/train.json and matched/test.json to data/dvi2fm_light
cp "${dvi2_path}matched/train.json" "${dvi2_path}matched/test.json" data/dvi2fm_light/

# Copy val.json from dvi2_path to data/dvi2fm_light
cp "${dvi2_path}val.json" data/dvi2fm_light/