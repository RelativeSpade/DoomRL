import os
import re


def get_latest_model(checkpoint_dir):
    pattern = re.compile(r'Doom(\d+)')
    checkpoint_files = [
        (file, int(pattern.search(file).group(1)))
        for file in os.listdir(checkpoint_dir)
        if pattern.search(file)
    ]
    if not checkpoint_files:
        return None, 0

    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])
    print('Latest checkpoint:', latest_checkpoint[0])
    return os.path.join(checkpoint_dir, latest_checkpoint[0]), latest_checkpoint[1]
