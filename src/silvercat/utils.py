import shutil
from pathlib import Path

import kagglehub


def download_lending_club_data() -> None:
    path = Path(kagglehub.dataset_download("wordsforthewise/lending-club"))

    for file in path.iterdir():
        if file.is_file():
            shutil.copy2(file, Path("data") / file.name)

    print("Successfully copied files to data/")
