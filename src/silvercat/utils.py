from dataclasses import dataclass
from pathlib import Path
import shutil

import kagglehub


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    processed_dir: Path
    models_dir: Path
    accepts: Path
    rejects: Path
    data_dict: Path


def setup(
    project_root: str | Path | None = None,
    download_data: bool = True,
) -> ProjectPaths:
    """Set up project paths and optionally download the Lending Club dataset."""

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    models_dir = data_dir / "models"

    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        accepts=data_dir / "accepted_2007_to_2018Q4.csv.gz",
        rejects=data_dir / "rejected_2007_to_2018Q4.csv.gz",
        data_dict=project_root / "LCDataDictionary.csv",
    )

    if download_data and not paths.accepts.exists():
        kaggle_path = Path(
            kagglehub.dataset_download("wordsforthewise/lending-club")
        )

        for file in kaggle_path.iterdir():
            if file.is_file():
                shutil.copy2(file, data_dir / file.name)

        print(f"Successfully copied Lending Club files to {data_dir}")
    elif paths.accepts.exists():
        print("Lending Club data already exists")

    print("Project root:", paths.project_root)
    print("Data directory:", paths.data_dir)
    print("Accepted data exists:", paths.accepts.exists())
    print("Rejected data exists:", paths.rejects.exists())
    print("Data dictionary exists:", paths.data_dict.exists())

    return paths
