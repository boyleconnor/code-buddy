"""Copied from: https://github.com/sayakpaul/hf-codegen/blob/main/data/parallel_clone_repos.py"""
import argparse
import os
import subprocess
from multiprocessing import Pool

LOCAL_DIRECTORY = "training_repos"


def mirror_repository(repository):
    """Locally clones a repository."""
    repository_url = f"https://github.com/{repository}.git"
    repository_path = os.path.join(LOCAL_DIRECTORY, repository)

    # FIXME: This clones the repo instead of downloading it
    # Clone the repository
    subprocess.run(["git", "clone", repository_url, repository_path])


def mirror_repositories():
    parser = argparse.ArgumentParser()
    parser.add_argument("repositories", type=str, nargs="+")
    args = parser.parse_args()

    # Get the list of repositories
    repositories: list[str] = args.repositories
    print(f"Total repositories: {len(repositories)}.")

    # Create the mirror directory if it doesn't exist
    if not os.path.exists(LOCAL_DIRECTORY):
        os.makedirs(LOCAL_DIRECTORY)

    # Mirror repositories using multiprocessing
    print("Cloning repositories.")
    with Pool() as pool:
        pool.map(mirror_repository, repositories)


if __name__ == "__main__":
    mirror_repositories()
