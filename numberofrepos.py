import requests
from requests.auth import HTTPBasicAuth

# Replace with your credentials
USERNAME = "your_bitbucket_username"
APP_PASSWORD = "your_app_password"  # App password, not your regular password
WORKSPACE = "your_workspace_id"

# Base URL
BASE_URL = f"https://api.bitbucket.org/2.0/repositories/{WORKSPACE}"

def get_all_repos():
    repos = []
    url = BASE_URL
    while url:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, APP_PASSWORD))
        data = response.json()
        for repo in data.get("values", []):
            repos.append(repo["slug"])
        url = data.get("next")  # Pagination
    return repos

def get_branches_for_repo(repo_slug):
    branches = []
    url = f"{BASE_URL}/{repo_slug}/refs/branches?pagelen=100"
    while url:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, APP_PASSWORD))
        data = response.json()
        for branch in data.get("values", []):
            branches.append(branch["name"])
        url = data.get("next")  # Pagination
    return branches

def main():
    total_branches = 0
    print("Fetching repositories and branches...")
    repos = get_all_repos()
    for repo in repos:
        branches = get_branches_for_repo(repo)
        print(f"Repository: {repo} - Branches: {len(branches)}")
        total_branches += len(branches)

    print("\n===================================")
    print(f"Total Repositories: {len(repos)}")
    print(f"Total Branches across all repositories: {total_branches}")
    print("===================================")

if __name__ == "__main__":
    main()
