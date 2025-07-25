import requests
from requests.auth import HTTPBasicAuth

# 🔐 Replace with your credentials and workspace ID
USERNAME = "your_bitbucket_username"
APP_PASSWORD = "your_app_password"  # App password, not your normal login password
WORKSPACE = "your_workspace_id"

# Base URL for Bitbucket API
BASE_URL = f"https://api.bitbucket.org/2.0/repositories/{WORKSPACE}"

def get_all_repos():
    repos = []
    url = BASE_URL
    while url:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, APP_PASSWORD))
        data = response.json()
        repos.extend([repo["slug"] for repo in data.get("values", [])])
        url = data.get("next")
    return repos

def count_branches(repo_slug):
    count = 0
    url = f"{BASE_URL}/{repo_slug}/refs/branches?pagelen=100"
    while url:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, APP_PASSWORD))
        data = response.json()
        count += len(data.get("values", []))
        url = data.get("next")
    return count

def main():
    total_branches = 0
    repos = get_all_repos()

    print(f"\nFound {len(repos)} repositories. Counting branches...\n")

    for repo in repos:
        branch_count = count_branches(repo)
        print(f"Repository: {repo} → Branches: {branch_count}")
        total_branches += branch_count

    print("\n============================")
    print(f"Total Branches: {total_branches}")
    print("============================")

if __name__ == "__main__":
    main()
