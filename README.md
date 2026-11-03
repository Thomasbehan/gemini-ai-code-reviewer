## Google Cloud credits are provided for this project `#VertexAISprint`
Thanks, [Google](https://github.com/google) :)

# Gemini AI Code Reviewer

A GitHub Action that automatically reviews pull requests using Google's Gemini AI.

## Features

- Review your PRs using the Gemini API — a full comprehensive re-review on every push, not just the first
- Inline, line-anchored comments with a concrete fix and a severity tag (`[BLOCKER]`, `[MAJOR]`, `[MINOR]`, `[NIT]`)
- Evidence rules that stop it flagging code it cannot see, or APIs it only remembers from older versions
- Reads the repository's own conventions (`REVIEW.md`, `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`,
  `.github/copilot-instructions.md`, `CONTRIBUTING.md`) and treats them as overriding its generic standards
- Adversarial verify pass, with the full file and repository context, that drops false-positive findings before posting
- Replies in-thread when a human responds to one of its review comments

![Demo](assets/img/Demo.png)
![Demo2](assets/img/Demo2.png)
[Video Demo](https://www.youtube.com/watch?v=pc1ffEFMIQo)

## Setup

1. To use this GitHub Action, you need an Gemini API key. If you don't have one, sign up for an API key
   at [Google AI Studio](https://makersuite.google.com/app/apikey).

2. Add the Gemini API key as a GitHub Secret in your repository with the name `GEMINI_API_KEY`. You can find more
   information about GitHub Secrets [here](https://docs.github.com/en/actions/reference/encrypted-secrets).

3. Create a `.github/workflows/code-review.yml` file in your repository and add the following content:

### Option 1: Automatic Review on Pull Request Events (Recommended)

This configuration automatically triggers code review when a PR is opened, updated, or reopened:

```yaml
name: AI Code Reviewer

on:
  pull_request:
    branches: [ main ]

permissions:
  issues: write
  pull-requests: write
  contents: read

jobs:
  gemini-code-review:
    runs-on: ubuntu-latest
    steps:
      - name: PR Info
        run: |
          echo "PR Number: ${{ github.event.pull_request.number }}"
          echo "Repository: ${{ github.repository }}"
          echo "Head SHA: ${{ github.event.pull_request.head.sha }}"
          echo "Base SHA: ${{ github.event.pull_request.base.sha }}"

      - name: Checkout Repo
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Get PR Details
        id: pr
        run: |
          echo "head_sha=${{ github.event.pull_request.head.sha }}" >> $GITHUB_OUTPUT
          echo "base_sha=${{ github.event.pull_request.base.sha }}" >> $GITHUB_OUTPUT

      - uses: Thomasbehan/gemini-ai-code-reviewer@main
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          EXCLUDE: "*.md,*.txt,package-lock.json,*.yml,*.yaml"
          SYSTEM_PROMPT: | # Optional: Custom system prompt for code reviews
            Review the code with the following guidelines:
            - Focus on security vulnerabilities and potential bugs
            - Check for proper error handling
            - Ensure code follows best practices
            - Verify performance considerations
```

### Option 2: Manual Trigger via Comment

This configuration requires manual triggering by commenting `/gemini-review` on a pull request:

```yaml
name: Gemini AI Code Reviewer

on:
  issue_comment:
    types: [created]

permissions: write-all

jobs:
  gemini-code-review:
    runs-on: ubuntu-latest
    if: |
      github.event.issue.pull_request &&
      contains(github.event.comment.body, '/gemini-review')
    steps:
      - name: PR Info
        run: |
          echo "Comment: ${{ github.event.comment.body }}"
          echo "Issue Number: ${{ github.event.issue.number }}"
          echo "Repository: ${{ github.repository }}"

      - name: Checkout Repo
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Get PR Details
        id: pr
        run: |
          PR_JSON=$(gh api repos/${{ github.repository }}/pulls/${{ github.event.issue.number }})
          echo "head_sha=$(echo $PR_JSON | jq -r .head.sha)" >> $GITHUB_OUTPUT
          echo "base_sha=$(echo $PR_JSON | jq -r .base.sha)" >> $GITHUB_OUTPUT
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - uses: Thomasbehan/gemini-ai-code-reviewer@main
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          EXCLUDE: "*.md,*.txt,package-lock.json,*.yml,*.yaml"
          SYSTEM_PROMPT: | # Optional: Custom system prompt for code reviews
            Review the code with the following guidelines:
            - Focus on security vulnerabilities and potential bugs
            - Check for proper error handling
            - Ensure code follows best practices
            - Verify performance considerations
```

### Option 3: Reply to review threads

The reviewer can also respond when a human **replies** to one of its own inline
review comments — answering questions, conceding false positives, or holding the
line with concrete evidence. Add a `pull_request_review_comment` trigger alongside
your review workflow (it only acts on replies to threads it started, and never
replies to itself):

```yaml
name: Gemini AI Code Reviewer — thread replies

on:
  pull_request_review_comment:
    types: [created]

permissions:
  pull-requests: write
  contents: read

jobs:
  gemini-thread-reply:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: Thomasbehan/gemini-ai-code-reviewer@main
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

## Configuration Options

### Required Parameters
- **GITHUB_TOKEN**: GitHub token to interact with the repository (automatically provided by GitHub Actions)
- **GEMINI_API_KEY**: Your Google Gemini API key (store this as a GitHub Secret)

### Optional Parameters

#### Noise Control and Limits
- MAX_COMMENTS_TOTAL: Optional hard cap on the total number of review comments posted per run (default: 0 = disabled).
- MAX_COMMENTS_PER_FILE: Optional cap on the number of comments per file (default: 0 = disabled).
- REVIEW_PRIORITY_THRESHOLD: Only post comments at or above this priority. One of: low, medium, high, critical. Example: high.
- REVIEW_MODE: Controls strictness. Options: strict, standard, lenient, security_focused, performance_focused. For fewer comments use lenient.

Example usage in workflow inputs (maps to env):
```yaml
- uses: Thomasbehan/gemini-ai-code-reviewer@main
  with:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    EXCLUDE: "*.md,*.txt,package-lock.json,*.yml,*.yaml"
    MAX_COMMENTS_TOTAL: 12
    MAX_COMMENTS_PER_FILE: 4
    REVIEW_PRIORITY_THRESHOLD: high
    REVIEW_MODE: lenient
```
- **Model**: reviews always run on `gemini-3.8-flash`, pinned by this action so every repository gets the same
  reviewer. **`GEMINI_MODEL` is deprecated and ignored**: if it is set, the run logs a warning asking you to remove it
  and carries on with the pinned model.
  
- **EXCLUDE**: Comma-separated list of file patterns to exclude from review (e.g., `*.md,*.txt,package-lock.json`)
  
- **SYSTEM_PROMPT**: Custom system prompt to define specific rules and guidelines for code reviews
  - If not provided, the action uses a default prompt based on best practices
  - Use this to customize the review focus (e.g., security-first, performance-focused, style-specific)
  - Example:
    ```yaml
    SYSTEM_PROMPT: |
      You are a senior code reviewer. Focus on:
      1. Security vulnerabilities (SQL injection, XSS, etc.)
      2. Performance bottlenecks
      3. Code maintainability and readability
      4. Proper error handling
      Do not comment on minor style issues.
    ```

### AI Model Settings

The defaults follow Google's guidance for Gemini 3 models:

- **Temperature**: `1.0` (default). Gemini 3 models are tuned for it; lower values can degrade their reasoning or cause looping.
- **Top P**: `0.95` (default).
- **Thinking level**: `high` (default). One of `low`, `medium` or `high`; more thinking means fewer false positives at higher latency.
- **Max Output Tokens**: `32768` (default). Thinking tokens count against this budget, so it leaves room for the answer.

These settings can be overridden via environment variables:
```yaml
env:
  GEMINI_TEMPERATURE: "1.0"       # Range: 0.0 to 2.0
  GEMINI_TOP_P: "0.95"            # Range: 0.0 to 1.0
  GEMINI_THINKING_LEVEL: "high"   # low, medium or high
  GEMINI_MAX_TOKENS: "32768"      # Maximum output tokens, including thinking
```

4. Commit the workflow file to your repository.

5. Usage depends on which option you chose:
   - **Option 1 (Automatic)**: The code review will automatically run when you open, update, or reopen a pull request to the configured branch.
   - **Option 2 (Manual)**: Comment `/gemini-review` on any pull request to trigger the code review.

## How It Works

This GitHub Action uses the Gemini AI API to provide code review feedback. It works by:

1. **Analyzing the changes**: It grabs the code modifications from your pull request and filters out any files you don't want reviewed.
2. **Consulting the Gemini model**: It sends chunks of the modified code to the Gemini for analysis.
3. **Providing feedback**: Gemini AI examines the code and generates review comments.
4. **Delivering the review**: The Action always posts a review on your pull request:
   - If **no issues are found**, the PR is **approved**
   - If **any finding is posted**, `[NIT]` included, the review **requests changes**, with one inline thread per finding
     and a summary tallying them by severity and listing the merge blockers
   - Reviews are anchored to the commit that was reviewed, so a push during a review never misplaces its comments

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Star History ⭐️

[![Star History Chart](https://api.star-history.com/svg?repos=truongnh1992/gemini-ai-code-reviewer&type=Date)](https://star-history.com/#truongnh1992/gemini-ai-code-reviewer&Date)