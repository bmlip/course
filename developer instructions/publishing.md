# Publishing Guide

This guide explains how to submit and publish your changes to BMLIP Colorized.

## Submitting Content

To submit content, **make a Pull Request (PR)**. This allows our automatic testing system to check for possible errors.

When you merge the PR, this creates a **commit on main**. This repository has a GitHub Action that automatically:
1. Generates a static website using the [Pluto static export system](https://plutojl.org/en/docs/notebooks-online/)
2. Updates the PlutoSliderServer

**When the action is complete, the website will show your new notebook.** The PlutoSliderServer is also updated automatically on every commit.




## Workflow: making a change
Here's a step-by-step workflow for making changes to existing lectures or adding new ones:

### Using GitHub Desktop

1. **Ensure your repository is up to date**
   - Open GitHub Desktop
   - Select the BMLIP Colorized repository
   - Click "Fetch origin" to get the latest changes
   - If there are changes, click "Pull origin" to update your local copy

2. **Create a new branch for your changes**
   - Click on the "Current Branch" dropdown
   - Click "New Branch"
   - Name your branch descriptively (e.g., `add-lecture-5` or `fix-typos-lecture-3`)
   - Click "Create Branch"

3. **Make your changes with Pluto**
   - Start Pluto.
   - For existing lectures: Navigate to the lecture file and click Open
   - For new lectures: Create a new notebook, and use the file picker at the top of the notebook to choose a path

4. **Commit your changes.** It is good to commit often, even without a nice commit message. Keep a Draft PR open while you work, so others can see your progress.
   - While working on your notebook, Pluto will auto-save your changes
   - In GitHub Desktop, you'll see your changes listed
   - Add a summary (short description) and description (more details if needed)
   - Click "Commit to [your-branch-name]"

5. **Push your changes to GitHub**
   - Click "Push origin" to upload your branch to GitHub

6. **Create a Pull Request**
   - Click "Create Pull Request" in GitHub Desktop (or go to the repository on GitHub.com, or `Cmd+R` in github desktop)
   - You will be navigated to github.com where you can create the PR
   - Click "Create pull request" or "Create draft PR"

7. **Wait for review and tests**
   - Automatic tests will run on your PR using GitHub Actions. These tests check for errors, broken links, and other issues.
   - Team members may review and suggest changes
   - Address any feedback by making additional commits to your branch

8. **Merge!**


