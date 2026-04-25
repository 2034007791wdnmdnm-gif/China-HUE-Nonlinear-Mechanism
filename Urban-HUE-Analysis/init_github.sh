#!/bin/bash

# GitHub Repository Initialization Script
# Urban HUE Analysis with SHAP

echo "======================================"
echo "GitHub Repository Setup"
echo "======================================"

# 1. Initialize Git repository
echo "Step 1: Initializing Git repository..."
git init

# 2. Add all files
echo "Step 2: Adding files to staging area..."
git add .

# 3. Create initial commit
echo "Step 3: Creating initial commit..."
git commit -m "Initial commit: Urban HUE Analysis with SHAP

- Gradient Boosting Regression model (R²=0.897, Adj R²=0.874)
- SHAP explainability analysis
- 10 types of visualizations
- Bootstrap confidence intervals
- LOOCV for small sample validation"

# 4. Prompt for GitHub remote
echo ""
echo "Step 4: Connect to GitHub"
echo "Please create a new repository on GitHub first at:"
echo "https://github.com/new"
echo ""
read -p "Enter your GitHub username: " username
read -p "Enter repository name (e.g., urban-hue-analysis): " repo

# 5. Add remote
echo "Step 5: Adding remote repository..."
git remote add origin https://github.com/$username/$repo.git

# 6. Push to GitHub
echo "Step 6: Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "======================================"
echo "✓ Successfully pushed to GitHub!"
echo "======================================"
echo "Repository URL: https://github.com/$username/$repo"
