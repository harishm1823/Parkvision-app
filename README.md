# ParkVision - AI Parking Detection System

🚗 **99.06% Accuracy** | Real-time parking space detection using YOLOv8

## Quick Deploy to Render (Free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 2. Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Use these settings:
   - **Environment**: Docker
   - **Plan**: Free
   - **Auto-Deploy**: Yes

### 3. Environment Variables (Optional)
- `CONFIDENCE_THRESHOLD`: 0.5 (default)
- `USE_GPU`: false (for free tier)

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/detect` - Upload image for detection
- `GET /` - Web interface

## Features

- 🎯 99.06% detection accuracy
- 🚀 Real-time processing
- 🌐 Web dashboard
- 📱 Mobile responsive
- 🐳 Docker ready
- ☁️ Cloud deployable

## Tech Stack

- **AI Model**: YOLOv8 (Ultralytics)
- **Backend**: Flask + Python
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Docker + Render