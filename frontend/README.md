# Frontend for Vector Memory Chat

A modern web interface for the Vector Memory Chat system built with vanilla JavaScript, HTML, and CSS.

## Features

✅ **Modern UI** - Clean, responsive design with gradient backgrounds
✅ **Real-time Chat** - Send messages and get AI responses
✅ **Session Management** - Create, resume, and list previous conversations
✅ **Knowledge Base Upload** - Upload documents (.txt, .pdf) via drag-and-drop
✅ **Context Controls** - Toggle context and knowledge base retrieval
✅ **Settings** - Configure API URL and user ID
✅ **Responsive** - Works on desktop and mobile devices

## Quick Start

### 1. Start the FastAPI Backend

```bash
# From project root
python src/api/chat_api.py
```

The API should be running at `http://localhost:8000`

### 2. Serve the Frontend

You have several options:

**Option A: Python HTTP Server**
```bash
cd frontend
python -m http.server 8080
```

**Option B: Node.js HTTP Server**
```bash
cd frontend
npx http-server -p 8080
```

**Option C: VS Code Live Server**
- Install "Live Server" extension
- Right-click on `index.html`
- Select "Open with Live Server"

### 3. Open in Browser

Visit: `http://localhost:8080`

## Usage

### Chat Interface

1. **Start Chatting**: Type your message and press Enter or click the send button
2. **Toggle Options**: Use checkboxes to enable/disable context and knowledge base
3. **View Session Info**: Click the sidebar to see session details

### Session Management

1. Click the **📋 Sessions** button to view all previous sessions
2. Click any session to resume it
3. Click **🆕 New Session** to start fresh
4. Click **🗑️ Clear Session** to reset the current conversation

### Upload Documents

1. Click the **📁 Upload** button
2. Select .txt or .pdf files
3. Click **Upload Files**
4. Documents will be indexed into the knowledge base

### Settings

1. Click the **⚙️ Settings** button
2. Configure:
   - **API URL**: Backend endpoint (default: http://localhost:8000)
   - **User ID**: Your unique identifier (default: default_user)
3. Click **Save Settings**

## Architecture

```
frontend/
├── index.html      # Main HTML structure
├── style.css       # Styling and animations
├── app.js          # JavaScript logic and API calls
└── README.md       # This file
```

### Key Components

- **Chat Interface**: Real-time messaging with typing indicators
- **Session Sidebar**: Displays current session info
- **Modal Dialogs**: For sessions, upload, and settings
- **API Client**: Handles all backend communication

## API Integration

The frontend communicates with the FastAPI backend through REST endpoints:

- `POST /chat` - Send messages
- `POST /session/new` - Create session
- `POST /session/resume` - Resume session
- `GET /session/list` - List sessions
- `POST /knowledge/upload` - Upload documents

## Customization

### Change Theme Colors

Edit CSS variables in `style.css`:

```css
:root {
    --primary-color: #4a90e2;    /* Main blue */
    --primary-hover: #357abd;    /* Hover blue */
    /* ... other colors ... */
}
```

### Modify API URL

1. Click Settings ⚙️
2. Update API URL
3. Save (persisted in localStorage)

## Browser Support

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ⚠️ IE11 not supported

## Troubleshooting

### "Failed to fetch" Error

- Ensure the FastAPI backend is running
- Check the API URL in Settings
- Verify CORS is enabled on the backend

### Upload Not Working

- Check file format (.txt or .pdf only)
- Verify backend has write permissions
- Check browser console for errors

### Sessions Not Loading

- Ensure ChromaDB is initialized
- Check that user_id matches between sessions
- Verify backend is accessible

## Future Enhancements

- [ ] Dark mode toggle
- [ ] Markdown rendering in messages
- [ ] Voice input
- [ ] Export conversations
- [ ] Message search
- [ ] WebSocket support for streaming

## License

Part of the LangChain AI Application project.
