// MIT License
// Copyright (c) 2025 github.com/paul356
// See LICENSE file for full license text

console.log('=== app.js loading ===');

// API Configuration
// When served from FastAPI, use relative URLs; otherwise use localhost
let API_URL = localStorage.getItem('apiUrl') || window.location.origin;
let USER_ID = localStorage.getItem('userId') || 'default_user';
let SESSION_ID = null;
let USE_CONTEXT = true;
let USE_KNOWLEDGE = true;

console.log('API_URL:', API_URL);

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const useContextCheckbox = document.getElementById('useContext');
const useKnowledgeCheckbox = document.getElementById('useKnowledge');
const userIdDisplay = document.getElementById('userIdDisplay');
const sessionIdDisplay = document.getElementById('sessionIdDisplay');
const messageCount = document.getElementById('messageCount');
const contextStatus = document.getElementById('contextStatus');
const knowledgeStatus = document.getElementById('knowledgeStatus');

// Button Elements
const settingsBtn = document.getElementById('settingsBtn');
const sessionsBtn = document.getElementById('sessionsBtn');
const uploadBtn = document.getElementById('uploadBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const clearSessionBtn = document.getElementById('clearSessionBtn');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const uploadFileBtn = document.getElementById('uploadFileBtn');

console.log('closeSidebarBtn element:', closeSidebarBtn);
console.log('closeSidebarBtn exists:', closeSidebarBtn !== null);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    createNewSession();

    // Check if sidebar is hidden and create restore button if needed
    const sidebar = document.getElementById('sidebar');
    if (sidebar && sidebar.classList.contains('hidden')) {
        createRestoreSidebarButton();
    }
});

function initializeApp() {
    userIdDisplay.textContent = USER_ID;
    document.getElementById('userId').value = USER_ID;
    document.getElementById('apiUrl').value = API_URL;

    USE_CONTEXT = useContextCheckbox.checked;
    USE_KNOWLEDGE = useKnowledgeCheckbox.checked;
}

function setupEventListeners() {
    // Send message
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });

    // Checkboxes
    useContextCheckbox.addEventListener('change', (e) => {
        USE_CONTEXT = e.target.checked;
        updateContextStatus();
    });

    useKnowledgeCheckbox.addEventListener('change', (e) => {
        USE_KNOWLEDGE = e.target.checked;
        updateKnowledgeStatus();
    });

    // Buttons
    settingsBtn.addEventListener('click', () => openModal('settingsModal'));
    sessionsBtn.addEventListener('click', () => {
        openModal('sessionsModal');
        loadSessions();
    });
    uploadBtn.addEventListener('click', () => openModal('uploadModal'));
    newSessionBtn.addEventListener('click', createNewSession);
    clearSessionBtn.addEventListener('click', clearSession);

    // Close sidebar button with null check
    if (closeSidebarBtn) {
        closeSidebarBtn.addEventListener('click', toggleSidebar);
        console.log('Close sidebar button listener added');
    } else {
        console.error('closeSidebarBtn not found!');
    }

    saveSettingsBtn.addEventListener('click', saveSettings);
    uploadFileBtn.addEventListener('click', uploadFiles);
}

// API Functions
async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showNotification('Error: ' + error.message, 'error');
        throw error;
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Disable input
    messageInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message to UI
    addMessageToUI(message, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show typing indicator
    const typingId = showTypingIndicator();

    try {
        const response = await apiRequest('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message: message,
                user_id: USER_ID,
                session_id: SESSION_ID,
                use_context: USE_CONTEXT,
                use_knowledge: USE_KNOWLEDGE
            })
        });

        // Update session ID
        SESSION_ID = response.session_id;
        updateSessionInfo();

        // Remove typing indicator and add response
        removeTypingIndicator(typingId);
        addMessageToUI(response.response, 'assistant');

    } catch (error) {
        removeTypingIndicator(typingId);
        addMessageToUI('Sorry, there was an error processing your message.', 'assistant');
    } finally {
        messageInput.disabled = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

async function createNewSession() {
    try {
        const response = await apiRequest('/session/new', {
            method: 'POST',
            body: JSON.stringify({ user_id: USER_ID })
        });

        SESSION_ID = response.session_id;
        clearMessagesUI();
        updateSessionInfo();
        showNotification('New session created!', 'success');
    } catch (error) {
        showNotification('Failed to create session', 'error');
    }
}

async function loadSessions() {
    const sessionsList = document.getElementById('sessionsList');
    sessionsList.innerHTML = '<p class="loading">Loading sessions...</p>';

    try {
        const response = await apiRequest(`/session/list?user_id=${USER_ID}`);

        if (response.sessions.length === 0) {
            sessionsList.innerHTML = '<p>No previous sessions found.</p>';
            return;
        }

        sessionsList.innerHTML = '';
        response.sessions.forEach(session => {
            const sessionItem = createSessionItem(session);
            sessionsList.appendChild(sessionItem);
        });
    } catch (error) {
        sessionsList.innerHTML = '<p class="error">Failed to load sessions.</p>';
    }
}

async function resumeSession(sessionId) {
    try {
        await apiRequest('/session/resume', {
            method: 'POST',
            body: JSON.stringify({
                user_id: USER_ID,
                session_id: sessionId
            })
        });

        SESSION_ID = sessionId;
        clearMessagesUI();

        // Load chat history for the session
        await loadChatHistory(sessionId);

        updateSessionInfo();
        closeModal('sessionsModal');
        showNotification('Session resumed!', 'success');
    } catch (error) {
        showNotification('Failed to resume session', 'error');
    }
}

async function loadChatHistory(sessionId) {
    try {
        const response = await apiRequest(`/session/history?user_id=${USER_ID}&session_id=${sessionId}`);

        if (response.history && response.history.length > 0) {
            // Clear welcome message
            chatMessages.innerHTML = '';

            // Add a timeline header
            const timelineHeader = document.createElement('div');
            timelineHeader.className = 'timeline-header';
            timelineHeader.innerHTML = '<h3>📜 Chat History</h3><div class="timeline-line"></div>';
            chatMessages.appendChild(timelineHeader);

            // Add each message to the timeline
            response.history.forEach((msg, index) => {
                const timestamp = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : 'Unknown time';
                addHistoryMessageToUI(msg.role, msg.content, timestamp, index);
            });

            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    } catch (error) {
        console.error('Failed to load chat history:', error);
        // Continue without history if it fails
    }
}

async function clearSession() {
    if (!confirm('Clear current session? (History will be preserved in database)')) return;

    try {
        await apiRequest('/session/clear', {
            method: 'POST',
            body: JSON.stringify({ user_id: USER_ID })
        });

        clearMessagesUI();
        showNotification('Session cleared!', 'success');
    } catch (error) {
        showNotification('Failed to clear session', 'error');
    }
}

async function uploadFiles() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;

    if (files.length === 0) {
        showNotification('Please select files to upload', 'error');
        return;
    }

    const uploadStatus = document.getElementById('uploadStatus');
    uploadStatus.className = 'upload-status';
    uploadStatus.textContent = 'Uploading...';

    for (const file of files) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('user_id', USER_ID);

            const response = await fetch(`${API_URL}/knowledge/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                uploadStatus.className = 'upload-status success';
                uploadStatus.textContent = `✅ ${result.message}`;
            } else {
                uploadStatus.className = 'upload-status error';
                uploadStatus.textContent = `❌ ${result.message}`;
            }
        } catch (error) {
            uploadStatus.className = 'upload-status error';
            uploadStatus.textContent = `❌ Failed to upload ${file.name}`;
        }
    }

    fileInput.value = '';
    setTimeout(() => {
        uploadStatus.textContent = '';
        uploadStatus.className = 'upload-status';
    }, 3000);
}

async function updateSessionInfo() {
    try {
        const response = await apiRequest(`/session/info?user_id=${USER_ID}`);

        sessionIdDisplay.textContent = SESSION_ID ? SESSION_ID.substring(0, 8) + '...' : 'None';
        sessionIdDisplay.title = SESSION_ID || 'No active session';
        messageCount.textContent = response.message_count || 0;
    } catch (error) {
        console.error('Failed to update session info:', error);
    }
}

// UI Functions
function addMessageToUI(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Process content for <think> tags
    const processedContent = processThinkTags(content);
    bubble.innerHTML = processedContent;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString();

    bubble.appendChild(time);
    messageDiv.appendChild(bubble);

    // Remove welcome message if exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Update message count
    const count = chatMessages.querySelectorAll('.message').length;
    messageCount.textContent = count;
}

function addHistoryMessageToUI(role, content, timestamp, index) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} history-message`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Process content for <think> tags
    const processedContent = processThinkTags(content);
    bubble.innerHTML = processedContent;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = timestamp;

    bubble.appendChild(time);
    messageDiv.appendChild(bubble);

    chatMessages.appendChild(messageDiv);
}

function processThinkTags(content) {
    // Check if content contains <think> tags
    const thinkRegex = /<think>([\s\S]*?)<\/think>/g;

    if (!thinkRegex.test(content)) {
        // No think tags, return escaped content
        return escapeHtml(content);
    }

    // Process content with think tags
    let processed = escapeHtml(content);
    processed = processed.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g, (match, thinkContent) => {
        const thinkId = 'think-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        return `
            <div class="think-container">
                <div class="think-header" onclick="toggleThink('${thinkId}')">
                    <span class="think-toggle" id="${thinkId}-toggle">▸</span>
                    <span class="think-label">Expand to view think process</span>
                </div>
                <div class="think-content" id="${thinkId}" style="display: none;">
                    ${thinkContent.trim()}
                </div>
            </div>
        `;
    });

    return processed;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function toggleThink(thinkId) {
    const content = document.getElementById(thinkId);
    const toggle = document.getElementById(thinkId + '-toggle');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = '▾';
    } else {
        content.style.display = 'none';
        toggle.textContent = '▸';
    }
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator-' + Date.now();

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span></span><span></span><span></span>';

    bubble.appendChild(typing);
    typingDiv.appendChild(bubble);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return typingDiv.id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) indicator.remove();
}

function clearMessagesUI() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h2>👋 Welcome Back!</h2>
            <p>Continue your conversation or start a new one.</p>
        </div>
    `;
    messageCount.textContent = '0';
}

function createSessionItem(session) {
    const div = document.createElement('div');
    div.className = 'session-item';
    div.onclick = () => resumeSession(session.session_id);

    div.innerHTML = `
        <div class="session-item-header">
            <span class="session-id">${session.session_id.substring(0, 16)}...</span>
            <span class="session-stats">${session.message_count} messages</span>
        </div>
        <div class="session-stats">Last active: ${new Date(session.last_timestamp).toLocaleString()}</div>
        <div class="session-preview">${session.preview}</div>
    `;

    return div;
}

function updateContextStatus() {
    contextStatus.textContent = USE_CONTEXT ? '✅ ON' : '❌ OFF';
}

function updateKnowledgeStatus() {
    knowledgeStatus.textContent = USE_KNOWLEDGE ? '✅ ON' : '❌ OFF';
}

function saveSettings() {
    const newApiUrl = document.getElementById('apiUrl').value.trim();
    const newUserId = document.getElementById('userId').value.trim();

    if (newApiUrl) {
        API_URL = newApiUrl;
        localStorage.setItem('apiUrl', API_URL);
    }

    if (newUserId) {
        USER_ID = newUserId;
        localStorage.setItem('userId', USER_ID);
        userIdDisplay.textContent = USER_ID;
    }

    closeModal('settingsModal');
    showNotification('Settings saved!', 'success');
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    let restoreBtn = document.getElementById('restoreSidebarBtn');

    sidebar.classList.toggle('hidden');

    // Show/hide restore button
    if (sidebar.classList.contains('hidden')) {
        if (!restoreBtn) {
            createRestoreSidebarButton();
        } else {
            restoreBtn.style.display = 'flex';
        }
    } else {
        if (restoreBtn) {
            restoreBtn.style.display = 'none';
        }
    }
}

function createRestoreSidebarButton() {
    const restoreBtn = document.createElement('button');
    restoreBtn.id = 'restoreSidebarBtn';
    restoreBtn.className = 'restore-sidebar-btn';
    restoreBtn.innerHTML = '📊';
    restoreBtn.title = 'Show Session Info';
    restoreBtn.onclick = toggleSidebar;

    // Add inline styles to ensure visibility
    restoreBtn.style.cssText = `
        position: fixed !important;
        left: 20px !important;
        bottom: 100px !important;
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        border: none !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 24px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
        z-index: 9999 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    `;

    document.body.appendChild(restoreBtn);
}

// Modal Functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.add('active');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('active');
}

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

// Notification Function
function showNotification(message, type = 'info') {
    // Simple console notification - can be enhanced with a toast library
    console.log(`[${type.toUpperCase()}] ${message}`);

    // You could add a toast notification here
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#4a90e2'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 2000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animations to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
