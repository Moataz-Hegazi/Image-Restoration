// Configuration
const API_BASE_URL = 'http://localhost:8000';
const COLORIZE_ENDPOINT = '/colorize';

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const resultsSection = document.getElementById('resultsSection');
const originalImage = document.getElementById('originalImage');
const colorizedImage = document.getElementById('colorizedImage');
const loadingSpinner = document.getElementById('loadingSpinner');
const colorizedWrapper = document.getElementById('colorizedWrapper');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

let selectedFile = null;
let colorizedImageData = null;

// ===== Event Listeners =====

// Upload box interactions
uploadBox.addEventListener('click', () => imageInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

uploadBtn.addEventListener('click', uploadAndColorize);
downloadBtn.addEventListener('click', downloadImage);
resetBtn.addEventListener('click', resetUI);

// ===== Functions =====

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
        showError('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        uploadBtn.style.display = 'inline-flex';
    };
    reader.readAsDataURL(file);
}

/**
 * Upload and colorize the image
 */
async function uploadAndColorize() {
    if (!selectedFile) {
        showError('No file selected');
        return;
    }

    try {
        // Show results section with loading spinner
        resultsSection.style.display = 'block';
        colorizedImage.style.display = 'none';
        loadingSpinner.style.display = 'flex';
        uploadBtn.disabled = true;

        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request
        const response = await fetch(`${API_BASE_URL}${COLORIZE_ENDPOINT}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Get the image blob
        const blob = await response.blob();
        colorizedImageData = blob;

        // Display colorized image
        const colorizedUrl = URL.createObjectURL(blob);
        colorizedImage.src = colorizedUrl;
        colorizedImage.style.display = 'block';
        loadingSpinner.style.display = 'none';

        uploadBtn.disabled = false;
    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to colorize image: ${error.message}`);
        loadingSpinner.style.display = 'none';
        uploadBtn.disabled = false;
        resultsSection.style.display = 'none';
    }
}

/**
 * Download the colorized image
 */
function downloadImage() {
    if (!colorizedImageData) {
        showError('No colorized image to download');
        return;
    }

    const url = URL.createObjectURL(colorizedImageData);
    const link = document.createElement('a');
    link.href = url;
    link.download = `colorized_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Reset the UI
 */
function resetUI() {
    selectedFile = null;
    colorizedImageData = null;
    originalImage.src = '';
    colorizedImage.src = '';
    imageInput.value = '';
    resultsSection.style.display = 'none';
    uploadBtn.style.display = 'none';
    uploadBtn.disabled = false;
    loadingSpinner.style.display = 'none';
    colorizedImage.style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    
    // Auto-hide after 5 seconds
    setTimeout(closeError, 5000);
}

/**
 * Close error message
 */
function closeError() {
    errorMessage.style.display = 'none';
}

// ===== Initialization =====

console.log('Image Colorization Frontend loaded');
console.log(`API Base URL: ${API_BASE_URL}`);
