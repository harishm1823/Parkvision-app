// ParkVision JavaScript

// Handle image upload
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const fpsDisplay = document.getElementById('fps-display');

    // Initialize with default values
    updateCounts(0, 0, 0);

    // Click to upload
    if (uploadArea) {
        uploadArea.addEventListener('click', () => {
            imageInput.click();
        });
    }

    // Handle file selection
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                uploadImage(file);
            }
        });
    }

    // Drag and drop
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = '#f0f8ff';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = '';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                uploadImage(file);
            }
        });
    }

    function uploadImage(file) {
        if (fpsDisplay) {
            fpsDisplay.textContent = 'Analyzing image...';
        }
        
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update the counts in real-time
                updateCounts(data.empty || 0, data.occupied || 0, data.total || 0);
                
                // Show detection summary
                if (data.total > 0) {
                    const availability = data.empty > 0 ? Math.round((data.empty / data.total) * 100) : 0;
                    if (fpsDisplay) {
                        fpsDisplay.textContent = `Found ${data.total} vehicles (${availability}% spaces available)`;
                    }
                } else {
                    if (fpsDisplay) {
                        fpsDisplay.textContent = 'No vehicles detected - all spaces available';
                    }
                    // If no vehicles detected, assume all spaces are empty
                    updateCounts(10, 0, 10); // Mock some empty spaces
                }
            } else {
                if (fpsDisplay) {
                    fpsDisplay.textContent = 'Detection failed: ' + (data.error || 'Unknown error');
                }
                updateCounts(0, 0, 0);
            }
        })
        .catch(error => {
            if (fpsDisplay) {
                fpsDisplay.textContent = 'Error: ' + error.message;
            }
            updateCounts(0, 0, 0);
        });
    }

    function updateCounts(empty, occupied, total) {
        // Update count displays
        const emptyEl = document.getElementById('empty-count');
        const occupiedEl = document.getElementById('occupied-count');
        const totalEl = document.getElementById('total-count');
        
        if (emptyEl) emptyEl.textContent = empty;
        if (occupiedEl) occupiedEl.textContent = occupied;
        if (totalEl) totalEl.textContent = total;
        
        // Calculate and update availability
        const availability = total > 0 ? Math.round((empty / total) * 100) : 0;
        
        const availabilityBar = document.querySelector('.availability-fill');
        const availabilityText = document.getElementById('availability-text');
        
        if (availabilityBar) {
            availabilityBar.style.width = availability + '%';
            
            // Change color based on availability
            if (availability > 70) {
                availabilityBar.style.backgroundColor = '#4CAF50'; // Green
            } else if (availability > 30) {
                availabilityBar.style.backgroundColor = '#FF9800'; // Orange
            } else {
                availabilityBar.style.backgroundColor = '#F44336'; // Red
            }
        }
        
        if (availabilityText) {
            availabilityText.textContent = availability + '% Available';
        }
    }

    // Make updateCounts available globally
    window.updateCounts = updateCounts;
});