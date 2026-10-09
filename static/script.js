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
                e.target.value = ''; // allow re-selecting the same file
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
                const empty = data.empty || 0;
                const occupied = data.occupied || 0;
                const total = data.total || 0;

                // Update the counts in real-time
                updateCounts(empty, occupied, total);

                // Show the image with detection boxes
                showResult(data.annotated_image);

                // Show detection summary (vehicles = occupied spaces)
                const availability = total > 0 ? Math.round((empty / total) * 100) : 0;
                if (fpsDisplay) {
                    if (occupied > 0) {
                        fpsDisplay.textContent = `Found ${occupied} vehicle${occupied === 1 ? '' : 's'} (${availability}% spaces available)`;
                    } else {
                        fpsDisplay.textContent = 'No vehicles detected - all spaces available';
                    }
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

    function showResult(src) {
        if (!uploadArea || !src) return;
        let img = document.getElementById('resultImage');
        if (!img) {
            img = document.createElement('img');
            img.id = 'resultImage';
            img.alt = 'Detection result';
            img.title = 'Click to upload another image';
            img.style.cssText = 'display:block;max-width:100%;max-height:600px;margin:0 auto;border-radius:8px;';
            uploadArea.appendChild(img);
        }
        // Hide the upload prompt while the result is shown
        Array.from(uploadArea.children).forEach(child => {
            if (child !== img) child.style.display = 'none';
        });
        img.src = src;
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