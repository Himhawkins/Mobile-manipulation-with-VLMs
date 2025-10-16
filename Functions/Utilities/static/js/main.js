// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const calibrateBtn = document.getElementById('calibrate-btn');
    const executeBtn = document.getElementById('execute-btn');
    const promptEntry = document.getElementById('prompt-entry');
    const refreshPortsBtn = document.getElementById('refresh-ports-btn');
    const serialMenu = document.getElementById('serial-menu');
    const statusMessage = document.getElementById('status-message');

    // --- Helper Functions ---
    /** Displays a status message at the bottom of the control panel. */
    function showStatus(message, type = 'success') {
        statusMessage.textContent = message;
        statusMessage.className = `status-message status-${type}`;
        setTimeout(() => {
            statusMessage.textContent = '';
            statusMessage.className = 'status-message';
        }, 4000);
    }

    /** Updates the text and style of the execute button based on state. */
    function updateExecuteButton(isExecuting) {
        if (isExecuting) {
            executeBtn.textContent = 'Stop';
            executeBtn.classList.add('stop');
        } else {
            executeBtn.textContent = 'Execute';
            executeBtn.classList.remove('stop');
        }
    }

    /** Fetches available serial ports and populates the dropdown. */
    async function fetchPorts() {
        try {
            const response = await fetch('/api/ports');
            const ports = await response.json();
            serialMenu.innerHTML = ''; // Clear existing options
            if (ports.length > 0) {
                ports.forEach(port => {
                    const option = document.createElement('option');
                    option.value = port;
                    option.textContent = port;
                    serialMenu.appendChild(option);
                });
            } else {
                const option = document.createElement('option');
                option.textContent = 'No Ports Found';
                serialMenu.appendChild(option);
            }
        } catch (error) {
            console.error('Error fetching ports:', error);
            showStatus('Could not fetch serial ports.', 'error');
        }
    }

    // --- Event Listeners ---
    calibrateBtn.addEventListener('click', async () => {
        const prompt = promptEntry.value;
        if (!prompt) {
            showStatus('Calibration prompt cannot be empty.', 'error');
            return;
        }

        calibrateBtn.disabled = true;
        calibrateBtn.textContent = 'Calibrating...';
        
        try {
            const response = await fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt }),
            });
            const result = await response.json();
            if (response.ok) {
                showStatus(result.message, 'success');
            } else {
                showStatus(result.message, 'error');
            }
        } catch (error) {
            console.error('Calibration error:', error);
            showStatus('Failed to send calibration request.', 'error');
        } finally {
            calibrateBtn.disabled = false;
            calibrateBtn.textContent = 'Calibrate Arena';
        }
    });

    executeBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/execute', { method: 'POST' });
            const result = await response.json();
            if (response.ok) {
                updateExecuteButton(result.is_executing);
                showStatus(`Execution ${result.status}.`, 'success');
            } else {
                updateExecuteButton(false); // Reset button on error
                showStatus(result.message, 'error');
            }
        } catch (error) {
            console.error('Execute error:', error);
            showStatus('Failed to send execute command.', 'error');
        }
    });

    refreshPortsBtn.addEventListener('click', fetchPorts);

    // --- Initial Load ---
    fetchPorts();
    // The video stream starts automatically from the <img> src attribute.
    // The Agent management functionality can be loaded here as well.
});